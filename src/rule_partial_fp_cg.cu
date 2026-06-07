// rule_partial_fp_cg.cu — Rule-level propagation, tile-per-rule (packed),
//                         cooperative grid with shared-memory local fixed point
//
// Each block copies M into shared memory, runs a local fixed-point loop
// on its assigned rules, then writes back to global M. The global loop
// continues until convergence across all blocks.

#include "../include/cuda_utils.cuh"
#include "../include/propagation_common.cuh"
#include "../include/reduction.cuh"
#include "../include/parser.hpp"
#include "../include/printer.hpp"

namespace cg = cooperative_groups;

template <int TILE_SIZE>
__global__ void kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int num_rules, int num_atoms,
    int* changed, int* contradiction
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int tilesPerBlock = block.size() / TILE_SIZE;
    int first_rule = blockIdx.x * tilesPerBlock;
    int last_rule  = min(first_rule + tilesPerBlock, num_rules);

    int start_lit = 0;
    int end_lit   = 0;

    if (first_rule < num_rules) {
        start_lit = rule_offsets[first_rule];
        end_lit   = rule_offsets[last_rule];
    }

    int num_lits_block = end_lit - start_lit;

    // Shared memory layout: M_local | lits_local | weights_local | local_changed | local_contradiction
    extern __shared__ int shared_mem[];
    int* M_local             = shared_mem;
    int* lits_local          = &shared_mem[num_atoms + 1];
    int* weights_local       = &shared_mem[num_atoms + 1 + num_lits_block];
    int* local_changed       = &shared_mem[num_atoms + 1 + num_lits_block * 2];
    int* local_contradiction = &shared_mem[num_atoms + 1 + num_lits_block * 2 + 1];

    int rule_id = first_rule + tile.meta_group_rank();

    int start_idx = 0, end_idx = 0, B = 0;
    int h_lit = 0, h_atom = 0, h_val = 0, h_not_val = 0;

    if (rule_id < num_rules) {
        start_idx = rule_offsets[rule_id];
        end_idx   = rule_offsets[rule_id + 1];
        B         = bound[rule_id];
        h_lit     = head[rule_id];
        h_atom    = abs(h_lit);
        h_val     = lit_sat_value(h_lit);
        h_not_val = lit_unsat_value(h_lit);
    }

    // Cache literals and weights into shared memory
    for (int i = start_lit + block.thread_rank(); i < end_lit; i += block.size()) {
        lits_local[i - start_lit]    = flat_lits[i];
        weights_local[i - start_lit] = flat_weights[i];
    }

    bool flag_global = true;
    while (flag_global) {
        // Reset global changed flag
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *changed = 0;
        }
        grid.sync();

        // Load global M into local copy
        for (int i = block.thread_rank(); i < num_atoms + 1; i += block.size()) {
            M_local[i] = M[i];
        }
        if (block.thread_rank() == 0) {
            *local_changed       = 0;
            *local_contradiction = 0;
        }
        block.sync();

        // ── Local fixed-point loop ────────────────────────────────────────
        bool flag_local = true;
        while (flag_local) {
            if (block.thread_rank() == 0) *local_changed = 0;
            block.sync();

            if (*local_contradiction) break;

            if (rule_id < num_rules) {
                int partial_sat   = 0;
                int partial_undef = 0;

                for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                    int lit    = lits_local[i - start_lit];
                    int weight = weights_local[i - start_lit];
                    int atom   = abs(lit);
                    int m_val  = M_local[atom];

                    if (m_val == UNDEF) {
                        partial_undef += weight;
                    } else if (literal_is_satisfied(lit, m_val)) {
                        partial_sat += weight;
                    }
                }

                tile_reduce_sums<TILE_SIZE>(tile, partial_sat, partial_undef);
                int S_sat  = tile.shfl(partial_sat, 0);
                int S_undef = tile.shfl(partial_undef, 0);
                int S_max  = S_sat + S_undef;

                // Body → Head
                if (tile.thread_rank() == 0) {
                    if (S_sat >= B) {
                        atomicAssign(M_local, h_atom, h_val, local_contradiction, local_changed);
                    } else if (S_max < B) {
                        atomicAssign(M_local, h_atom, h_not_val, local_contradiction, local_changed);
                    }
                }
                tile.sync();

                // Head → Body
                int h_val_cur = M_local[h_atom];
                if (h_val_cur != UNDEF) {
                    bool h_sat = literal_is_satisfied(h_lit, h_val_cur);

                    for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                        int lit    = lits_local[i - start_lit];
                        int weight = weights_local[i - start_lit];
                        int atom   = abs(lit);

                        if (M_local[atom] == UNDEF) {
                            if (h_sat) {
                                if (S_max - weight < B) {
                                    atomicAssign(M_local, atom, lit_sat_value(lit), local_contradiction, local_changed);
                                }
                            } else {
                                if (S_sat + weight >= B) {
                                    atomicAssign(M_local, atom, lit_unsat_value(lit), local_contradiction, local_changed);
                                }
                            }
                        }
                    }
                }
            }

            block.sync();
            if (*local_changed == 0 || *local_contradiction == 1) flag_local = false;
        }

        // Write local M back to global M
        for (int i = block.thread_rank(); i < num_atoms + 1; i += block.size()) {
            if (M_local[i] != UNDEF) {
                atomicAssign(M, i, M_local[i], contradiction, changed);
            }
        }
        grid.sync();

        if (*changed == 0 || *contradiction == 1) flag_global = false;
    }
}


int host(DIMACSInput& input) {
    CudaBuffer<int> d_M(input.M);
    CudaBuffer<int> d_head(input.head);
    CudaBuffer<int> d_bound(input.bound);
    CudaBuffer<int> d_rule_offsets(input.rule_offsets);
    CudaBuffer<int> d_flat_lits(input.flat_lits);
    CudaBuffer<int> d_flat_weights(input.flat_weights);
    CudaBuffer<int> d_changed(1);
    CudaBuffer<int> d_contradiction(1);

    const int TILE_SIZE = 16;
    const int THREADS_PER_BLOCK = 256;
    int tiles_per_block = THREADS_PER_BLOCK / TILE_SIZE;
    int blocks_per_grid = (input.num_rules + tiles_per_block - 1) / tiles_per_block;

    int max_lits_per_block = tiles_per_block * 16;
    int max_shared_mem = (int)input.M.size() + max_lits_per_block * 2 + 2;

    int* p_M     = d_M.ptr();
    int* p_head  = d_head.ptr();
    int* p_bound = d_bound.ptr();
    int* p_roff  = d_rule_offsets.ptr();
    int* p_lits  = d_flat_lits.ptr();
    int* p_wts   = d_flat_weights.ptr();
    int* p_chg   = d_changed.ptr();
    int* p_ctr   = d_contradiction.ptr();

    void* args[] = {
        &p_M, &p_head, &p_bound, &p_roff, &p_lits, &p_wts,
        &input.num_rules, &input.num_atoms, &p_chg, &p_ctr
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)kernel<TILE_SIZE>,
        dim3(blocks_per_grid), dim3(THREADS_PER_BLOCK),
        args, max_shared_mem * sizeof(int), 0
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_contradiction = d_contradiction.download_scalar();
    d_M.download(input.M);

    return h_contradiction;
}

int main() {
    DIMACSInput input = parse_dimacs_input();
    int contradiction = host(input);
    print_structure(input, contradiction);
    return 0;
}