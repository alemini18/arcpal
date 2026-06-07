// rule_packed_cg.cu — Rule-level propagation, tile-per-rule (packed), cooperative grid
//
// Multiple rules per block using cooperative groups tiles.
// Persistent kernel with grid sync. Shuffle-based reduction.

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
    int num_rules, int* changed, int* contradiction
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int rule_id   = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();
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

    bool flag = true;
    while (flag) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *changed = 0;
        }
        grid.sync();

        if (rule_id < num_rules) {
            // ── Compute S_sat and S_undef ─────────────────────────────────
            int partial_sat   = 0;
            int partial_undef = 0;

            for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                int lit    = flat_lits[i];
                int atom   = abs(lit);
                int weight = flat_weights[i];
                int m_val  = M[atom];

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

            // ── Body → Head ───────────────────────────────────────────────
            if (tile.thread_rank() == 0) {
                if (S_sat >= B) {
                    atomicAssign(M, h_atom, h_val, contradiction, changed);
                } else if (S_max < B) {
                    atomicAssign(M, h_atom, h_not_val, contradiction, changed);
                }
            }
            tile.sync();

            // ── Head → Body ───────────────────────────────────────────────
            int h_val_cur = M[h_atom];
            if (h_val_cur != UNDEF) {
                bool h_sat = literal_is_satisfied(h_lit, h_val_cur);

                for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                    int lit    = flat_lits[i];
                    int atom   = abs(lit);
                    int weight = flat_weights[i];

                    if (M[atom] == UNDEF) {
                        if (h_sat) {
                            if (S_max - weight < B) {
                                atomicAssign(M, atom, lit_sat_value(lit), contradiction, changed);
                            }
                        } else {
                            if (S_sat + weight >= B) {
                                atomicAssign(M, atom, lit_unsat_value(lit), contradiction, changed);
                            }
                        }
                    }
                }
            }
        }

        grid.sync();
        if (*changed == 0 || *contradiction == 1) {
            flag = false;
        }
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
        &input.num_rules, &p_chg, &p_ctr
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)kernel<TILE_SIZE>,
        dim3(blocks_per_grid), dim3(THREADS_PER_BLOCK),
        args, 0, 0
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