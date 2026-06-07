// rule_packed.cu — Rule-level propagation, tile-per-rule (packed), host-loop iteration
//
// Multiple rules per block using cooperative groups tiles (warp-level).
// Shuffle-based reduction. Host loops kernel launches until convergence.

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
    if (*contradiction) return;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();
    if (rule_id >= num_rules) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx   = rule_offsets[rule_id + 1];
    int B         = bound[rule_id];

    // ── Compute S_sat and S_undef via tile shuffle reduction ──────────────
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

    int h_lit     = head[rule_id];
    int h_atom    = abs(h_lit);
    int h_val     = lit_sat_value(h_lit);
    int h_not_val = lit_unsat_value(h_lit);

    // ── Body → Head ──────────────────────────────────────────────────────
    if (tile.thread_rank() == 0) {
        if (S_sat >= B) {
            atomicAssign(M, h_atom, h_val, contradiction, changed);
        } else if (S_max < B) {
            atomicAssign(M, h_atom, h_not_val, contradiction, changed);
        }
    }
    tile.sync();

    // ── Head → Body ──────────────────────────────────────────────────────
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

    int h_changed = 1, h_contradiction = 0;

    while (h_changed == 1 && h_contradiction == 0) {
        d_changed.zero();

        kernel<TILE_SIZE><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
            d_M.ptr(), d_head.ptr(), d_bound.ptr(), d_rule_offsets.ptr(),
            d_flat_lits.ptr(), d_flat_weights.ptr(),
            input.num_rules, d_changed.ptr(), d_contradiction.ptr()
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        h_changed       = d_changed.download_scalar();
        h_contradiction = d_contradiction.download_scalar();
    }

    d_M.download(input.M);
    return h_contradiction;
}

int main() {
    DIMACSInput input = parse_dimacs_input();
    int contradiction = host(input);
    print_structure(input, contradiction);
    return 0;
}