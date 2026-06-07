// atom_packed.cu — Atom-level propagation, tile-per-rule (packed), host-loop iteration
//
// Uses reverse tables for incremental updates. Tile-level shuffle reduction.
// Host loops update_sums + deduce kernels until convergence.

#include "../include/cuda_utils.cuh"
#include "../include/propagation_common.cuh"
#include "../include/reduction.cuh"
#include "../include/parser.hpp"
#include "../include/printer.hpp"
#include "../include/reverse_tables.hpp"

namespace cg = cooperative_groups;


template <int TILE_SIZE>
__global__ void init_sums_kernel(
    const int* M, const int* rule_offsets, const int* flat_lits, const int* flat_weights,
    int* S_sat, int* S_undef, int* updated_rules, int num_rules
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();
    if (rule_id >= num_rules) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx   = rule_offsets[rule_id + 1];

    int partial_sat   = 0;
    int partial_undef = 0;

    for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
        int lit    = flat_lits[i];
        int weight = flat_weights[i];
        int atom   = abs(lit);
        int m_val  = M[atom];

        if (m_val == UNDEF) {
            partial_undef += weight;
        } else if (literal_is_satisfied(lit, m_val)) {
            partial_sat += weight;
        }
    }

    tile_reduce_sums<TILE_SIZE>(tile, partial_sat, partial_undef);

    if (tile.thread_rank() == 0) {
        S_sat[rule_id]        = partial_sat;
        S_undef[rule_id]      = partial_undef;
        updated_rules[rule_id] = 1;
    }
}


__global__ void update_sums_kernel(
    const int* modified_atoms, int num_modified,
    const int* M,
    const int* atom_body_offsets, const int* atom_body_rules,
    const int* atom_body_lits, const int* atom_body_weights,
    const int* atom_head_offsets, const int* atom_head_rules,
    int* S_sat, int* S_undef, int* updated_rules
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_modified) return;

    int atom  = modified_atoms[idx];
    int m_val = M[atom];

    int body_start = atom_body_offsets[atom];
    int body_end   = atom_body_offsets[atom + 1];

    for (int i = body_start; i < body_end; i++) {
        int rule_id = atom_body_rules[i];
        int lit     = atom_body_lits[i];
        int weight  = atom_body_weights[i];

        atomicSub(&S_undef[rule_id], weight);
        if (literal_is_satisfied(lit, m_val)) {
            atomicAdd(&S_sat[rule_id], weight);
        }
        updated_rules[rule_id] = 1;
    }

    int head_start = atom_head_offsets[atom];
    int head_end   = atom_head_offsets[atom + 1];

    for (int i = head_start; i < head_end; i++) {
        int rule_id = atom_head_rules[i];
        updated_rules[rule_id] = 1;
    }
}

template <int TILE_SIZE>
__global__ void deduce_kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int* S_sat_global, int* S_undef_global, int* updated_rules,
    int num_rules, int* contradiction, int* queue_out, int* num_out
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();
    if (rule_id >= num_rules || *contradiction) return;
    if (updated_rules[rule_id] == 0) return;
    tile.sync();

    if (tile.thread_rank() == 0) {
        updated_rules[rule_id] = 0;
    }

    int start_idx = rule_offsets[rule_id];
    int end_idx   = rule_offsets[rule_id + 1];
    int B         = bound[rule_id];

    int S_sat  = S_sat_global[rule_id];
    int S_undef = S_undef_global[rule_id];
    int S_max  = S_sat + S_undef;

    int h_lit     = head[rule_id];
    int h_atom    = abs(h_lit);
    int h_val     = lit_sat_value(h_lit);
    int h_not_val = lit_unsat_value(h_lit);

    // ── Body → Head ──────────────────────────────────────────────────────
    if (tile.thread_rank() == 0) {
        if (S_sat >= B) {
            atomicAssignAndQueue(M, h_atom, h_val, contradiction, queue_out, num_out);
        } else if (S_max < B) {
            atomicAssignAndQueue(M, h_atom, h_not_val, contradiction, queue_out, num_out);
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
                        atomicAssignAndQueue(M, atom, lit_sat_value(lit), contradiction, queue_out, num_out);
                    }
                } else {
                    if (S_sat + weight >= B) {
                        atomicAssignAndQueue(M, atom, lit_unsat_value(lit), contradiction, queue_out, num_out);
                    }
                }
            }
        }
    }
}


int host(DIMACSInput& input, ReverseTables& revt) {
    CudaBuffer<int> d_M(input.M);
    CudaBuffer<int> d_head(input.head);
    CudaBuffer<int> d_bound(input.bound);
    CudaBuffer<int> d_rule_offsets(input.rule_offsets);
    CudaBuffer<int> d_flat_lits(input.flat_lits);
    CudaBuffer<int> d_flat_weights(input.flat_weights);

    CudaBuffer<int> d_atom_body_offsets(revt.atom_body_offsets);
    CudaBuffer<int> d_atom_body_rules(revt.atom_body_rules);
    CudaBuffer<int> d_atom_body_lits(revt.atom_body_lits);
    CudaBuffer<int> d_atom_body_weights(revt.atom_body_weights);
    CudaBuffer<int> d_atom_head_offsets(revt.atom_head_offsets);
    CudaBuffer<int> d_atom_head_rules(revt.atom_head_rules);

    CudaBuffer<int> d_S_sat(input.num_rules);
    CudaBuffer<int> d_S_undef(input.num_rules);
    CudaBuffer<int> d_updated_rules(input.num_rules);

    CudaBuffer<int> d_queue_in((size_t)input.num_atoms);
    CudaBuffer<int> d_queue_out((size_t)input.num_atoms);
    CudaBuffer<int> d_num_out(1);
    CudaBuffer<int> d_contradiction(1);

    const int TILE_SIZE = 16;
    const int THREADS_PER_BLOCK = 256;
    int tiles_per_block = THREADS_PER_BLOCK / TILE_SIZE;
    int blocks_per_grid = (input.num_rules + tiles_per_block - 1) / tiles_per_block;

    // ── Init sums ────────────────────────────────────────────────────────
    init_sums_kernel<TILE_SIZE><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
        d_M.ptr(), d_rule_offsets.ptr(), d_flat_lits.ptr(), d_flat_weights.ptr(),
        d_S_sat.ptr(), d_S_undef.ptr(), d_updated_rules.ptr(), input.num_rules
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── First deduce ─────────────────────────────────────────────────────
    d_num_out.zero();
    deduce_kernel<TILE_SIZE><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
        d_M.ptr(), d_head.ptr(), d_bound.ptr(), d_rule_offsets.ptr(),
        d_flat_lits.ptr(), d_flat_weights.ptr(),
        d_S_sat.ptr(), d_S_undef.ptr(), d_updated_rules.ptr(),
        input.num_rules, d_contradiction.ptr(), d_queue_out.ptr(), d_num_out.ptr()
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_num_out       = d_num_out.download_scalar();
    int h_contradiction = d_contradiction.download_scalar();

    // ── Propagation loop ─────────────────────────────────────────────────
    int* p_queue_in  = d_queue_in.ptr();
    int* p_queue_out = d_queue_out.ptr();

    while (h_num_out > 0 && h_contradiction == 0) {
        int* temp = p_queue_in;
        p_queue_in  = p_queue_out;
        p_queue_out = temp;

        int h_num_in = h_num_out;
        d_num_out.zero();

        int blocks = (h_num_in + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        update_sums_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            p_queue_in, h_num_in, d_M.ptr(),
            d_atom_body_offsets.ptr(), d_atom_body_rules.ptr(),
            d_atom_body_lits.ptr(), d_atom_body_weights.ptr(),
            d_atom_head_offsets.ptr(), d_atom_head_rules.ptr(),
            d_S_sat.ptr(), d_S_undef.ptr(), d_updated_rules.ptr()
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        deduce_kernel<TILE_SIZE><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
            d_M.ptr(), d_head.ptr(), d_bound.ptr(), d_rule_offsets.ptr(),
            d_flat_lits.ptr(), d_flat_weights.ptr(),
            d_S_sat.ptr(), d_S_undef.ptr(), d_updated_rules.ptr(),
            input.num_rules, d_contradiction.ptr(), p_queue_out, d_num_out.ptr()
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        h_num_out       = d_num_out.download_scalar();
        h_contradiction = d_contradiction.download_scalar();
    }

    d_M.download(input.M);
    return h_contradiction;
}

int main() {
    DIMACSInput input = parse_dimacs_input();
    ReverseTables revt;
    build_reverse_tables(input, revt);
    int contradiction = host(input, revt);
    print_structure(input, contradiction);
    return 0;
}