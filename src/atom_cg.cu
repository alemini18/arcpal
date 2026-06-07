// atom_cg.cu — Atom-level propagation, block-per-rule deduction, cooperative grid
//
// Uses reverse tables for incremental updates. init_sums launched separately,
// then a persistent cooperative kernel handles the update+deduce loop.

#include "../include/cuda_utils.cuh"
#include "../include/propagation_common.cuh"
#include "../include/reduction.cuh"
#include "../include/parser.hpp"
#include "../include/printer.hpp"
#include "../include/reverse_tables.hpp"

namespace cg = cooperative_groups;


__global__ void init_sums_kernel(
    const int* M, const int* rule_offsets, const int* flat_lits, const int* flat_weights,
    int* S_sat, int* S_undef, int* updated_rules, int num_rules
) {
    int rule_id = blockIdx.x;
    if (rule_id >= num_rules) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx   = rule_offsets[rule_id + 1];

    extern __shared__ int shared_mem[];
    int* shared_sat   = shared_mem;
    int* shared_undef = &shared_mem[blockDim.x];

    int partial_sat   = 0;
    int partial_undef = 0;

    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
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

    block_reduce_sums(shared_sat, shared_undef, partial_sat, partial_undef);

    if (threadIdx.x == 0) {
        S_sat[rule_id]        = shared_sat[0];
        S_undef[rule_id]      = shared_undef[0];
        updated_rules[rule_id] = 1;
    }
}


__global__ void kernel(
    int* M,
    const int* atom_body_offsets, const int* atom_body_rules,
    const int* atom_body_lits, const int* atom_body_weights,
    const int* atom_head_offsets, const int* atom_head_rules,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int* S_sat_arr, int* S_undef_arr, int* updated_rules,
    int num_rules, int* contradiction,
    int* queue_0, int* queue_1, int* num_out, int* num_in, int* swap_flag
) {
    cg::grid_group grid = cg::this_grid();

    int* queue_out = queue_0;
    int rule_id = blockIdx.x;

    if (*contradiction) return;

    // ── First deduce ─────────────────────────────────────────────────────
    if (rule_id < num_rules && updated_rules[rule_id] != 0) {
        if (threadIdx.x == 0) {
            updated_rules[rule_id] = 0;
        }

        int start_idx = rule_offsets[rule_id];
        int end_idx   = rule_offsets[rule_id + 1];
        int B         = bound[rule_id];

        int s_sat_val  = S_sat_arr[rule_id];
        int s_undef_val = S_undef_arr[rule_id];
        int S_max      = s_sat_val + s_undef_val;

        int h_lit     = head[rule_id];
        int h_atom    = abs(h_lit);
        int h_val     = lit_sat_value(h_lit);
        int h_not_val = lit_unsat_value(h_lit);

        if (threadIdx.x == 0) {
            if (s_sat_val >= B) {
                atomicAssignAndQueue(M, h_atom, h_val, contradiction, queue_out, num_out);
            } else if (S_max < B) {
                atomicAssignAndQueue(M, h_atom, h_not_val, contradiction, queue_out, num_out);
            }
        }
        __syncthreads();

        int h_val_cur = M[h_atom];
        if (h_val_cur != UNDEF) {
            bool h_sat = literal_is_satisfied(h_lit, h_val_cur);

            for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
                int lit    = flat_lits[i];
                int atom   = abs(lit);
                int weight = flat_weights[i];

                if (M[atom] == UNDEF) {
                    if (h_sat) {
                        if (S_max - weight < B) {
                            atomicAssignAndQueue(M, atom, lit_sat_value(lit), contradiction, queue_out, num_out);
                        }
                    } else {
                        if (s_sat_val + weight >= B) {
                            atomicAssignAndQueue(M, atom, lit_unsat_value(lit), contradiction, queue_out, num_out);
                        }
                    }
                }
            }
        }
    }

    grid.sync();

    bool flag = true;
    while (flag) {
        int* queue_in = (*swap_flag == 1) ? queue_0 : queue_1;
        queue_out     = (*swap_flag == 1) ? queue_1 : queue_0;

        // ── Update sums ──────────────────────────────────────────────────
        int idx = grid.thread_rank();
        if (idx < *num_in) {
            int atom  = queue_in[idx];
            int m_val = M[atom];

            int body_start = atom_body_offsets[atom];
            int body_end   = atom_body_offsets[atom + 1];

            for (int i = body_start; i < body_end; i++) {
                int r_id   = atom_body_rules[i];
                int lit    = atom_body_lits[i];
                int weight = atom_body_weights[i];

                atomicSub(&S_undef_arr[r_id], weight);
                if (literal_is_satisfied(lit, m_val)) {
                    atomicAdd(&S_sat_arr[r_id], weight);
                }
                updated_rules[r_id] = 1;
            }

            int head_start = atom_head_offsets[atom];
            int head_end   = atom_head_offsets[atom + 1];
            for (int i = head_start; i < head_end; i++) {
                int r_id = atom_head_rules[i];
                updated_rules[r_id] = 1;
            }
        }

        grid.sync();
        if (*contradiction) break;

        // ── Deduce ───────────────────────────────────────────────────────
        if (rule_id < num_rules && updated_rules[rule_id] != 0) {
            __syncthreads();
            if (threadIdx.x == 0) {
                updated_rules[rule_id] = 0;
            }

            int start_idx = rule_offsets[rule_id];
            int end_idx   = rule_offsets[rule_id + 1];
            int B         = bound[rule_id];

            int s_sat_val  = S_sat_arr[rule_id];
            int s_undef_val = S_undef_arr[rule_id];
            int S_max      = s_sat_val + s_undef_val;

            int h_lit     = head[rule_id];
            int h_atom    = abs(h_lit);
            int h_val     = lit_sat_value(h_lit);
            int h_not_val = lit_unsat_value(h_lit);

            if (threadIdx.x == 0) {
                if (s_sat_val >= B) {
                    atomicAssignAndQueue(M, h_atom, h_val, contradiction, queue_out, num_out);
                } else if (S_max < B) {
                    atomicAssignAndQueue(M, h_atom, h_not_val, contradiction, queue_out, num_out);
                }
            }
            __syncthreads();

            int h_val_cur = M[h_atom];
            if (h_val_cur != UNDEF) {
                bool h_sat = literal_is_satisfied(h_lit, h_val_cur);

                for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
                    int lit    = flat_lits[i];
                    int atom   = abs(lit);
                    int weight = flat_weights[i];

                    if (M[atom] == UNDEF) {
                        if (h_sat) {
                            if (S_max - weight < B) {
                                atomicAssignAndQueue(M, atom, lit_sat_value(lit), contradiction, queue_out, num_out);
                            }
                        } else {
                            if (s_sat_val + weight >= B) {
                                atomicAssignAndQueue(M, atom, lit_unsat_value(lit), contradiction, queue_out, num_out);
                            }
                        }
                    }
                }
            }
        }

        grid.sync();
        if (grid.thread_rank() == 0) {
            if (*num_out == 0 || *contradiction != 0) {
                flag = false;
            } else {
                *swap_flag = 1 - *swap_flag;
                *num_in    = *num_out;
                *num_out   = 0;
            }
        }
        grid.sync();
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

    CudaBuffer<int> d_queue_0((size_t)input.num_atoms);
    CudaBuffer<int> d_queue_1((size_t)input.num_atoms);
    CudaBuffer<int> d_num_out(1);
    CudaBuffer<int> d_num_in(1);
    CudaBuffer<int> d_swap_flag(1);
    CudaBuffer<int> d_contradiction(1);

    const int THREADS_PER_BLOCK = 256;
    int blocks_per_grid = input.num_rules;
    int shared_mem_size = THREADS_PER_BLOCK * 2 * sizeof(int);

    // ── Init sums ────────────────────────────────────────────────────────
    init_sums_kernel<<<blocks_per_grid, THREADS_PER_BLOCK, shared_mem_size>>>(
        d_M.ptr(), d_rule_offsets.ptr(), d_flat_lits.ptr(), d_flat_weights.ptr(),
        d_S_sat.ptr(), d_S_undef.ptr(), d_updated_rules.ptr(), input.num_rules
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Launch persistent kernel ─────────────────────────────────────────
    int* p_M      = d_M.ptr();
    int* p_abo    = d_atom_body_offsets.ptr();
    int* p_abr    = d_atom_body_rules.ptr();
    int* p_abl    = d_atom_body_lits.ptr();
    int* p_abw    = d_atom_body_weights.ptr();
    int* p_aho    = d_atom_head_offsets.ptr();
    int* p_ahr    = d_atom_head_rules.ptr();
    int* p_head   = d_head.ptr();
    int* p_bound  = d_bound.ptr();
    int* p_roff   = d_rule_offsets.ptr();
    int* p_lits   = d_flat_lits.ptr();
    int* p_wts    = d_flat_weights.ptr();
    int* p_ssat   = d_S_sat.ptr();
    int* p_sund   = d_S_undef.ptr();
    int* p_upd    = d_updated_rules.ptr();
    int* p_ctr    = d_contradiction.ptr();
    int* p_q0     = d_queue_0.ptr();
    int* p_q1     = d_queue_1.ptr();
    int* p_nout   = d_num_out.ptr();
    int* p_nin    = d_num_in.ptr();
    int* p_swap   = d_swap_flag.ptr();

    void* args[] = {
        &p_M,
        &p_abo, &p_abr, &p_abl, &p_abw,
        &p_aho, &p_ahr,
        &p_head, &p_bound, &p_roff, &p_lits, &p_wts,
        &p_ssat, &p_sund, &p_upd,
        &input.num_rules, &p_ctr,
        &p_q0, &p_q1, &p_nout, &p_nin, &p_swap
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)kernel,
        blocks_per_grid, THREADS_PER_BLOCK,
        args
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_contradiction = d_contradiction.download_scalar();
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