// rule_cg.cu — Rule-level propagation, block-per-rule, cooperative grid persistent kernel
//
// One cooperative kernel launch. Each block handles one rule.
// Grid sync replaces host-loop iteration.

#include "../include/cuda_utils.cuh"
#include "../include/propagation_common.cuh"
#include "../include/reduction.cuh"
#include "../include/parser.hpp"
#include "../include/printer.hpp"

namespace cg = cooperative_groups;

__global__ void kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int num_rules, int* changed, int* contradiction
) {
    cg::grid_group grid = cg::this_grid();
    int rule_id = blockIdx.x;

    extern __shared__ int shared_mem[];
    int* shared_sat   = shared_mem;
    int* shared_undef = &shared_mem[blockDim.x];

    bool flag = true;

    while (flag) {
        // Reset changed flag
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *changed = 0;
        }
        grid.sync();

        if (rule_id < num_rules) {
            int start_idx = rule_offsets[rule_id];
            int end_idx   = rule_offsets[rule_id + 1];
            int B         = bound[rule_id];

            // ── Compute S_sat and S_undef ─────────────────────────────────
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

            int S_sat  = shared_sat[0];
            int S_undef = shared_undef[0];
            int S_max  = S_sat + S_undef;

            int h_lit     = head[rule_id];
            int h_atom    = abs(h_lit);
            int h_val     = lit_sat_value(h_lit);
            int h_not_val = lit_unsat_value(h_lit);

            // ── Body → Head ───────────────────────────────────────────────
            if (threadIdx.x == 0) {
                if (S_sat >= B) {
                    atomicAssign(M, h_atom, h_val, contradiction, changed);
                } else if (S_max < B) {
                    atomicAssign(M, h_atom, h_not_val, contradiction, changed);
                }
            }
            __syncthreads();

            // ── Head → Body ───────────────────────────────────────────────
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

    const int THREADS_PER_BLOCK = 256;
    int blocks_per_grid = input.num_rules;
    int sharedMemSize   = THREADS_PER_BLOCK * 2 * sizeof(int);

    void* kernelArgs[] = {
        (void*)&d_M,       (void*)&d_head,      (void*)&d_bound,
        (void*)&d_rule_offsets, (void*)&d_flat_lits, (void*)&d_flat_weights,
        (void*)&input.num_rules, (void*)&d_changed, (void*)&d_contradiction
    };

    // CudaBuffer stores the pointer internally — we need raw pointers for args
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
        (void*)kernel,
        dim3(blocks_per_grid), dim3(THREADS_PER_BLOCK),
        args, sharedMemSize, 0
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