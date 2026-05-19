#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cooperative_groups.h> 
#include "../include/parser.hpp" 
#include "../include/pretty_printer.hpp"

namespace cg = cooperative_groups; 


__device__ void atomicDeduce(int* M, int atom_id, int deduced_val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom_id], UNDEF, deduced_val);
    if (old_val == UNDEF) {
        *changed = 1;
    } else if (old_val != deduced_val) {
        *contradiction = 1; 
    }
}

__global__ void propagation_kernel_cooperative(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_literals, const int* flat_weights,
    int num_rules, int* global_changed, int* global_contradiction
) {
    cg::grid_group grid = cg::this_grid();
    
    int rule_id = blockIdx.x;

    __shared__ int S_sat_shared;
    __shared__ int S_undef_shared;
    
    bool flag = true;
    
    while (flag) {
        
        if (*global_contradiction) break;

        for (int current_rule_id = blockIdx.x; current_rule_id < num_rules; current_rule_id += gridDim.x) {
    
            int start_idx = rule_offsets[current_rule_id];
            int end_idx = rule_offsets[current_rule_id + 1];
            int num_literals = end_idx - start_idx;
            int B = bound[current_rule_id];

            if (threadIdx.x == 0) {
                S_sat_shared = 0;
                S_undef_shared = 0;
            }
            __syncthreads();

            int partial_S_sat = 0;
            int partial_S_undef = 0;

            for (int i = threadIdx.x; i < num_literals; i += blockDim.x) {
                int lit_idx = start_idx + i;
                int lit = flat_literals[lit_idx];
                int atom = abs(lit);
                int weight = flat_weights[lit_idx];

                int m_val = M[atom];
                
                if (m_val == UNDEF) {
                    partial_S_undef += weight;
                } else if (((lit > 0) && m_val == TRUE) || ((lit < 0) && m_val == FALSE)) {
                    partial_S_sat += weight;
                }
            }

            atomicAdd(&S_sat_shared, partial_S_sat);
            atomicAdd(&S_undef_shared, partial_S_undef);
            __syncthreads();

            int S_sat = S_sat_shared;
            int S_undef = S_undef_shared;
            int S_max = S_sat + S_undef;

            int h_lit = head[rule_id];
            int h_atom = abs(h_lit);
            int h_val = (h_lit > 0) ? TRUE : FALSE;
            int h_not_val = (h_lit > 0) ? FALSE : TRUE;

            // Body -> Head
            if (threadIdx.x == 0) {
                if (S_sat >= B) { 
                    atomicDeduce(M, h_atom, h_val, global_contradiction, global_changed);
                } else if (S_max < B) { 
                    atomicDeduce(M, h_atom, h_not_val, global_contradiction, global_changed);
                }
            }
            __syncthreads();

            // Head -> Body
            int h_val_cur = M[h_atom];
            
            if (h_val_cur != UNDEF) {
                bool h_sat = ((h_lit > 0) && h_val_cur == TRUE) || ((h_lit < 0) && h_val_cur == FALSE);
            
                for (int i = threadIdx.x; i < num_literals; i += blockDim.x) {
                    int lit_idx = start_idx + i;
                    int lit = flat_literals[lit_idx];
                    int atom = abs(lit);
                    int weight = flat_weights[lit_idx];
                    bool wants_true = (lit > 0);
                    
                    int lit_val_if_sat = wants_true ? TRUE : FALSE;
                    int lit_val_if_falsified = wants_true ? FALSE : TRUE;

                    if (M[atom] == UNDEF) {
                        if (h_sat) { 
                            if (S_max - weight < B) { 
                                atomicDeduce(M, atom, lit_val_if_sat, global_contradiction, global_changed);
                            }
                        } else {
                            if (S_sat + weight >= B) { 
                                atomicDeduce(M, atom, lit_val_if_falsified, global_contradiction, global_changed);
                            }
                        }
                    }
                }   
            }
        } 

        grid.sync();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (*global_changed == 0 || *global_contradiction == 1) {
                *global_changed = -1; 
            } else {
                *global_changed = 0; 
            }
        }

        grid.sync();

        if (*global_changed == -1) {
            flag = false;
        }
    }
}

int run_propagation(PropagatorInput& input) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_literals, *d_flat_weights;
    int *d_changed, *d_contradiction;

    cudaMalloc(&d_M, input.M.size() * sizeof(int));
    cudaMemcpy(d_M, input.M.data(), input.M.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_head, input.num_rules * sizeof(int));
    cudaMemcpy(d_head, input.head.data(), input.num_rules * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bound, input.num_rules * sizeof(int));
    cudaMemcpy(d_bound, input.bound.data(), input.num_rules * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rule_offsets, input.rule_offsets.size() * sizeof(int));
    cudaMemcpy(d_rule_offsets, input.rule_offsets.data(), input.rule_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_flat_literals, input.flat_literals.size() * sizeof(int));
    cudaMemcpy(d_flat_literals, input.flat_literals.data(), input.flat_literals.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_flat_weights, input.flat_weights.size() * sizeof(int));
    cudaMemcpy(d_flat_weights, input.flat_weights.data(), input.flat_weights.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_contradiction, sizeof(int));
    
    cudaMemset(d_changed, 0, sizeof(int));
    cudaMemset(d_contradiction, 0, sizeof(int));

    int threadsPerBlock = 256; 
    int numBlocks = 0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, 
        propagation_kernel_cooperative, 
        threadsPerBlock, 
        0
);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    int maxCooperativeBlocks = numBlocks * deviceProp.multiProcessorCount;

    int blocksPerGrid = (input.num_rules < maxCooperativeBlocks) ? input.num_rules : maxCooperativeBlocks;

    printf("Lancio Cooperative Kernel con %d blocchi (Max supportati: %d)\n", blocksPerGrid, maxCooperativeBlocks);

    void* kernelArgs[] = {
    (void*)&d_M,
    (void*)&d_head,
    (void*)&d_bound,
    (void*)&d_rule_offsets,
    (void*)&d_flat_literals,
    (void*)&d_flat_weights,
    (void*)&input.num_rules,
    (void*)&d_changed,
    (void*)&d_contradiction
};

    cudaError_t err = cudaLaunchCooperativeKernel(
        propagation_kernel_cooperative,
        dim3(blocksPerGrid), dim3(threadsPerBlock),
        kernelArgs,
        0, 0
    );

    if (err != cudaSuccess){
        std::cerr << "Errore CUDA Launch: " << cudaGetErrorString(err) << "\n";
        return 0;
    }

    cudaDeviceSynchronize();

    int h_contradiction;
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_M); cudaFree(d_head); cudaFree(d_bound); 
    cudaFree(d_rule_offsets); cudaFree(d_flat_literals); 
    cudaFree(d_flat_weights); cudaFree(d_changed); 
    cudaFree(d_contradiction);

    return h_contradiction;
}


int main() {


    try {
        PropagatorInput input = parse_dimacs_input();
        
        printf("Parsing completato: %d atomi, %d regole.\n", input.num_atoms, input.num_rules);

        bool contradiction = run_propagation(input);

        printf("SUCCESSO\n");

        pretty_print_structure(input);

    } catch (const std::exception& e) {
        std::cerr << "\nEccezione catturata: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}