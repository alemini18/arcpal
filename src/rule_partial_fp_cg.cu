#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 

namespace cg = cooperative_groups;
using namespace std;

__device__ void atomic_assign(int* M, int atom_id, int val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom_id], UNDEF, val);
    if (old_val == UNDEF) {
        *changed = 1;
    } else if (old_val != val) {
        *contradiction = 1; 
    }
}

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

    int tiles_per_block = block.size() / TILE_SIZE; 
    int first_rule = blockIdx.x * tiles_per_block;
    int last_rule = min(first_rule + tiles_per_block, num_rules);
    
    int start_lit = 0;
    int end_lit = 0;
    
    if (first_rule < num_rules) {
        start_lit = rule_offsets[first_rule];
        end_lit = rule_offsets[last_rule];
    }

    int num_lits_block = end_lit - start_lit;

    extern __shared__ int shared_mem[];
    int* M_local = shared_mem;
    int* lits_local = &shared_mem[num_atoms + 1];
    int* weights_local = &shared_mem[num_atoms + 1 + num_lits_block];
    int* local_changed = &shared_mem[num_atoms + 1 + num_lits_block * 2];
    int* local_contradiction = &shared_mem[num_atoms + 1 + num_lits_block * 2 + 1];

    int rule_id = first_rule + tile.meta_group_rank();
    
    int start_idx = 0;
    int end_idx = 0;
    int B = 0;
    int h_lit = 0;
    int h_atom = 0;
    int h_val = 0;
    int h_not_val = 0;

    if (rule_id < num_rules) {
        start_idx = rule_offsets[rule_id];
        end_idx = rule_offsets[rule_id + 1];
        B = bound[rule_id];
        h_lit = head[rule_id];
        h_atom = abs(h_lit);
        h_val = (h_lit > 0) ? TRUE : FALSE;
        h_not_val = (h_lit > 0) ? FALSE : TRUE;
    }

    for (int i = start_lit + block.thread_rank(); i < end_lit; i += block.size()) {
        lits_local[i - start_lit] = flat_lits[i];
        weights_local[i - start_lit] = flat_weights[i];
    }
    block.sync();

    bool flag_global = true;
    while(flag_global) {

        if(blockIdx.x == 0 && threadIdx.x == 0) *changed = 0;
        grid.sync();

        for (int i = block.thread_rank(); i < num_atoms + 1; i += block.size()) {
            M_local[i] = M[i];
        }

        if (block.thread_rank() == 0) {
            *local_changed = 0;
            *local_contradiction = 0;
        }
        
        block.sync();

        bool flag_local = true;
        while(flag_local) {
            
            if (block.thread_rank() == 0) *local_changed = 0;
            block.sync();

            if (*local_contradiction) break;

            if (rule_id < num_rules) {
                int partial_S_sat = 0;
                int partial_S_undef = 0;

                for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                    
                    int lit = lits_local[i - start_lit];
                    int weight = weights_local[i - start_lit];
                    int atom = abs(lit);

                    int m_val = M_local[atom]; 
                    
                    if (m_val == UNDEF) {
                        partial_S_undef += weight;
                    } else if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
                        partial_S_sat += weight;
                    }
                }

                for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
                    partial_S_sat += tile.shfl_down(partial_S_sat, offset);
                    partial_S_undef += tile.shfl_down(partial_S_undef, offset);
                }
                int S_sat = tile.shfl(partial_S_sat, 0);
                int S_undef = tile.shfl(partial_S_undef, 0);
                int S_max = S_sat + S_undef;
                int h_assigned = UNDEF;

                // Body -> Head
                if (tile.thread_rank() == 0) {
                    if (S_sat >= B) { 
                        atomic_assign(M_local, h_atom, h_val, local_contradiction, local_changed);
                    } else if (S_max < B) { 
                        atomic_assign(M_local, h_atom, h_not_val, local_contradiction, local_changed);
                    }
                    h_assigned = M_local[h_atom];
                }
                
                tile.sync();

                // Head -> Body
                h_assigned = tile.shfl(h_assigned,0);
                if (h_assigned != UNDEF) {
                    bool h_sat = ((h_lit > 0) && h_assigned == TRUE) || ((h_lit < 0) && h_assigned == FALSE);

                    for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {                      
                        int lit = lits_local[i - start_lit];
                        int weight = weights_local[i - start_lit];
                        int atom = abs(lit);
                        
                        int lit_val = (lit > 0) ? TRUE : FALSE;
                        int lit_not_val = (lit > 0) ? FALSE : TRUE;

                        if (M_local[atom] == UNDEF) {
                            if (h_sat) { 
                                if (S_max - weight < B) { 
                                    atomic_assign(M_local, atom, lit_val, local_contradiction, local_changed);
                                }
                            } else { 
                                if (S_sat + weight >= B) { 
                                    atomic_assign(M_local, atom, lit_not_val, local_contradiction, local_changed);
                                }
                            }
                        }
                    }
                }
            }

            block.sync(); 
            if (*local_changed == 0 || *local_contradiction == 1) flag_local = false;
        }

        if(block.thread_rank() == 0 && *local_contradiction) *contradiction = 1;

        for (int i = block.thread_rank(); i < num_atoms + 1; i += block.size()) {
            if (M_local[i] != UNDEF) {
                atomic_assign(M, i, M_local[i], contradiction, changed);
            }
        }
        grid.sync();

        if (*changed == 0 || *contradiction == 1) flag_global = false;
    }
}

int host(DIMACSInput& input) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_lits, *d_flat_weights;
    int *d_changed, *d_contradiction;

    cudaFree(0); // Crea il contesto CUDA prima della regione misurata

    const int TILE_SIZE = 16; 
    const int THREADS_PER_BLOCK = 256; 
    
    int tiles_per_block = THREADS_PER_BLOCK / TILE_SIZE; 
    int blocks_per_grid = (input.num_rules + tiles_per_block - 1) / tiles_per_block;

    int max_lits_per_block = 0;
    for (int b = 0; b < blocks_per_grid; b++) {
        int first_rule = b * tiles_per_block;
        int last_rule = (first_rule + tiles_per_block < input.num_rules) ? first_rule + tiles_per_block : input.num_rules;
        int lits = input.rule_offsets[last_rule] - input.rule_offsets[first_rule];
        if (lits > max_lits_per_block) max_lits_per_block = lits;
    }
    
    int max_shared_mem = input.M.size() + max_lits_per_block + max_lits_per_block + 2;

    int device_id = 0;
    cudaGetDevice(&device_id);

    int max_shared_per_block;
    cudaDeviceGetAttribute(&max_shared_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    if (max_shared_mem * sizeof(int) > (size_t)max_shared_per_block) {
        cerr<<"Input too large: "<<max_shared_mem * sizeof(int)<<" bytes of shared memory per block required, "<<max_shared_per_block<<" available"<<endl;
        return 2;
    }

    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, (const void*)kernel<TILE_SIZE>, THREADS_PER_BLOCK, max_shared_mem * sizeof(int));
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

    // Ogni blocco lavora su regole fisse, quindi la griglia non puo' essere ridotta ai blocchi residenti
    if (blocks_per_grid > num_SMs * num_blocks_per_sm) {
        cerr<<"Input too large: "<<blocks_per_grid<<" blocks required, "<<num_SMs * num_blocks_per_sm<<" co-resident blocks available"<<endl;
        return 2;
    }

    cudaMalloc(&d_M, input.M.size() * sizeof(int));
    cudaMemcpy(d_M, input.M.data(), input.M.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_head, input.num_rules * sizeof(int));
    cudaMemcpy(d_head, input.head.data(), input.num_rules * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bound, input.num_rules * sizeof(int));
    cudaMemcpy(d_bound, input.bound.data(), input.num_rules * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rule_offsets, input.rule_offsets.size() * sizeof(int));
    cudaMemcpy(d_rule_offsets, input.rule_offsets.data(), input.rule_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_flat_lits, input.flat_lits.size() * sizeof(int));
    cudaMemcpy(d_flat_lits, input.flat_lits.data(), input.flat_lits.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_flat_weights, input.flat_weights.size() * sizeof(int));
    cudaMemcpy(d_flat_weights, input.flat_weights.data(), input.flat_weights.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_contradiction, sizeof(int));
    cudaMemset(d_contradiction, 0, sizeof(int));
    cudaMemset(d_changed, 0, sizeof(int));

    void* kernel_args[] = {
        (void*)&d_M,
        (void*)&d_head,
        (void*)&d_bound,
        (void*)&d_rule_offsets,
        (void*)&d_flat_lits,
        (void*)&d_flat_weights,
        (void*)&input.num_rules,
        (void*)&input.num_atoms,
        (void*)&d_changed,
        (void*)&d_contradiction
    };

    {
    nvtx3::scoped_range marker("fixpoint");
    cudaError_t launch_err = cudaLaunchCooperativeKernel(
        (const void*)kernel<TILE_SIZE>,
        dim3(blocks_per_grid), dim3(THREADS_PER_BLOCK),
        kernel_args,
        max_shared_mem * sizeof(int),
        0
    );

    if (launch_err != cudaSuccess) {
        cerr<<"Kernel Launch Error: "<< cudaGetErrorString(launch_err)<<endl;
    }

    cudaDeviceSynchronize();
    }

    int h_contradiction = 2;

    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_head);
    cudaFree(d_bound); 
    cudaFree(d_rule_offsets);
    cudaFree(d_flat_lits);
    cudaFree(d_flat_weights);
    cudaFree(d_changed);
    cudaFree(d_contradiction);

    return h_contradiction;
}

int main() {
    DIMACSInput input = parse_dimacs_input();
    int contradiction = host(input);
    print_structure(input,contradiction);

}