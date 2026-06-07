#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h> 

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 
#include "../include/reverse_tables.hpp"

namespace cg = cooperative_groups;


__device__ void atomicAssignAndQueue(int* M, int atom, int val, int* contradiction, int* queue_out, int* num_out) {
    int old_val = atomicCAS(&M[atom], UNDEF, val);
    if (old_val == UNDEF) {
        int idx = atomicAdd(num_out, 1);
        queue_out[idx] = atom;
    } else if (old_val != val) {
        *contradiction = 1; 
    }
}


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
    int end_idx = rule_offsets[rule_id + 1];

    int partial_sat = 0;
    int partial_undef = 0;

    for (int i = start_idx + tile.thread_rank(); i < end_idx; i+=tile.size()) {
        int lit = flat_lits[i];
        int weight = flat_weights[i];
        int atom = abs(lit);
        int m_val = M[atom];

        if (m_val == UNDEF) {
            partial_undef += weight;
        } else if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
            partial_sat += weight;
        }
    }

    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        partial_sat += tile.shfl_down(partial_sat, offset);
        partial_undef += tile.shfl_down(partial_undef, offset);
    }

    if(tile.thread_rank() == 0){
        S_sat[rule_id] = partial_sat;
        S_undef[rule_id] = partial_undef;
        updated_rules[rule_id] = 1;
    }
}

template<int TILE_SIZE>
__global__ void kernel(
    int* M,
    const int* atom_body_offsets, const int* atom_body_rules, const int* atom_body_lits, const int* atom_body_weights,
    const int* atom_head_offsets, const int* atom_head_rules,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int* S_sat, int* S_undef, int* updated_rules,
    int num_rules, int* contradiction,
    int* queue_0, int* queue_1, int* num_out, int* num_in, int* swap_flag
) {
    cg::grid_group grid = cg::this_grid();
    
    int* queue_out = queue_0;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();

        if(rule_id < num_rules && !*contradiction && updated_rules[rule_id] != 0){
            tile.sync();
        if (tile.thread_rank() == 0) {
            updated_rules[rule_id] = 0;
        }
        int start_idx = rule_offsets[rule_id];
        int end_idx = rule_offsets[rule_id + 1];
        int B = bound[rule_id];

        int local_S_sat = S_sat[rule_id];
        int local_S_undef = S_undef[rule_id];
        int S_max = local_S_sat + local_S_undef;

        int h_lit = head[rule_id];
        int h_atom = abs(h_lit);
        int h_val = (h_lit > 0) ? TRUE : FALSE;
        int h_not_val = (h_lit > 0) ? FALSE : TRUE;

        if (tile.thread_rank() == 0) {
            if (local_S_sat >= B) { 
                atomicAssignAndQueue(M, h_atom, h_val, contradiction, queue_out, num_out);
            } else if (S_max < B) { 
                atomicAssignAndQueue(M, h_atom, h_not_val, contradiction, queue_out, num_out);
            }
        }
    
        tile.sync();
        h_val = M[h_atom];
        
        if (h_val != UNDEF) {
            bool h_sat = ((h_lit > 0) && h_val == TRUE) || ((h_lit < 0) && h_val == FALSE);

            for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                int lit = flat_lits[i];
                int atom = abs(lit);
                int weight = flat_weights[i];
                
                int lit_val = (lit > 0) ? TRUE : FALSE;
                int lit_not_val = (lit > 0) ? FALSE : TRUE;

                if (M[atom] == UNDEF) {
                    if (h_sat) { 
                        if (S_max - weight < B) { 
                            atomicAssignAndQueue(M, atom, lit_val, contradiction, queue_out, num_out);
                        }
                    } else { 
                        if (local_S_sat + weight >= B) { 
                            atomicAssignAndQueue(M, atom, lit_not_val, contradiction, queue_out, num_out);
                        }
                    }
                }
            }
        }
        }
    grid.sync();

    bool flag = true;
    
    while (flag) {

        int* queue_in  = (*swap_flag == 1) ? queue_0 : queue_1;
        queue_out      = (*swap_flag == 1) ? queue_1 : queue_0;

  
        int idx = grid.thread_rank();
        if (idx < *num_in){
            int atom = queue_in[idx];
            int m_val = M[atom]; 

            int body_start = atom_body_offsets[atom];
            int body_end = atom_body_offsets[atom + 1];
            
            for (int i = body_start; i < body_end; i++) {
                int rule_id = atom_body_rules[i];
                int lit = atom_body_lits[i];
                int weight = atom_body_weights[i];

                atomicSub(&S_undef[rule_id], weight);
                
                if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
                    atomicAdd(&S_sat[rule_id], weight);
                }
                
                updated_rules[rule_id] = 1;
            }

            int head_start = atom_head_offsets[atom];
            int head_end = atom_head_offsets[atom + 1];
            
            for (int i = head_start; i < head_end; i++) {
                int rule_id = atom_head_rules[i];
                updated_rules[rule_id] = 1;
            }
        
        }
        grid.sync();
        if (*contradiction) break;
        
        if (updated_rules[rule_id] != 0){
            tile.sync();
            if (tile.thread_rank() == 0) {
                updated_rules[rule_id] = 0;
            }

            int start_idx = rule_offsets[rule_id];
            int end_idx = rule_offsets[rule_id + 1];
            int B = bound[rule_id];

            int local_S_sat = S_sat[rule_id];
            int local_S_undef = S_undef[rule_id];
            int S_max = local_S_sat + local_S_undef;

            int h_lit = head[rule_id];
            int h_atom = abs(h_lit);
            int h_val = (h_lit > 0) ? TRUE : FALSE;
            int h_not_val = (h_lit > 0) ? FALSE : TRUE;

            if (tile.thread_rank() == 0) {
                if (local_S_sat >= B) { 
                    atomicAssignAndQueue(M, h_atom, h_val, contradiction, queue_out, num_out);
                } else if (S_max < B) { 
                    atomicAssignAndQueue(M, h_atom, h_not_val, contradiction, queue_out, num_out);
                }
            }
            tile.sync();

            h_val = M[h_atom];
            
            if (h_val != UNDEF) {
                bool h_sat = ((h_lit > 0) && h_val == TRUE) || ((h_lit < 0) && h_val == FALSE);

                for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
                    int lit = flat_lits[i];
                    int atom = abs(lit);
                    int weight = flat_weights[i];
                    
                    int lit_val = (lit > 0) ? TRUE : FALSE;
                    int lit_not_val = (lit > 0) ? FALSE : TRUE;

                    if (M[atom] == UNDEF) {
                        if (h_sat) { 
                            if (S_max - weight < B) { 
                                atomicAssignAndQueue(M, atom, lit_val, contradiction, queue_out, num_out);
                            }
                        } else { 
                            if (local_S_sat + weight >= B) { 
                                atomicAssignAndQueue(M, atom, lit_not_val, contradiction, queue_out, num_out);
                            }
                        }
                    }
                }
            }
        }
    
        grid.sync();
        if (grid.thread_rank() == 0) {
            if (*num_out == 0 || *contradiction != 0) {
                *num_in = 0; 
            } else {
                *swap_flag = 1 - *swap_flag;
                *num_in = *num_out;
                *num_out = 0;
            }
        }
        grid.sync();
    }
}

int host(DIMACSInput& input, ReverseTables& revt) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_lits, *d_flat_weights;
    int *d_atom_body_offsets, *d_atom_body_rules, *d_atom_body_lits, *d_atom_body_weights;
    int *d_atom_head_offsets, *d_atom_head_rules;
    int *d_S_sat, *d_S_undef, *d_updated_rules;
    int *d_queue_0, *d_queue_1, *d_num_out, *d_num_in, *d_swap_flag, *d_contradiction;

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

    cudaMalloc(&d_atom_body_offsets, revt.atom_body_offsets.size() * sizeof(int));
    cudaMemcpy(d_atom_body_offsets, revt.atom_body_offsets.data(), revt.atom_body_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_atom_body_rules, revt.atom_body_rules.size() * sizeof(int));
    cudaMemcpy(d_atom_body_rules, revt.atom_body_rules.data(), revt.atom_body_rules.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_atom_body_lits, revt.atom_body_lits.size() * sizeof(int));
    cudaMemcpy(d_atom_body_lits, revt.atom_body_lits.data(), revt.atom_body_lits.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_atom_body_weights, revt.atom_body_weights.size() * sizeof(int));
    cudaMemcpy(d_atom_body_weights, revt.atom_body_weights.data(), revt.atom_body_weights.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_atom_head_offsets, revt.atom_head_offsets.size() * sizeof(int));
    cudaMemcpy(d_atom_head_offsets, revt.atom_head_offsets.data(), revt.atom_head_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_atom_head_rules, revt.atom_head_rules.size() * sizeof(int));
    cudaMemcpy(d_atom_head_rules, revt.atom_head_rules.data(), revt.atom_head_rules.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_S_sat, input.num_rules * sizeof(int));
    cudaMalloc(&d_S_undef, input.num_rules * sizeof(int));
    cudaMalloc(&d_updated_rules, input.num_rules * sizeof(int));

    cudaMalloc(&d_queue_0, input.num_atoms * sizeof(int));
    cudaMalloc(&d_queue_1, input.num_atoms * sizeof(int));
    
    cudaMalloc(&d_num_out, sizeof(int));
    cudaMalloc(&d_num_in, sizeof(int));
    cudaMalloc(&d_swap_flag, sizeof(int));
    cudaMalloc(&d_contradiction, sizeof(int));

    cudaMemset(d_contradiction, 0, sizeof(int));
    cudaMemset(d_num_out, 0, sizeof(int));
    cudaMemset(d_num_in, 0, sizeof(int));
    
    cudaMemset(d_swap_flag, 0, sizeof(int));
    
    const int TILE_SIZE = 16;
    const int THREADS_PER_BLOCK = 256; 
    int tiles_per_block = THREADS_PER_BLOCK / TILE_SIZE; 
    int blocks_per_grid = (input.num_rules + tiles_per_block - 1) / tiles_per_block;

    init_sums_kernel<TILE_SIZE><<<blocks_per_grid, THREADS_PER_BLOCK>>>(
        d_M, d_rule_offsets, d_flat_lits, d_flat_weights, 
        d_S_sat, d_S_undef, d_updated_rules, input.num_rules
    );
    cudaDeviceSynchronize();    

    void* kernel_args[] = {
        &d_M,
        &d_atom_body_offsets,
        &d_atom_body_rules, 
        &d_atom_body_lits, 
        &d_atom_body_weights,
        &d_atom_head_offsets,
        &d_atom_head_rules,
        &d_head, 
        &d_bound, 
        &d_rule_offsets,
        &d_flat_lits, 
        &d_flat_weights,
        &d_S_sat, 
        &d_S_undef, 
        &d_updated_rules,
        &input.num_rules, 
        &d_contradiction,
        &d_queue_0, 
        &d_queue_1, 
        &d_num_out, 
        &d_num_in, 
        &d_swap_flag
    };

    cudaLaunchCooperativeKernel(
        (void*)kernel<TILE_SIZE>, 
        blocks_per_grid, 
        THREADS_PER_BLOCK,
        kernel_args
    );
    cudaDeviceSynchronize();

    int h_contradiction = 0;
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_head);
    cudaFree(d_bound);
    cudaFree(d_rule_offsets); 
    cudaFree(d_flat_lits);
    cudaFree(d_flat_weights);
    cudaFree(d_atom_body_offsets);
    cudaFree(d_atom_body_rules);
    cudaFree(d_atom_body_lits);
    cudaFree(d_atom_body_weights);
    cudaFree(d_atom_head_offsets);
    cudaFree(d_atom_head_rules);
    cudaFree(d_S_sat);
    cudaFree(d_S_undef);
    cudaFree(d_updated_rules);
    cudaFree(d_queue_0);
    cudaFree(d_queue_1);
    cudaFree(d_num_out);
    cudaFree(d_num_in);
    cudaFree(d_contradiction);
    cudaFree(d_swap_flag);

    return h_contradiction;
}

int main() {

    DIMACSInput input = parse_dimacs_input();
    ReverseTables revt;
    build_reverse_tables(input, revt);
    int contradiction = host(input, revt);
    print_structure(input, contradiction);

}