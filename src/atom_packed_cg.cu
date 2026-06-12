#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h> 
#include <nvtx3/nvtx3.hpp>

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 
#include "../include/reverse_tables.hpp"

namespace cg = cooperative_groups;


__device__ void atomic_assign_and_queue(int* M, int atom, int val, int* contradiction, int* queue, int* q_size) {
    int old_val = atomicCAS(&M[atom], UNDEF, val);
    if (old_val == UNDEF) {
        int idx = atomicAdd(q_size, 1);
        queue[idx] = atom;
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
__device__ void update_sums_kernel(
    int idx, const int* modified_atoms,
    const int* M,
    const int* atom_body_offsets, const int* atom_body_rules, const int* atom_body_lits, const int* atom_body_weights,
    const int* atom_head_offsets, const int* atom_head_rules,
    int* S_sat, int* S_undef, int* updated_rules
) {
    int atom = modified_atoms[idx];
    int m_val = M[atom]; //No UNDEF

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

template <int TILE_SIZE>
__device__ void deduce_kernel(
    int rule_id, int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int* S_sat_global, int* S_undef_global, int* updated_rules,
     int* contradiction, int* queue, int* q_size
) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    if (*contradiction) return;

    if (updated_rules[rule_id] == 0) return;
    tile.sync();

    if (tile.thread_rank() == 0) {
        updated_rules[rule_id] = 0;
    }

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int B = bound[rule_id];

    int S_sat = S_sat_global[rule_id];
    int S_undef = S_undef_global[rule_id];
    int S_max = S_sat + S_undef;

    int h_lit = head[rule_id];
    int h_atom = abs(h_lit);
    int h_val = (h_lit > 0) ? TRUE : FALSE;
    int h_not_val = (h_lit > 0) ? FALSE : TRUE;

    if (tile.thread_rank() == 0) {
        if (S_sat >= B) { 
            atomic_assign_and_queue(M, h_atom, h_val, contradiction, queue, q_size);
        } else if (S_max < B) { 
            atomic_assign_and_queue(M, h_atom, h_not_val, contradiction, queue, q_size);
        }
        h_val = M[h_atom];
    }
    tile.sync();

    h_val = tile.shfl(h_val,0);
    
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
                        atomic_assign_and_queue(M, atom, lit_val, contradiction, queue, q_size);
                    }
                } else { 
                    if (S_sat + weight >= B) { 
                        atomic_assign_and_queue(M, atom, lit_not_val, contradiction, queue, q_size);
                    }
                }
            }
        }
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
    int* queue, int* q_size
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);
        
    if (*contradiction) return;
    
    int total_tiles = gridDim.x * tile.meta_group_size();

    while (true) {

        for(int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();
            rule_id < num_rules; rule_id += total_tiles){
            
            deduce_kernel<TILE_SIZE>(rule_id, M, head, bound, rule_offsets,
            flat_lits, flat_weights, S_sat, S_undef,
            updated_rules, contradiction,
            queue, q_size);
        }

        grid.sync();
        if (*q_size == 0 || *contradiction) break;

        for(int idx = grid.thread_rank(); idx < *q_size; idx += grid.size()){

            update_sums_kernel(idx,
                queue,
                M,
                atom_body_offsets, atom_body_rules, atom_body_lits, atom_body_weights,
                atom_head_offsets, atom_head_rules,
                S_sat, S_undef, updated_rules);
        }

        grid.sync();
        if (*contradiction) break;
        if(grid.thread_rank() == 0)*q_size = 0;
        grid.sync();
    }
}

int host(DIMACSInput& input, ReverseTables& revt) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_lits, *d_flat_weights;
    int *d_atom_body_offsets, *d_atom_body_rules, *d_atom_body_lits, *d_atom_body_weights;
    int *d_atom_head_offsets, *d_atom_head_rules;
    int *d_S_sat, *d_S_undef, *d_updated_rules;
    int *d_queue, *d_q_size, *d_contradiction;

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

    cudaMalloc(&d_queue, input.num_atoms * sizeof(int));
    
    cudaMalloc(&d_q_size, sizeof(int));
    cudaMalloc(&d_contradiction, sizeof(int));

    cudaMemset(d_contradiction, 0, sizeof(int));
    cudaMemset(d_q_size, 0, sizeof(int));
    
    
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
        &d_queue, 
        &d_q_size, 
    };

    cudaError_t launch_err = cudaLaunchCooperativeKernel(
        (void*)kernel<TILE_SIZE>, 
        blocks_per_grid, 
        THREADS_PER_BLOCK,
        kernel_args
    );

    if (launch_err != cudaSuccess) {
        cerr<<"Kernel Launch Error: "<<cudaGetErrorString(launch_err)<<endl;
    }
    cudaDeviceSynchronize();

    int h_contradiction = 2;
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
    cudaFree(d_queue);
    cudaFree(d_q_size);
    cudaFree(d_contradiction);

    return h_contradiction;
}

int main() {

    DIMACSInput input = parse_dimacs_input();
    ReverseTables revt;
    {
    nvtx3::scoped_range marker("build_reverse_tables");
    build_reverse_tables(input, revt);
    }
    int contradiction = host(input, revt);
    print_structure(input, contradiction);

}