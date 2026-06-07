#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 

namespace cg = cooperative_groups;

__device__ void atomicAssign(int* M, int atom_id, int val, int* contradiction, int* changed) {
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
    int num_rules, int* changed, int* contradiction
) {
    
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int B = bound[rule_id];

    bool flag = true;
    while(flag){

    if(blockIdx.x == 0 && threadIdx.x == 0){
        *changed = 0;
    }
    grid.sync();

    if(rule_id < num_rules){

        int partial_S_sat = 0;
        int partial_S_undef = 0;

        for (int i = start_idx + tile.thread_rank(); i < end_idx; i += tile.size()) {
            int lit = flat_lits[i];
            int atom = abs(lit);
            int weight = flat_weights[i]; 

            int m_val = M[atom];
            
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

        int h_lit = head[rule_id];
        int h_atom = abs(h_lit);
        int h_val = (h_lit > 0) ? TRUE : FALSE;
        int h_not_val = (h_lit > 0) ? FALSE : TRUE;

        // Body -> Head
        if (tile.thread_rank() == 0) {
            if (S_sat >= B) { 
                atomicAssign(M, h_atom, h_val, contradiction, changed);
            } else if (S_max < B) { 
                atomicAssign(M, h_atom, h_not_val, contradiction, changed);
            }
        }

        tile.sync();

        // Head -> Body
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
                            atomicAssign(M, atom, lit_val, contradiction, changed);
                        }
                    } else { 
                        if (S_sat + weight >= B) { 
                            atomicAssign(M, atom, lit_not_val, contradiction, changed);
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

bool host(DIMACSInput& input) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_lits, *d_flat_weights;
    int *d_changed, *d_contradiction;

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

    const int TILE_SIZE = 16; 
    const int THREADS_PER_BLOCK = 256; 
    
    int tiles_per_block = THREADS_PER_BLOCK / TILE_SIZE; 
    int blocks_per_grid = (input.num_rules + tiles_per_block - 1) / tiles_per_block;

    void* kernelArgs[] = {
    (void*)&d_M,
    (void*)&d_head,
    (void*)&d_bound,
    (void*)&d_rule_offsets,
    (void*)&d_flat_lits,
    (void*)&d_flat_weights,
    (void*)&input.num_rules,
    (void*)&d_changed,
    (void*)&d_contradiction
};

    cudaLaunchCooperativeKernel(
        kernel<TILE_SIZE>,
        dim3(blocks_per_grid), dim3(THREADS_PER_BLOCK),
        kernelArgs,
        0, 0
    );

    cudaDeviceSynchronize();

    int h_contradiction;

    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_contradiction, d_contradiction,sizeof(int), cudaMemcpyDeviceToHost);

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
    bool contradiction = host(input);
    print_structure(input,contradiction);

}