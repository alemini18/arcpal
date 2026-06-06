#include <cuda_runtime.h>

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 

__device__ void atomicAssign(int* M, int atom, int val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom], UNDEF, val);
    if (old_val == UNDEF) {
        *changed = 1;
    } else if (old_val != val) {
        *contradiction = 1;
    }
}

__global__ void kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int num_rules, int* changed, int* contradiction
) {
    int rule_id = blockIdx.x;
    if (rule_id >= num_rules || *contradiction) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int B = bound[rule_id];

    extern __shared__ int shared_mem[];
    int* S_sat_shared = shared_mem;                      
    int* S_undef_shared = &shared_mem[blockDim.x];       

    S_sat_shared[threadIdx.x] = 0;
    S_undef_shared[threadIdx.x] = 0;
    __syncthreads();

    int partial_S_sat = 0;
    int partial_S_undef = 0;

    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
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

    S_sat_shared[threadIdx.x] = partial_S_sat;
    S_undef_shared[threadIdx.x] = partial_S_undef;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(threadIdx.x < offset){
        S_sat_shared[threadIdx.x] += S_sat_shared[threadIdx.x + offset];
        S_undef_shared[threadIdx.x] += S_undef_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    int S_sat = S_sat_shared[0];
    int S_undef = S_undef_shared[0];
    int S_max = S_sat + S_undef;

    int h_lit = head[rule_id];
    int h_atom = abs(h_lit);
    int h_val = (h_lit > 0) ? TRUE : FALSE;
    int h_not_val = (h_lit > 0) ? FALSE : TRUE;

    //  Body -> Head
    if (threadIdx.x == 0) {
        if (S_sat >= B) { 
            atomicAssign(M, h_atom, h_val, contradiction, changed);
        } else if (S_max < B) { 
            atomicAssign(M, h_atom, h_not_val, contradiction, changed);
        }
    }
    __syncthreads();

    // Head -> Body
    h_val = M[h_atom];

    if (h_val != UNDEF) {

        bool h_sat = ((h_lit > 0) && h_val == TRUE) || ((h_lit < 0) && h_val == FALSE);

        for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
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

    int threadsPerBlock = 256; 
    int blocksPerGrid = input.num_rules;
    int sharedMemSize = threadsPerBlock * 2 * sizeof(int);
    int h_changed = 1, h_contradiction = 0;

    while (h_changed == 1 && h_contradiction == 0) {
        cudaMemset(d_changed, 0, sizeof(int));

        kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_M, d_head, d_bound, d_rule_offsets, d_flat_lits, d_flat_weights,
            input.num_rules, d_changed, d_contradiction
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
    } 

    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

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

    return 0;
}