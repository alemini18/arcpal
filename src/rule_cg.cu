#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h> 
#include "../include/parser.hpp" 
#include "../include/printer.hpp"

namespace cg = cooperative_groups; 
using namespace std;


__device__ void atomicAssign(int* M, int atom, int val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom], UNDEF, val);
    if (old_val == UNDEF) {
        *changed = 1;
    } else if (old_val != val) {
        *contradiction = 1;
    }
}

__device__ void calc_sums(
    int* M,
    const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int* S_sat_shared_arr, int* S_undef_shared_arr,
    int* S_sat_shared, int* S_undef_shared, int start_idx, int end_idx
){
    
    int partial_S_sat = 0;
    int partial_S_undef = 0;

    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        int lit = flat_lits[i];
        int atom = abs(lit);
        int weight = flat_weights[i];

        int m_val = M[atom];
            
        if (m_val == UNDEF) {
            partial_S_undef += weight;
        } else if (((lit > 0) && m_val == TRUE) || ((lit < 0) && m_val == FALSE)) {
            partial_S_sat += weight;
        }
    }

    S_sat_shared_arr[threadIdx.x] = partial_S_sat;
    S_undef_shared_arr[threadIdx.x] = partial_S_undef;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset /= 2){
        if(threadIdx.x < offset){
            S_sat_shared_arr[threadIdx.x] += S_sat_shared_arr[threadIdx.x + offset];
            S_undef_shared_arr[threadIdx.x] += S_undef_shared_arr[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *S_sat_shared = S_sat_shared_arr[0];
        *S_undef_shared = S_undef_shared_arr[0];
    }
}

__device__ void deduce_head(
    int* M, int h_atom, int h_val, int h_not_val, int B,
    int S_sat, int S_max, int* h_val_shared,
    int* changed, int* contradiction
){
    if (S_sat >= B) { 
        atomicAssign(M, h_atom, h_val, contradiction, changed);
    } else if (S_max < B) { 
        atomicAssign(M, h_atom, h_not_val, contradiction, changed);
    }
}

__device__ void deduce_body(
    int* M, int B, bool h_sat, int* changed, int* contradiction,
    const int* flat_lits, const int* flat_weights, int start_idx, int end_idx,
    int S_max, int S_sat
){
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


__device__ void update_rule(
    int rule_id, int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int* S_sat_shared_arr, int* S_undef_shared_arr,
    int* S_sat_shared, int* S_undef_shared, int* h_val_shared,
    int* changed, int* contradiction
){
    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int B = bound[rule_id];

    calc_sums(
        M, rule_offsets, flat_lits, flat_weights, 
        S_sat_shared_arr, S_undef_shared_arr, S_sat_shared, 
        S_undef_shared, start_idx, end_idx
    );
    __syncthreads();
    
    int S_sat = *S_sat_shared;
    int S_undef = *S_undef_shared;
    int S_max = S_sat + S_undef;

    int h_lit = head[rule_id];
    int h_atom = abs(h_lit);
    int h_val = (h_lit > 0) ? TRUE : FALSE;
    int h_not_val = (h_lit > 0) ? FALSE : TRUE;

    if (threadIdx.x == 0) {
        deduce_head(M, h_atom, h_val, h_not_val, B, S_sat, S_max, changed, contradiction);
        *h_val_shared = M[h_atom];
    }
    __syncthreads();

    h_val = *h_val_shared;
    
    if (h_val != UNDEF) {
        bool h_sat = ((h_lit > 0) && h_val == TRUE) || ((h_lit < 0) && h_val == FALSE);
        deduce_body(M, B, h_sat, changed, contradiction, flat_lits, flat_weights, start_idx, end_idx, S_max, S_sat);
    }
}

__global__ void kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_lits, const int* flat_weights,
    int num_rules, int* changed, int* contradiction
) {
    cg::grid_group grid = cg::this_grid();

    __shared__ int S_sat_shared_arr[256];
    __shared__ int S_undef_shared_arr[256];
    __shared__ int S_sat_shared;
    __shared__ int S_undef_shared;
    __shared__ int h_val_shared;
    
    bool flag = true;
    
    while (flag) {

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *changed = 0;
        }
        grid.sync();

        for (int rule_id = blockIdx.x; rule_id < num_rules; rule_id += gridDim.x) {
            update_rule(
                rule_id, M, head, bound, rule_offsets,
                flat_lits, flat_weights,
                S_sat_shared_arr, S_undef_shared_arr,
                &S_sat_shared, &S_undef_shared, &h_val_shared,
                changed, contradiction
            );
        }
        grid.sync();

        if (*changed == 0 || *contradiction == 1) {
            flag = false;
        }
        grid.sync();
    }
}

int host(DIMACSInput& input) {
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
    
    cudaMemset(d_changed, 0, sizeof(int));
    cudaMemset(d_contradiction, 0, sizeof(int));

    const int THREADS_PER_BLOCK = 256; 
    
    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, (const void*)kernel, THREADS_PER_BLOCK, 0);
    int device_id = 0;
    cudaGetDevice(&device_id);
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);
    
    int blocks_per_grid = num_SMs * num_blocks_per_sm;
    if (blocks_per_grid > input.num_rules) {
        blocks_per_grid = input.num_rules;
    }
    if (blocks_per_grid == 0) blocks_per_grid = 1;

    void* kernel_args[] = {
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

    cudaError_t launch_err = cudaLaunchCooperativeKernel(
        kernel,
        dim3(blocks_per_grid), dim3(THREADS_PER_BLOCK),
        kernel_args,
        0, 0
    );
    if (launch_err != cudaSuccess) {
        cerr<<"Kernel Launch Error: "<<cudaGetErrorString(launch_err)<<endl;
    }

    cudaDeviceSynchronize();

    int h_contradiction;
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
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
    int contradiction = host(input);
    print_structure(input, contradiction);
}