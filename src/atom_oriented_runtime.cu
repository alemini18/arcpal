#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 

struct ReverseTables{
    std::vector<int> atom_body_offsets;
    std::vector<int> flat_atom_body_rules;
    std::vector<int> flat_atom_body_lits;
    std::vector<int> flat_atom_body_weights;

    std::vector<int> atom_head_offsets;
    std::vector<int> flat_atom_head_rules;
};

void build_reverse_tables(PropagatorInput& input, ReverseTables& revt) {
    int num_atoms = input.num_atoms;
    int num_rules = input.num_rules;

    // Atoms in DIMACS typically range from 1 to num_atoms.
    // We size the offset arrays to num_atoms + 2 to safely index atom_id 
    // and hold the final boundary offset at [num_atoms + 1].
    revt.atom_body_offsets.assign(num_atoms + 2, 0);
    revt.atom_head_offsets.assign(num_atoms + 2, 0);

    // --- Step 1: Count occurrences of each atom ---
    for (int r = 0; r < num_rules; ++r) {
        // Count head occurrences
        int h_lit = input.head[r];
        if (h_lit != 0) { // Assuming 0 is not a valid literal
            int h_atom = std::abs(h_lit);
            revt.atom_head_offsets[h_atom + 1]++;
        }

        // Count body occurrences
        int start = input.rule_offsets[r];
        int end = input.rule_offsets[r + 1];
        for (int i = start; i < end; ++i) {
            int lit = input.flat_literals[i];
            int atom = std::abs(lit);
            revt.atom_body_offsets[atom + 1]++;
        }
    }

    // --- Step 2: Prefix Sums (Exclusive Scan) ---
    for (int a = 1; a <= num_atoms + 1; ++a) {
        revt.atom_body_offsets[a] += revt.atom_body_offsets[a - 1];
        revt.atom_head_offsets[a] += revt.atom_head_offsets[a - 1];
    }

    // --- Step 3: Allocate flat arrays ---
    revt.flat_atom_body_rules.resize(revt.atom_body_offsets.back());
    revt.flat_atom_body_lits.resize(revt.atom_body_offsets.back());
    revt.flat_atom_body_weights.resize(revt.atom_body_offsets.back());
    
    revt.flat_atom_head_rules.resize(revt.atom_head_offsets.back());

    // --- Step 4: Populate flat arrays ---
    // We copy the offset arrays to use them as sliding insertion pointers
    std::vector<int> current_body_offset = revt.atom_body_offsets;
    std::vector<int> current_head_offset = revt.atom_head_offsets;

    for (int r = 0; r < num_rules; ++r) {
        // Populate head
        int h_lit = input.head[r];
        if (h_lit != 0) {
            int h_atom = std::abs(h_lit);
            int h_idx = current_head_offset[h_atom]++;
            revt.flat_atom_head_rules[h_idx] = r;
        }

        // Populate body
        int start = input.rule_offsets[r];
        int end = input.rule_offsets[r + 1];
        for (int i = start; i < end; ++i) {
            int lit = input.flat_literals[i];
            int atom = std::abs(lit);
            int weight = input.flat_weights[i];

            int b_idx = current_body_offset[atom]++;
            revt.flat_atom_body_rules[b_idx] = r;
            revt.flat_atom_body_lits[b_idx] = lit;
            revt.flat_atom_body_weights[b_idx] = weight;
        }
    }
}

// ----------------------------------------------------------------------------
// Device Functions
// ----------------------------------------------------------------------------

__device__ void atomicDeduceAtomQueue(int* M, int atom_id, int deduced_val, int* contradiction, int* queue_out, int* num_out) {
    // Attempt to write the deduced value. If it was UNDEF, we succeeded.
    int old_val = atomicCAS(&M[atom_id], UNDEF, deduced_val);
    
    if (old_val == UNDEF) {
        // We are the first to deduce this atom. Add it to the next-step queue.
        int idx = atomicAdd(num_out, 1);
        queue_out[idx] = atom_id;
    } else if (old_val != deduced_val) {
        // Contradiction detected
        *contradiction = 1; 
    }
}

// ----------------------------------------------------------------------------
// Phase 1: Initialization Kernel
// ----------------------------------------------------------------------------
// Calculates the initial S_sat and S_undef for each rule and marks all rules 
// as touched so they are evaluated in the very first iteration.
__global__ void init_S_kernel(
    const int* M, const int* rule_offsets, const int* flat_literals, const int* flat_weights,
    int* S_sat, int* S_undef, int* touched_rules, int num_rules
) {
    int rule_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rule_id >= num_rules) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];

    int partial_sat = 0;
    int partial_undef = 0;

    for (int i = start_idx; i < end_idx; i++) {
        int lit = flat_literals[i];
        int weight = flat_weights[i];
        int atom = abs(lit);
        int m_val = M[atom];

        if (m_val == UNDEF) {
            partial_undef += weight;
        } else if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
            partial_sat += weight;
        }
    }

    S_sat[rule_id] = partial_sat;
    S_undef[rule_id] = partial_undef;
    touched_rules[rule_id] = 1; // Mark for Phase 3
}

// ----------------------------------------------------------------------------
// Phase 2: Propagation Kernel
// ----------------------------------------------------------------------------
// Processes atoms modified in the previous step. Updates S_sat and S_undef globally
// and marks affected rules as touched.
__global__ void propagate_modifications_kernel(
    const int* modified_atoms, int num_modified,
    const int* M,
    const int* atom_body_offsets, const int* flat_atom_body_rules, const int* flat_atom_body_lits, const int* flat_atom_body_weights,
    const int* atom_head_offsets, const int* flat_atom_head_rules,
    int* S_sat, int* S_undef, int* touched_rules
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_modified) return;

    int atom = modified_atoms[idx];
    int m_val = M[atom]; // Will be TRUE or FALSE, not UNDEF

    // 1. Process rules where the atom appears in the Body
    int body_start = atom_body_offsets[atom];
    int body_end = atom_body_offsets[atom + 1];
    
    for (int i = body_start; i < body_end; i++) {
        int rule_id = flat_atom_body_rules[i];
        int lit = flat_atom_body_lits[i];
        int weight = flat_atom_body_weights[i];

        // The atom is no longer UNDEF, so subtract its weight
        atomicSub(&S_undef[rule_id], weight);
        
        // If the assignment satisfies the literal, add its weight to S_sat
        if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
            atomicAdd(&S_sat[rule_id], weight);
        }
        
        // Mark rule to be evaluated
        touched_rules[rule_id] = 1;
    }

    // 2. Process rules where the atom is the Head (doesn't change S_sat/S_undef, but triggers evaluation)
    int head_start = atom_head_offsets[atom];
    int head_end = atom_head_offsets[atom + 1];
    
    for (int i = head_start; i < head_end; i++) {
        int rule_id = flat_atom_head_rules[i];
        touched_rules[rule_id] = 1;
    }
}

// ----------------------------------------------------------------------------
// Phase 3: Evaluation Kernel
// ----------------------------------------------------------------------------
// Reads the touched rules, checks for bound triggers, and pushes newly deduced
// atoms to the next iteration queue.
__global__ void evaluate_deductions_kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_literals, const int* flat_weights,
    int* S_sat_global, int* S_undef_global, int* touched_rules,
    int num_rules, int* contradiction, int* queue_out, int* num_out
) {
    int rule_id = blockIdx.x;
    if (rule_id >= num_rules || *contradiction) return;

    // Fast return if this rule was not affected by recent changes
    if (touched_rules[rule_id] == 0) return;

    // Thread 0 clears the flag
    if (threadIdx.x == 0) {
        touched_rules[rule_id] = 0;
    }

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int num_literals = end_idx - start_idx;
    int B = bound[rule_id];

    int S_sat = S_sat_global[rule_id];
    int S_undef = S_undef_global[rule_id];
    int S_max = S_sat + S_undef;

    int h_lit = head[rule_id];
    int h_atom = abs(h_lit);
    int h_val = (h_lit > 0) ? TRUE : FALSE;
    int h_not_val = (h_lit > 0) ? FALSE : TRUE;

    // Body -> Head inference (Only 1 thread needs to do this per rule)
    if (threadIdx.x == 0) {
        if (S_sat >= B) { 
            atomicDeduceAtomQueue(M, h_atom, h_val, contradiction, queue_out, num_out);
        } else if (S_max < B) { 
            atomicDeduceAtomQueue(M, h_atom, h_not_val, contradiction, queue_out, num_out);
        }
    }
    __syncthreads();

    // Head -> Body inference (Parallelized over literals)
    int h_val_cur = M[h_atom];
    
    if (h_val_cur != UNDEF) {
        bool h_sat = ((h_lit > 0) && h_val_cur == TRUE) || ((h_lit < 0) && h_val_cur == FALSE);

        for (int i = threadIdx.x; i < num_literals; i += blockDim.x) {
            int lit_idx = start_idx + i;
            int lit = flat_literals[lit_idx];
            int atom = abs(lit);
            int weight = flat_weights[lit_idx];
            
            int lit_val = (lit > 0) ? TRUE : FALSE;
            int lit_not_val = (lit > 0) ? FALSE : TRUE;

            if (M[atom] == UNDEF) {
                if (h_sat) { 
                    if (S_max - weight < B) { 
                        atomicDeduceAtomQueue(M, atom, lit_val, contradiction, queue_out, num_out);
                    }
                } else { 
                    if (S_sat + weight >= B) { 
                        atomicDeduceAtomQueue(M, atom, lit_not_val, contradiction, queue_out, num_out);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Host Execution Function
// ----------------------------------------------------------------------------
bool run_propagation_atom_oriented(PropagatorInput& input, ReverseTables& revt) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_literals, *d_flat_weights;
    int *d_atom_body_offsets, *d_flat_atom_body_rules, *d_flat_atom_body_lits, *d_flat_atom_body_weights;
    int *d_atom_head_offsets, *d_flat_atom_head_rules;
    int *d_S_sat, *d_S_undef, *d_touched_rules;
    int *d_queue_in, *d_queue_out, *d_num_out, *d_contradiction;

    // 1. Allocate & Copy Forward CSR rules
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

    // 2. Allocate & Copy Reverse CSR Tables
    cudaMalloc(&d_atom_body_offsets, revt.atom_body_offsets.size() * sizeof(int));
    cudaMemcpy(d_atom_body_offsets, revt.atom_body_offsets.data(), revt.atom_body_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_flat_atom_body_rules, revt.flat_atom_body_rules.size() * sizeof(int));
    cudaMemcpy(d_flat_atom_body_rules, revt.flat_atom_body_rules.data(), revt.flat_atom_body_rules.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_flat_atom_body_lits, revt.flat_atom_body_lits.size() * sizeof(int));
    cudaMemcpy(d_flat_atom_body_lits, revt.flat_atom_body_lits.data(), revt.flat_atom_body_lits.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_flat_atom_body_weights, revt.flat_atom_body_weights.size() * sizeof(int));
    cudaMemcpy(d_flat_atom_body_weights, revt.flat_atom_body_weights.data(), revt.flat_atom_body_weights.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_atom_head_offsets, revt.atom_head_offsets.size() * sizeof(int));
    cudaMemcpy(d_atom_head_offsets, revt.atom_head_offsets.data(), revt.atom_head_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_flat_atom_head_rules, revt.flat_atom_head_rules.size() * sizeof(int));
    cudaMemcpy(d_flat_atom_head_rules, revt.flat_atom_head_rules.data(), revt.flat_atom_head_rules.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Allocate Tracking Variables
    cudaMalloc(&d_S_sat, input.num_rules * sizeof(int));
    cudaMalloc(&d_S_undef, input.num_rules * sizeof(int));
    cudaMalloc(&d_touched_rules, input.num_rules * sizeof(int));

    // Queues size bounded by memory limit / total number of atoms
    cudaMalloc(&d_queue_in, input.num_atoms * sizeof(int));
    cudaMalloc(&d_queue_out, input.num_atoms * sizeof(int));
    
    cudaMalloc(&d_num_out, sizeof(int));
    cudaMalloc(&d_contradiction, sizeof(int));

    cudaMemset(d_contradiction, 0, sizeof(int));

    // --- EXECUTION PIPELINE ---
    
    int h_contradiction = 0;
    int h_num_out = 0;
    
    // Phase 1: Init (Calculate base sums and trigger all rules for first evaluate)
    int init_blocks = (input.num_rules + 255) / 256;
    init_S_kernel<<<init_blocks, 256>>>(
        d_M, d_rule_offsets, d_flat_literals, d_flat_weights, 
        d_S_sat, d_S_undef, d_touched_rules, input.num_rules
    );
    cudaDeviceSynchronize();

    // Initial Phase 3: Evaluate everything based on the init values
    cudaMemset(d_num_out, 0, sizeof(int));
    evaluate_deductions_kernel<<<input.num_rules, 256>>>(
        d_M, d_head, d_bound, d_rule_offsets, d_flat_literals, d_flat_weights,
        d_S_sat, d_S_undef, d_touched_rules, input.num_rules, d_contradiction, d_queue_out, d_num_out
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&h_num_out, d_num_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);

    // Main Loop: Iterate to Fixed Point
    while (h_num_out > 0 && h_contradiction == 0) {
        
        // Swap queues: previous round's output becomes this round's input
        int* temp = d_queue_in;
        d_queue_in = d_queue_out;
        d_queue_out = temp;
        
        int h_num_in = h_num_out;
        cudaMemset(d_num_out, 0, sizeof(int)); // Reset output counter

        // Phase 2: Propagate
        int prop_blocks = (h_num_in + 255) / 256;
        propagate_modifications_kernel<<<prop_blocks, 256>>>(
            d_queue_in, h_num_in, d_M,
            d_atom_body_offsets, d_flat_atom_body_rules, d_flat_atom_body_lits, d_flat_atom_body_weights,
            d_atom_head_offsets, d_flat_atom_head_rules,
            d_S_sat, d_S_undef, d_touched_rules
        );
        cudaDeviceSynchronize();

        // Phase 3: Evaluate
        evaluate_deductions_kernel<<<input.num_rules, 256>>>(
            d_M, d_head, d_bound, d_rule_offsets, d_flat_literals, d_flat_weights,
            d_S_sat, d_S_undef, d_touched_rules, input.num_rules, d_contradiction, d_queue_out, d_num_out
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_num_out, d_num_out, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Copy Final Truth Assignments Back to Host
    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Free all device memory
    cudaFree(d_M); cudaFree(d_head); cudaFree(d_bound); cudaFree(d_rule_offsets); 
    cudaFree(d_flat_literals); cudaFree(d_flat_weights);
    cudaFree(d_atom_body_offsets); cudaFree(d_flat_atom_body_rules); cudaFree(d_flat_atom_body_lits); cudaFree(d_flat_atom_body_weights);
    cudaFree(d_atom_head_offsets); cudaFree(d_flat_atom_head_rules);
    cudaFree(d_S_sat); cudaFree(d_S_undef); cudaFree(d_touched_rules);
    cudaFree(d_queue_in); cudaFree(d_queue_out); cudaFree(d_num_out); cudaFree(d_contradiction);

    return h_contradiction;
}

int main() {
    try {
        PropagatorInput input = parse_dimacs_input();
        ReverseTables revt;
        build_reverse_tables(input, revt);
        bool contradiction = run_propagation_atom_oriented(input,revt);
        print_structure(input);
    } catch (const std::exception& e) {
        std::cerr << "\nEccezione catturata: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}