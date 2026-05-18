#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// Importiamo le strutture e la firma del parser dal tuo file header
#include "../include/parser.hpp" 

// =========================================================================
// KERNEL E FUNZIONI DEVICE
// =========================================================================

__device__ void atomicDeduce(int* M, int atom_id, int deduced_val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom_id], UNDEF_VAL, deduced_val);
    if (old_val == UNDEF_VAL) {
        // Scrittura avvenuta con successo
        *changed = 1;
    } else if (old_val != deduced_val) {
        // Valore già presente ma complementare: contraddizione!
        *contradiction = 1; 
    }
}

__global__ void propagate_kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_literals, const int* flat_weights,
    int num_rules, int* changed, int* contradiction
) {
    int rule_id = blockIdx.x;
    if (rule_id >= num_rules || *contradiction) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int num_literals = end_idx - start_idx;
    int B = bound[rule_id];

    __shared__ int S_sat_shared;
    __shared__ int S_undef_shared;

    if (threadIdx.x == 0) {
        S_sat_shared = 0;
        S_undef_shared = 0;
    }
    __syncthreads();

    // FASE 1: Calcolo di S_sat e S_undef
    int local_S_sat = 0;
    int local_S_undef = 0;

    for (int i = threadIdx.x; i < num_literals; i += blockDim.x) {
        int lit_idx = start_idx + i;
        int lit = flat_literals[lit_idx];
        int atom = abs(lit);
        int weight = flat_weights[lit_idx];
        bool wants_true = (lit > 0); 

        int m_val = M[atom];
        
        if (m_val == UNDEF_VAL) {
            local_S_undef += weight;
        } else if ((wants_true && m_val == TRUE_VAL) || (!wants_true && m_val == FALSE_VAL)) {
            local_S_sat += weight;
        }
    }

    atomicAdd(&S_sat_shared, local_S_sat);
    atomicAdd(&S_undef_shared, local_S_undef);
    __syncthreads();

    int S_sat = S_sat_shared;
    int S_undef = S_undef_shared;
    int S_max = S_sat + S_undef;

    // Analisi della Head
    int h_lit = head[rule_id];
    int h_atom = abs(h_lit);
    bool h_wants_true = (h_lit > 0);
    int head_val_if_sat = h_wants_true ? TRUE_VAL : FALSE_VAL;
    int head_val_if_falsified = h_wants_true ? FALSE_VAL : TRUE_VAL;

    // FASE 2: Propagazione Body -> Head (Solo Thread 0)
    if (threadIdx.x == 0) {
        if (S_sat >= B) { 
            atomicDeduce(M, h_atom, head_val_if_sat, contradiction, changed);
        } else if (S_max < B) { 
            atomicDeduce(M, h_atom, head_val_if_falsified, contradiction, changed);
        }
    }
    __syncthreads();

    // FASE 3: Propagazione Head -> Body (Tutti i thread)
    int h_val_current = M[h_atom];
    int is_head_satisfied = 0; 
    
    if (h_val_current != UNDEF_VAL) {
        if ((h_wants_true && h_val_current == TRUE_VAL) || (!h_wants_true && h_val_current == FALSE_VAL)) {
            is_head_satisfied = 1;
        } else {
            is_head_satisfied = -1;
        }
    }

    for (int i = threadIdx.x; i < num_literals; i += blockDim.x) {
        int lit_idx = start_idx + i;
        int lit = flat_literals[lit_idx];
        int atom = abs(lit);
        int weight = flat_weights[lit_idx];
        bool wants_true = (lit > 0);
        
        int lit_val_if_sat = wants_true ? TRUE_VAL : FALSE_VAL;
        int lit_val_if_falsified = wants_true ? FALSE_VAL : TRUE_VAL;

        if (M[atom] == UNDEF_VAL) {
            if (is_head_satisfied == 1) { 
                if (S_max - weight < B) { 
                    atomicDeduce(M, atom, lit_val_if_sat, contradiction, changed);
                }
            } else if (is_head_satisfied == -1) { 
                if (S_sat + weight >= B) { 
                    atomicDeduce(M, atom, lit_val_if_falsified, contradiction, changed);
                }
            }
        }
    }
}

// =========================================================================
// FUNZIONE HOST: GESTIONE TRASFERIMENTI E LOOP
// =========================================================================

void run_propagation(PropagatorInput& input) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_literals, *d_flat_weights;
    int *d_changed, *d_contradiction;

    cudaMalloc((void**)&d_M, input.M.size() * sizeof(int));
    cudaMemcpy(d_M, input.M.data(), input.M.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_head, input.num_rules * sizeof(int));
    cudaMemcpy(d_head, input.head.data(), input.num_rules * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_bound, input.num_rules * sizeof(int));
    cudaMemcpy(d_bound, input.bound.data(), input.num_rules * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_rule_offsets, input.rule_offsets.size() * sizeof(int));
    cudaMemcpy(d_rule_offsets, input.rule_offsets.data(), input.rule_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_flat_literals, input.flat_literals.size() * sizeof(int));
    cudaMemcpy(d_flat_literals, input.flat_literals.data(), input.flat_literals.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_flat_weights, input.flat_weights.size() * sizeof(int));
    cudaMemcpy(d_flat_weights, input.flat_weights.data(), input.flat_weights.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_changed, sizeof(int));
    cudaMalloc((void**)&d_contradiction, sizeof(int));
    cudaMemset(d_contradiction, 0, sizeof(int));

    int iteration = 0;
    int threadsPerBlock = 256; 
    int blocksPerGrid = input.num_rules;
    int h_changed, h_contradiction;

    printf("--- Inizio Propagazione ---\n");
    do {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        propagate_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_M, d_head, d_bound, d_rule_offsets, d_flat_literals, d_flat_weights,
            input.num_rules, d_changed, d_contradiction
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
        
        iteration++;
        printf("Iterazione %d: %s\n", iteration, h_changed ? "Nuove deduzioni" : "Nessuna modifica");

    } while (h_changed == 1 && h_contradiction == 0);

    // Recupero il risultato finale di M
    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

    if (h_contradiction) {
        printf("\n>>> Raggiunta una CONTRADDIZIONE (Iterazione %d).\n", iteration);
    } else {
        printf("\n>>> Punto Fisso Raggiunto (Iterazione %d).\n", iteration);
    }

    cudaFree(d_M); cudaFree(d_head); cudaFree(d_bound); 
    cudaFree(d_rule_offsets); cudaFree(d_flat_literals); 
    cudaFree(d_flat_weights); cudaFree(d_changed); cudaFree(d_contradiction);
}

// =========================================================================
// MAIN
// =========================================================================

int main(int argc, char** argv) {


    try {
        PropagatorInput input = parse_dimacs_input();
        
        printf("Parsing completato: %d atomi, %d regole.\n", input.num_atoms, input.num_rules);

        // Lancio la funzione host per eseguire il kernel
        run_propagation(input);

        // Per evitare di intasare il terminale con milioni di atomi, stampiamo 
        // ad esempio solo i primi 10 (o tutti se sono di meno)
        int num_to_print = (input.num_atoms < 10) ? input.num_atoms : 10;
        printf("\nStato Finale dei primi %d Atomi:\n", num_to_print);
        for (int i = 1; i <= num_to_print; i++) {
            printf("Atomo %d: ", i);
            if (input.M[i] == TRUE_VAL) printf("VERO\n");
            else if (input.M[i] == FALSE_VAL) printf("FALSO\n");
            else printf("UNDEF\n");
        }

    } catch (const std::exception& e) {
        std::cerr << "\nEccezione catturata: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}