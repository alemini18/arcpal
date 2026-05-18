#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cooperative_groups.h> // LIBRERIA AGGIUNTA

namespace cg = cooperative_groups; // Alias per comodità

#include "../include/parser.hpp" 

// =========================================================================
// KERNEL E FUNZIONI DEVICE
// =========================================================================

__device__ void atomicDeduce(int* M, int atom_id, int deduced_val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom_id], UNDEF_VAL, deduced_val);
    if (old_val == UNDEF_VAL) {
        *changed = 1;
    } else if (old_val != deduced_val) {
        *contradiction = 1; 
    }
}

// Nota: Aggiunto il loop interno e l'uso di cg::grid_group
__global__ void propagate_kernel_cooperative(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_literals, const int* flat_weights,
    int num_rules, int* global_changed, int* global_contradiction, int* iterations
) {
    // 1. Inizializziamo il gruppo cooperativo globale
    cg::grid_group grid = cg::this_grid();
    
    int rule_id = blockIdx.x;
    
    // Variabili condivise per il calcolo dei pesi
    __shared__ int S_sat_shared;
    __shared__ int S_undef_shared;
    
    // Flag locali per controllare il ciclo di questo blocco
    bool keep_running = true;

    // Loop principale di punto fisso (eseguito interamente su GPU)
    while (keep_running) {
        
        // Se c'è una contraddizione globale, ci fermiamo
        if (*global_contradiction) break;

        // Se il blocco gestisce una regola valida, esegue la logica
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

            int h_lit = head[rule_id];
            int h_atom = abs(h_lit);
            bool h_wants_true = (h_lit > 0);
            int head_val_if_sat = h_wants_true ? TRUE_VAL : FALSE_VAL;
            int head_val_if_falsified = h_wants_true ? FALSE_VAL : TRUE_VAL;

            // FASE 2: Propagazione Body -> Head (Solo Thread 0)
            if (threadIdx.x == 0) {
                if (S_sat >= B) { 
                    atomicDeduce(M, h_atom, head_val_if_sat, global_contradiction, global_changed);
                } else if (S_max < B) { 
                    atomicDeduce(M, h_atom, head_val_if_falsified, global_contradiction, global_changed);
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
                            atomicDeduce(M, atom, lit_val_if_sat, global_contradiction, global_changed);
                        }
                    } else if (is_head_satisfied == -1) { 
                        if (S_sat + weight >= B) { 
                            atomicDeduce(M, atom, lit_val_if_falsified, global_contradiction, global_changed);
                        }
                    }
                }
            }
        } // Fine della logica della singola regola

        // =====================================================================
        // SINCRONIZZAZIONE GLOBALE E GESTIONE DEL LOOP
        // =====================================================================
        
        // 1. Tutti i thread di tutti i blocchi aspettano che le deduzioni siano finite
        grid.sync();

        // 2. Solo un singolo thread di tutta la grid (Blocco 0, Thread 0) gestisce lo stato
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (*global_changed == 0 || *global_contradiction == 1) {
                // Punto fisso o contraddizione: dobbiamo fermarci
                *global_changed = -1; // Usiamo -1 come segnale di stop
            } else {
                // Continuiamo: prepariamo i flag per il prossimo giro
                *global_changed = 0; 
                (*iterations)++;
            }
        }

        // 3. Tutti i thread aspettano che il Thread 0 abbia deciso se continuare
        grid.sync();

        // 4. Tutti i thread leggono la decisione e aggiornano il loro ciclo while
        if (*global_changed == -1) {
            keep_running = false;
        }
    }
}

// =========================================================================
// FUNZIONE HOST: LANCIO COOPERATIVO
// =========================================================================

void run_propagation(PropagatorInput& input) {
    int *d_M, *d_head, *d_bound, *d_rule_offsets, *d_flat_literals, *d_flat_weights;
    int *d_changed, *d_contradiction, *d_iterations;

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
    cudaMalloc((void**)&d_iterations, sizeof(int));
    
    cudaMemset(d_changed, 0, sizeof(int));
    cudaMemset(d_contradiction, 0, sizeof(int));
    cudaMemset(d_iterations, 0, sizeof(int)); // Inizializziamo a 0

    // DENTRO A run_propagation, prima del lancio del kernel:

int threadsPerBlock = 256; 
int numBlocks = 0;

// Chiediamo a CUDA qual è il numero MASSIMO di blocchi attivi per questo specifico kernel
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, 
    (void*)propagate_kernel_cooperative, 
    threadsPerBlock, 
    0 // Shared memory dinamica (0 nel nostro caso)
);

// Calcoliamo i blocchi totali moltiplicando per il numero di multiprocessori
int deviceId;
cudaGetDevice(&deviceId);
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, deviceId);

int maxCooperativeBlocks = numBlocks * deviceProp.multiProcessorCount;

// Usiamo il numero minore tra le regole totali e il limite fisico della GPU
int blocksPerGrid = (input.num_rules < maxCooperativeBlocks) ? input.num_rules : maxCooperativeBlocks;

printf("Lancio Cooperative Kernel con %d blocchi (Max supportati: %d)\n", blocksPerGrid, maxCooperativeBlocks);

    // LANCIO DEL KERNEL COOPERATIVO
    // A differenza di <<<...>>>, dobbiamo usare cudaLaunchCooperativeKernel
    void* kernelArgs[] = {
        (void*)&d_M, (void*)&d_head, (void*)&d_bound, (void*)&d_rule_offsets, 
        (void*)&d_flat_literals, (void*)&d_flat_weights, (void*)&input.num_rules, 
        (void*)&d_changed, (void*)&d_contradiction, (void*)&d_iterations
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)propagate_kernel_cooperative,
        dim3(blocksPerGrid), dim3(threadsPerBlock),
        kernelArgs,
        0, 0
    );

    if (err != cudaSuccess) {
        std::cerr << "Errore CUDA Launch: " << cudaGetErrorString(err) << "\n";
        // Potrebbe fallire se ci sono troppi blocchi rispetto agli SM disponibili (vedi sotto)
        return;
    }

    cudaDeviceSynchronize(); // Aspettiamo che il kernel (tutto il loop) finisca

    int h_contradiction, h_iterations;
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_iterations, d_iterations, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

    if (h_contradiction) {
        printf("\n>>> Raggiunta una CONTRADDIZIONE (Iterazioni totali: %d).\n", h_iterations);
    } else {
        printf("\n>>> Punto Fisso Raggiunto (Iterazioni totali: %d).\n", h_iterations);
    }

    cudaFree(d_M); cudaFree(d_head); cudaFree(d_bound); 
    cudaFree(d_rule_offsets); cudaFree(d_flat_literals); 
    cudaFree(d_flat_weights); cudaFree(d_changed); 
    cudaFree(d_contradiction); cudaFree(d_iterations);
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