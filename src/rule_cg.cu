#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../include/parser.hpp" 
#include "../include/printer.hpp" 

namespace cg = cooperative_groups;

__device__ void atomicDeduce(int* M, int atom_id, int deduced_val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom_id], UNDEF, deduced_val);
    if (old_val == UNDEF) {
        *changed = 1;
    } else if (old_val != deduced_val) {
        *contradiction = 1; 
    }
}

template <int TILE_SIZE>
__global__ void propagation_kernel(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_literals, const int* flat_weights,
    int num_rules, int num_atoms, 
    int* global_changed, int* global_contradiction
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    // 1. CALCOLO DEI LIMITI DEL BLOCCO
    int tilesPerBlock = block.size() / TILE_SIZE; 
    int first_rule_in_block = blockIdx.x * tilesPerBlock;
    int last_rule_in_block = min(first_rule_in_block + tilesPerBlock, num_rules);
    
    int start_lit_block = 0;
    int num_lits_block = 0;
    
    // Scopriamo quanti letterali in totale gestisce questo blocco
    if (first_rule_in_block < num_rules) {
        start_lit_block = rule_offsets[first_rule_in_block];
        int end_lit_block = rule_offsets[last_rule_in_block];
        num_lits_block = end_lit_block - start_lit_block;
    }

    // 2. PARTIZIONAMENTO DELLA MEMORIA CONDIVISA DINAMICA
    extern __shared__ int shared_mem[];
    int* M_local             = shared_mem;
    int* lits_local          = &shared_mem[num_atoms + 1];
    int* weights_local       = &shared_mem[num_atoms + 1 + num_lits_block];
    int* local_changed       = &shared_mem[num_atoms + 1 + num_lits_block * 2];
    int* local_contradiction = &shared_mem[num_atoms + 1 + num_lits_block * 2 + 1];

    int rule_id = first_rule_in_block + tile.meta_group_rank();
    bool valid_rule = (rule_id < num_rules);
    
    int start_idx = 0, end_idx = 0, num_literals = 0, B = 0;
    int h_lit = 0, h_atom = 0, h_val = 0, h_not_val = 0;

    if (valid_rule) {
        start_idx = rule_offsets[rule_id];
        end_idx = rule_offsets[rule_id + 1];
        num_literals = end_idx - start_idx;
        B = bound[rule_id];
        h_lit = head[rule_id];
        h_atom = abs(h_lit);
        h_val = (h_lit > 0) ? TRUE : FALSE;
        h_not_val = (h_lit > 0) ? FALSE : TRUE;
    }

    for (int i = block.thread_rank(); i < num_lits_block; i += block.size()) {
        lits_local[i] = flat_literals[start_lit_block + i];
        weights_local[i] = flat_weights[start_lit_block + i];
    }

    bool flag_global = true;
    while(flag_global) {

        // --- FASE 1: CARICAMENTO COOPERATIVO ---
        // A. Caricamento Atomi
        for (int i = block.thread_rank(); i < num_atoms + 1; i += block.size()) {
            M_local[i] = M[i];
        }
        

        if (block.thread_rank() == 0) {
            *local_changed = 0;
            *local_contradiction = 0;
        }
        
        // ASPETTIAMO CHE TUTTA LA SHARED MEMORY SIA PRONTA
        block.sync();

        //if (*global_contradiction) break;

        // --- FASE 2: PUNTO FISSO LOCALE ---
        bool flag_local = true;
        while(flag_local) {
            
            if (block.thread_rank() == 0) *local_changed = 0;
            block.sync();

            if (*local_contradiction) break;

            if (valid_rule) {
                int partial_S_sat = 0;
                int partial_S_undef = 0;

                for (int i = tile.thread_rank(); i < num_literals; i += tile.size()) {
                    int global_lit_idx = start_idx + i;
                    // Calcoliamo l'indice relativo per leggere dalla Shared Memory
                    int local_lit_idx = global_lit_idx - start_lit_block; 
                    
                    int lit = lits_local[local_lit_idx];        // LETTURA DA CACHE VELOCISSIMA!
                    int weight = weights_local[local_lit_idx];  // LETTURA DA CACHE VELOCISSIMA!
                    int atom = abs(lit);

                    int m_val = M_local[atom]; 
                    
                    if (m_val == UNDEF) {
                        partial_S_undef += weight;
                    } else if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
                        partial_S_sat += weight;
                    }
                }

                // Warp Reduction
                for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
                    partial_S_sat += tile.shfl_down(partial_S_sat, offset);
                    partial_S_undef += tile.shfl_down(partial_S_undef, offset);
                }
                int S_sat = tile.shfl(partial_S_sat, 0);
                int S_undef = tile.shfl(partial_S_undef, 0);
                int S_max = S_sat + S_undef;

                // Body -> Head
                if (tile.thread_rank() == 0) {
                    if (S_sat >= B) { 
                        atomicDeduce(M_local, h_atom, h_val, local_contradiction, local_changed);
                    } else if (S_max < B) { 
                        atomicDeduce(M_local, h_atom, h_not_val, local_contradiction, local_changed);
                    }
                }
                
                tile.sync();

                // Head -> Body
                int h_val_cur = M_local[h_atom];
                if (h_val_cur != UNDEF) {
                    bool h_sat = ((h_lit > 0) && h_val_cur == TRUE) || ((h_lit < 0) && h_val_cur == FALSE);

                    for (int i = tile.thread_rank(); i < num_literals; i += tile.size()) {
                        int global_lit_idx = start_idx + i;
                        int local_lit_idx = global_lit_idx - start_lit_block;
                        
                        int lit = lits_local[local_lit_idx];
                        int weight = weights_local[local_lit_idx];
                        int atom = abs(lit);
                        
                        int lit_val = (lit > 0) ? TRUE : FALSE;
                        int lit_not_val = (lit > 0) ? FALSE : TRUE;

                        if (M_local[atom] == UNDEF) {
                            if (h_sat) { 
                                if (S_max - weight < B) { 
                                    atomicDeduce(M_local, atom, lit_val, local_contradiction, local_changed);
                                }
                            } else { 
                                if (S_sat + weight >= B) { 
                                    atomicDeduce(M_local, atom, lit_not_val, local_contradiction, local_changed);
                                }
                            }
                        }
                    }
                }
            }

            block.sync(); 
            if (*local_changed == 0 || *local_contradiction == 1) flag_local = false;
            block.sync(); 
        }

        // --- FASE 3: MERGE IN MEMORIA GLOBALE ---
        for (int i = block.thread_rank(); i < num_atoms + 1; i += block.size()) {
            if (M_local[i] != UNDEF) {
                atomicDeduce(M, i, M_local[i], global_contradiction, global_changed);
            }
        }

        // --- FASE 4: SINCRONIZZAZIONE GRIGLIA ---
        grid.sync();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (*global_changed == 0 || *global_contradiction == 1) {
                *global_changed = -1; 
            } else {
                *global_changed = 0;  
            }
        }

        grid.sync();
        if (*global_changed == -1) flag_global = false;
    }
}

bool run_propagation(PropagatorInput& input) {
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
    cudaMemset(d_contradiction, 0, sizeof(int));
    cudaMemset(d_changed, 0, sizeof(int)); // Inizializza a 0

    const int TILE_SIZE = 16; 
    int threadsPerBlock = 256; 
    
    int tilesPerBlock = threadsPerBlock / TILE_SIZE; // 16 regole per blocco
    int blocksPerGrid = (input.num_rules + tilesPerBlock - 1) / tilesPerBlock;

    // --- CALCOLO DELLA SHARED MEMORY ---
    // Caso peggiore: un blocco gestisce 16 regole piene, ognuna con 16 letterali = 256 letterali.
    int max_lits_per_block = tilesPerBlock * 16; 
    
    // Spazio richiesto: Atomi + Letterali + Pesi + 2 Flag
    int maxSharedInts = input.M.size() + max_lits_per_block + max_lits_per_block + 2;
    int sharedMemSize = maxSharedInts * sizeof(int);

    void* kernelArgs[] = {
        (void*)&d_M,
        (void*)&d_head,
        (void*)&d_bound,
        (void*)&d_rule_offsets,
        (void*)&d_flat_literals,
        (void*)&d_flat_weights,
        (void*)&input.num_rules,
        (void*)&input.num_atoms, // Passiamo il numero di atomi
        (void*)&d_changed,
        (void*)&d_contradiction
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)propagation_kernel<TILE_SIZE>, // Casting necessario con i template
        dim3(blocksPerGrid), dim3(threadsPerBlock),
        kernelArgs,
        sharedMemSize, // Allocazione dinamica memoria condivisa
        0
    );

    if (err != cudaSuccess){
        std::cerr << "Errore CUDA Launch: " << cudaGetErrorString(err) << "\n";
        return 0;
    }

    cudaDeviceSynchronize();

    int h_contradiction;

    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_M); cudaFree(d_head); cudaFree(d_bound); 
    cudaFree(d_rule_offsets); cudaFree(d_flat_literals); 
    cudaFree(d_flat_weights); cudaFree(d_changed); cudaFree(d_contradiction);

    return h_contradiction;
}

int main() {
    try {
        PropagatorInput input = parse_dimacs_input();
        bool contradiction = run_propagation(input);
        print_structure(input,contradiction);
    } catch (const std::exception& e) {
        std::cerr << "\nEccezione: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}