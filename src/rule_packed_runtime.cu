#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
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

// Definiamo il Tile Size a compile-time tramite template
template <int TILE_SIZE>
__global__ void propagation_kernel_cg(
    int* M,
    const int* head, const int* bound, const int* rule_offsets,
    const int* flat_literals, const int* flat_weights,
    int num_rules, int* changed, int* contradiction
) {
    if (*contradiction) return;

    // Inizializzazione Cooperative Groups e suddivisione in Tile
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    // Identificazione della regola assegnata a questo specifico Tile
    int rule_id = (blockIdx.x * tile.meta_group_size()) + tile.meta_group_rank();

    if (rule_id >= num_rules) return;

    int start_idx = rule_offsets[rule_id];
    int end_idx = rule_offsets[rule_id + 1];
    int num_literals = end_idx - start_idx;
    int B = bound[rule_id];

    int partial_S_sat = 0;
    int partial_S_undef = 0;

    // I thread nel Tile cooperano sui letterali della regola
    for (int i = tile.thread_rank(); i < num_literals; i += tile.size()) {
        int lit_idx = start_idx + i;
        int lit = flat_literals[lit_idx];
        int atom = abs(lit);
        int weight = flat_weights[lit_idx]; 

        int m_val = M[atom];
        
        if (m_val == UNDEF) {
            partial_S_undef += weight;
        } else if (((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE))) {
            partial_S_sat += weight;
        }
    }

    // Riduzione parallela nei registri (Warp Shuffle)
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        partial_S_sat += tile.shfl_down(partial_S_sat, offset);
        partial_S_undef += tile.shfl_down(partial_S_undef, offset);
    }

    // Broadcast dei risultati a tutti i thread del Tile
    int S_sat = tile.shfl(partial_S_sat, 0);
    int S_undef = tile.shfl(partial_S_undef, 0);
    int S_max = S_sat + S_undef;

    int h_lit = head[rule_id];
    int h_atom = abs(h_lit);
    int h_val = (h_lit > 0) ? TRUE : FALSE;
    int h_not_val = (h_lit > 0) ? FALSE : TRUE;

    // Fase 1: Body -> Head
    if (tile.thread_rank() == 0) {
        if (S_sat >= B) { 
            atomicDeduce(M, h_atom, h_val, contradiction, changed);
        } else if (S_max < B) { 
            atomicDeduce(M, h_atom, h_not_val, contradiction, changed);
        }
    }
    
    // Sincronizzazione a livello di Tile prima della Fase 2
    tile.sync();

    // Fase 2: Head -> Body
    int h_val_cur = M[h_atom];

    if (h_val_cur != UNDEF) {
        bool h_sat = ((h_lit > 0) && h_val_cur == TRUE) || ((h_lit < 0) && h_val_cur == FALSE);

        for (int i = tile.thread_rank(); i < num_literals; i += tile.size()) {
            int lit_idx = start_idx + i;
            int lit = flat_literals[lit_idx];
            int atom = abs(lit);
            int weight = flat_weights[lit_idx];
            
            int lit_val = (lit > 0) ? TRUE : FALSE;
            int lit_not_val = (lit > 0) ? FALSE : TRUE;

            if (M[atom] == UNDEF) {
                if (h_sat) { 
                    if (S_max - weight < B) { 
                        atomicDeduce(M, atom, lit_val, contradiction, changed);
                    }
                } else { 
                    if (S_sat + weight >= B) { 
                        atomicDeduce(M, atom, lit_not_val, contradiction, changed);
                    }
                }
            }
        }
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

    // Impostazioni della griglia per i Cooperative Groups
    const int TILE_SIZE = 16; 
    int threadsPerBlock = 256; 
    
    int tilesPerBlock = threadsPerBlock / TILE_SIZE; 
    int blocksPerGrid = (input.num_rules + tilesPerBlock - 1) / tilesPerBlock;

    int h_changed, h_contradiction;

    // Loop di propagazione a punto fisso
    do {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        propagation_kernel_cg<TILE_SIZE><<<blocksPerGrid, threadsPerBlock>>>(
            d_M, d_head, d_bound, d_rule_offsets, d_flat_literals, d_flat_weights,
            input.num_rules, d_changed, d_contradiction
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_contradiction, d_contradiction, sizeof(int), cudaMemcpyDeviceToHost);

    } while (h_changed == 1 && h_contradiction == 0);

    // Recupero del modello aggiornato
    cudaMemcpy(input.M.data(), d_M, input.M.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Pulizia della memoria
    cudaFree(d_M); cudaFree(d_head); cudaFree(d_bound); 
    cudaFree(d_rule_offsets); cudaFree(d_flat_literals); 
    cudaFree(d_flat_weights); cudaFree(d_changed); cudaFree(d_contradiction);

    return h_contradiction;
}

int main() {
    try {
        PropagatorInput input = parse_dimacs_input();
        
        bool contradiction = run_propagation(input);

        print_structure(input);

    } catch (const std::exception& e) {
        std::cerr << "\nEccezione catturata: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}