#pragma once
#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ─── Block-level shared-memory reduction ───────────────────────────────────────

/**
 * Reduces two partial sums (S_sat, S_undef) across all threads in a block
 * using shared memory.  After this call, thread 0 holds the final sums
 * in shared_sat[0] and shared_undef[0], and all threads can read them
 * after the trailing __syncthreads().
 *
 * @param shared_sat    shared memory array of size >= blockDim.x
 * @param shared_undef  shared memory array of size >= blockDim.x
 * @param partial_sat   this thread's partial S_sat
 * @param partial_undef this thread's partial S_undef
 */
__device__ __forceinline__
void block_reduce_sums(int* shared_sat, int* shared_undef,
                       int partial_sat, int partial_undef) {
    shared_sat[threadIdx.x]   = partial_sat;
    shared_undef[threadIdx.x] = partial_undef;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_sat[threadIdx.x]   += shared_sat[threadIdx.x + offset];
            shared_undef[threadIdx.x] += shared_undef[threadIdx.x + offset];
        }
        __syncthreads();
    }
}

// ─── Warp-shuffle tile reduction ───────────────────────────────────────────────

/**
 * Reduces two partial sums across a cooperative-groups tile using shuffle.
 * After this call, thread_rank() == 0 holds the final sums.
 * All threads in the tile can obtain the result via tile.shfl(val, 0).
 *
 * @param tile          the cooperative groups tile
 * @param partial_sat   [in/out] this thread's partial S_sat → final sum on lane 0
 * @param partial_undef [in/out] this thread's partial S_undef → final sum on lane 0
 */
template <int TILE_SIZE>
__device__ __forceinline__
void tile_reduce_sums(cg::thread_block_tile<TILE_SIZE>& tile,
                      int& partial_sat, int& partial_undef) {
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        partial_sat   += tile.shfl_down(partial_sat, offset);
        partial_undef += tile.shfl_down(partial_undef, offset);
    }
}

#endif // REDUCTION_CUH
