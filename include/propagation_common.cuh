#pragma once
#ifndef PROPAGATION_COMMON_CUH
#define PROPAGATION_COMMON_CUH

#include "parser.hpp"

// ─── Literal evaluation helpers ────────────────────────────────────────────────

/**
 * Returns true if the literal `lit` is satisfied under assignment value `m_val`.
 * A positive literal is satisfied when m_val == TRUE;
 * a negative literal is satisfied when m_val == FALSE.
 */
__device__ __forceinline__
bool literal_is_satisfied(int lit, int m_val) {
    return ((lit > 0) && (m_val == TRUE)) || ((lit < 0) && (m_val == FALSE));
}

/**
 * Returns the truth value that would satisfy the literal:
 *   positive lit → TRUE, negative lit → FALSE.
 */
__device__ __forceinline__
int lit_sat_value(int lit) {
    return (lit > 0) ? TRUE : FALSE;
}

/**
 * Returns the truth value that would falsify the literal:
 *   positive lit → FALSE, negative lit → TRUE.
 */
__device__ __forceinline__
int lit_unsat_value(int lit) {
    return (lit > 0) ? FALSE : TRUE;
}

// ─── Atomic assignment (rule-level variants) ───────────────────────────────────

/**
 * Atomically assigns `val` to M[atom] if currently UNDEF.
 * Sets *changed = 1 on success, *contradiction = 1 on conflict.
 * Used by rule-level solvers that iterate until no changes occur.
 */
__device__ __forceinline__
void atomicAssign(int* M, int atom, int val, int* contradiction, int* changed) {
    int old_val = atomicCAS(&M[atom], UNDEF, val);
    if (old_val == UNDEF) {
        *changed = 1;
    } else if (old_val != val) {
        *contradiction = 1;
    }
}

// ─── Atomic assignment + enqueue (atom-level variants) ─────────────────────────

/**
 * Atomically assigns `val` to M[atom] if currently UNDEF, then enqueues
 * the atom into queue_out for the next propagation wave.
 * Sets *contradiction = 1 on conflict.
 * Used by atom-level solvers that track modified atoms via work queues.
 */
__device__ __forceinline__
void atomicAssignAndQueue(int* M, int atom, int val,
                          int* contradiction, int* queue_out, int* num_out) {
    int old_val = atomicCAS(&M[atom], UNDEF, val);
    if (old_val == UNDEF) {
        int idx = atomicAdd(num_out, 1);
        queue_out[idx] = atom;
    } else if (old_val != val) {
        *contradiction = 1;
    }
}

#endif // PROPAGATION_COMMON_CUH
