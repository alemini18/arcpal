#pragma once
#ifndef DIMACS_PARSER_HPP
#define DIMACS_PARSER_HPP

#include <vector>
#include <string>
#include <stdexcept>

// Enums for truth values to maintain readability across CPU and GPU
enum TruthValue {
    FALSE = 0,
    UNDEF = -1,
    TRUE  = 1
};

// Structure holding the flattened CSR-like rule set ready for GPU transfer
struct PropagatorInput {
    int num_atoms;
    int num_rules;

    // Assignment array: 1-based indexing (size num_atoms + 1)
    // M[0] is a padding byte left unused so atom IDs map directly to indices.
    std::vector<int> M;

    // Rule metadata (size num_rules)
    std::vector<int> head;
    std::vector<int> bound;
    
    // CSR Offsets (size num_rules + 1)
    // Rule 'i' owns literals from rule_offsets[i] to rule_offsets[i+1] - 1
    std::vector<int> rule_offsets;

    // Flattened arrays for variable-length bodies
    std::vector<int> flat_literals;
    std::vector<int> flat_weights;

    // Constructor to ensure clean initialization
    PropagatorInput() : num_atoms(0), num_rules(0) {}
};

/**
 * Parses a DIMACS-inspired logic propagation file.
 * * @param filename The path to the input text file.
 * @return PropagatorInput populated with flattened CSR arrays ready for cudaMemcpy.
 * @throws std::runtime_error if the file cannot be opened or contains syntax errors.
 */
PropagatorInput parse_dimacs_input();

#endif // DIMACS_PARSER_HPP