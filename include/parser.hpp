#pragma once
#ifndef DIMACS_PARSER_HPP
#define DIMACS_PARSER_HPP

#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

enum TruthValue {
    FALSE = 0,
    UNDEF = -1,
    TRUE  = 1
};

struct PropagatorInput {
    int num_atoms;
    int num_rules;

    // Assignment array: 1-based indexing (size num_atoms + 1)
    vector<int> M;

    // Rule metadata (size num_rules)
    vector<int> head;
    vector<int> bound;
    
    // CSR Offsets (size num_rules + 1)
    // Rule 'i' owns literals from rule_offsets[i] to rule_offsets[i+1] - 1
    vector<int> rule_offsets;

    // Flattened arrays for variable-length bodies
    vector<int> flat_literals;
    vector<int> flat_weights;

    PropagatorInput() : num_atoms(0), num_rules(0) {}
};


PropagatorInput parse_dimacs_input();

#endif // DIMACS_PARSER_HPP