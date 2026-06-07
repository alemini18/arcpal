#pragma once
#ifndef DIMACS_PARSER_HPP
#define DIMACS_PARSER_HPP

#include <vector>

enum TruthValue {
    FALSE = 0,
    UNDEF = -1,
    TRUE  = 1
};

struct DIMACSInput {
    int num_atoms;
    int num_rules;

    std::vector<int> M;

    std::vector<int> head;
    std::vector<int> bound;
    std::vector<int> rule_offsets;
    std::vector<int> flat_lits;
    std::vector<int> flat_weights;

    DIMACSInput() : num_atoms(0), num_rules(0) {}
};


DIMACSInput parse_dimacs_input();

#endif // DIMACS_PARSER_HPP