#ifndef DIMACS_PARSER_HPP
#define DIMACS_PARSER_HPP

#include <vector>

using namespace std;

enum TruthValue {
    FALSE = 0,
    UNDEF = -1,
    TRUE  = 1
};

struct DIMACSInput {
    int num_atoms;
    int num_rules;

    vector<int> M;

    vector<int> head;
    vector<int> bound;
    vector<int> rule_offsets;
    vector<int> flat_lits;
    vector<int> flat_weights;

    DIMACSInput() : num_atoms(0), num_rules(0) {}
};


DIMACSInput parse_dimacs_input();

#endif // DIMACS_PARSER_HPP