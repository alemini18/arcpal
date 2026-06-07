#pragma once
#ifndef REVERSE_TABLES
#define REVERSE_TABLES

#include <vector>
#include "parser.hpp"

using namespace std;

struct ReverseTables{
    vector<int> atom_body_offsets;
    vector<int> atom_body_rules;
    vector<int> atom_body_lits;
    vector<int> atom_body_weights;

    vector<int> atom_head_offsets;
    vector<int> atom_head_rules;
};

void build_reverse_tables(DIMACSInput& input, ReverseTables& revt);

#endif  // REVERSE_TABLES

