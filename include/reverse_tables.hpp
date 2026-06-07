#pragma once
#ifndef REVERSE_TABLES_HPP
#define REVERSE_TABLES_HPP

#include <vector>
#include "parser.hpp"

struct ReverseTables {
    std::vector<int> atom_body_offsets;
    std::vector<int> atom_body_rules;
    std::vector<int> atom_body_lits;
    std::vector<int> atom_body_weights;

    std::vector<int> atom_head_offsets;
    std::vector<int> atom_head_rules;
};

void build_reverse_tables(DIMACSInput& input, ReverseTables& revt);

#endif  // REVERSE_TABLES_HPP
