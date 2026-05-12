#include "../include/utils.hpp"
#include "../include/parser.hpp"
#include <vector>
#include <cmath>

void assign_val_to_literal(std::vector<int>& M, int lit, int val) {
    int atom_id = std::abs(lit);
    if(lit>0)   M[atom_id] = val;
    else        M[atom_id] = 1-val;
}

int get_literal_val(std::vector<int>& M, int lit) {
    int val = M[std::abs(lit)];
    if (val == UNDEF_VAL) return UNDEF_VAL;
    if (lit > 0) return val;
    return 1 - val;
}