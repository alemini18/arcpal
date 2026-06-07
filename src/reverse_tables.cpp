#include <vector>
#include <cmath>

#include "../include/reverse_tables.hpp"

using namespace std;

void build_reverse_tables(DIMACSInput& input, ReverseTables& revt) {
    int num_atoms = input.num_atoms;
    int num_rules = input.num_rules;

    revt.atom_body_offsets.assign(num_atoms + 2, 0);
    revt.atom_head_offsets.assign(num_atoms + 2, 0);


    for (int i = 0; i < num_rules; i++) {
        
        int h_lit = input.head[i];
        if (h_lit != 0) {
            int h_atom = abs(h_lit);
            revt.atom_head_offsets[h_atom + 1]++;
        }

        
        int start = input.rule_offsets[i];
        int end = input.rule_offsets[i + 1];
        for (int j = start; j < end; j++) {
            int lit = input.flat_lits[j];
            int atom = abs(lit);
            revt.atom_body_offsets[atom + 1]++;
        }
    }

    
    for (int i = 1; i <= num_atoms + 1; i++) {
        revt.atom_body_offsets[i] += revt.atom_body_offsets[i - 1];
        revt.atom_head_offsets[i] += revt.atom_head_offsets[i - 1];
    }

    
    revt.atom_body_rules.resize(revt.atom_body_offsets.back());
    revt.atom_body_lits.resize(revt.atom_body_offsets.back());
    revt.atom_body_weights.resize(revt.atom_body_offsets.back());
    
    revt.atom_head_rules.resize(revt.atom_head_offsets.back());

    vector<int> body_offset = revt.atom_body_offsets;
    vector<int> head_offset = revt.atom_head_offsets;

    for (int i = 0; i < num_rules; i++) {
        
        int h_lit = input.head[i];
        if (h_lit != 0) {
            int h_atom = abs(h_lit);
            int h_idx = head_offset[h_atom]++;
            revt.atom_head_rules[h_idx] = i;
        }

        int start = input.rule_offsets[i];
        int end = input.rule_offsets[i + 1];
        for (int j = start; j < end; j++) {
            int lit = input.flat_lits[j];
            int atom = abs(lit);
            int weight = input.flat_weights[j];

            int b_idx = body_offset[atom]++;
            revt.atom_body_rules[b_idx] = i;
            revt.atom_body_lits[b_idx] = lit;
            revt.atom_body_weights[b_idx] = weight;
        }
    }
}