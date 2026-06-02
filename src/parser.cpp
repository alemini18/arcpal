#include "../include/parser.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

using namespace std;


PropagatorInput parse_dimacs_input() {

    PropagatorInput data;
    string line;
    int current_rule = 0;

    while (getline(cin, line)) {
        if (line.empty()) continue;

        istringstream iss(line);
        char type;
        iss >> type;

        if (type == 'p') {
            iss >> data.num_atoms >> data.num_rules;
            
            data.M = vector<int>(data.num_atoms + 1, UNDEF);
            
            data.head.resize(data.num_rules);
            data.bound.resize(data.num_rules);
            data.rule_offsets.resize(data.num_rules + 1, 0);
            data.rule_offsets[0] = 0; // First rule
            
        } else if (type == 'r') {

            int h, b, k;
            iss >> h >> b >> k;

            data.head[current_rule]  = h;
            data.bound[current_rule] = b;

            for (int i = 0; i < k; i++) {
                int lit, weight;
                iss >> lit >> weight;
                data.flat_literals.push_back(lit);
                data.flat_weights.push_back(weight);
            }

            current_rule++;
            data.rule_offsets[current_rule] = data.flat_literals.size();

        } else if (type == 'a') {
            int init_lit;

            while (iss >> init_lit && init_lit != 0) {
                int atom = abs(init_lit);
                data.M[atom] = (init_lit > 0) ? TRUE : FALSE;
            }
        }
    }

    return data;
}

