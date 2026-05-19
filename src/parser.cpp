#include "../include/parser.hpp"
#include <iostream>
#include <sstream>
#include <cmath>


PropagatorInput parse_dimacs_input() {

    PropagatorInput data;
    std::string line;
    int current_rule = 0;

    while (std::getline(std::cin, line)) {
        // Trim leading whitespace or skip empty lines and comments
        if (line.empty() || line[0] == 'c') continue;

        std::istringstream iss(line);
        char type;
        iss >> type;

        if (type == 'p') {
            iss >> data.num_atoms >> data.num_rules;
            
            // Initialize assignments to UNDEF. 
            // Sized num_atoms + 1 to safely accommodate 1-based literal indices.
            data.M.assign(data.num_atoms + 1, UNDEF);
            
            // Pre-allocate exact memory for rule metadata
            data.head.resize(data.num_rules);
            data.bound.resize(data.num_rules);
            data.rule_offsets.resize(data.num_rules + 1, 0);
            
        } else if (type == 'r') {
            if (current_rule >= data.num_rules) {
                throw std::runtime_error("Error: Found more rules than specified in the 'p' header.");
            }

            int h, b, k;
            iss >> h >> b >> k;

            data.head[current_rule]  = h;
            data.bound[current_rule] = b;
            
            // Mark the starting index in the flattened array for this rule
            data.rule_offsets[current_rule] = data.flat_literals.size();

            // Parse the k literal-weight pairs sequentially
            for (int i = 0; i < k; ++i) {
                int lit, weight;
                iss >> lit >> weight;
                data.flat_literals.push_back(lit);
                data.flat_weights.push_back(weight);
            }

            current_rule++;
            // Instantly seal the end offset for the current rule
            data.rule_offsets[current_rule] = data.flat_literals.size();

        } else if (type == 'a') {
            int init_lit;
            // Parse initial assignments until the trailing 0 terminator
            while (iss >> init_lit && init_lit != 0) {
                int atom = std::abs(init_lit);
                if (atom > data.num_atoms || atom == 0) {
                    throw std::runtime_error("Error: Initial assignment references invalid atom ID.");
                }
                // Assign TRUE (1) if positive, FALSE (-1) if negative
                data.M[atom] = (init_lit > 0) ? TRUE : FALSE;
            }
        }
    }

    if (current_rule != data.num_rules) {
        throw std::runtime_error("Error: Header specified " + std::to_string(data.num_rules) + 
                                 " rules, but parsed exactly " + std::to_string(current_rule));
    }

    return data;
}

