#include "../include/parser.hpp" 
#include "../include/utils.hpp"
#include "../include/printer.hpp"
#include <iostream>
#include <string>
#include <vector>


/**
 * Prints the complete PropagatorInput structure along with real-time 
 * evaluation of sums, head truth values, body truth values, and rule satisfaction.
 */
void print_structure(PropagatorInput& data, bool is_contradiction) {

    std::cout << "s " << (is_contradiction ? "CONTRADICTION" : "SUCCESS") << "\n";
    std::cout << "v ";
        
    for (int atom = 1; atom <= data.num_atoms; ++atom) {
        int state = data.M[atom];
        if (state == TRUE_VAL) {
            std::cout << atom << " ";
        } else if (state == FALSE_VAL) {
            std::cout << -atom << " ";
        }
    }
    std::cout << "0\n";
    std::cout << "d ";

    std::vector<int> undef_rules;

    for (int r = 1; r <= data.num_rules; ++r) {
        int head      = data.head[r-1];
        int bound     = data.bound[r-1];
        int start_ptr = data.rule_offsets[r-1];
        int end_ptr   = data.rule_offsets[r];

        int S_sat = 0;
        int S_undef = 0;

        for (int i = start_ptr; i < end_ptr; ++i) {
            int lit = data.flat_literals[i];
            int weight = data.flat_weights[i];
            
            int lit_val = get_literal_val(data.M, lit);
            if (lit_val == TRUE_VAL) {
                S_sat += weight;
            } else if (lit_val == UNDEF_VAL) {
                S_undef += weight;
            }
        }

        int S_max = S_sat + S_undef;

        int head_status = get_literal_val(data.M, head);
        
        int body_status = -1;
        if (S_sat >= bound) {
            body_status = 1;
        } else if (S_max < bound) {
            body_status = 0;
        }

        std::string rule_status;
        if (head_status == -1 || body_status == -1) {
            undef_rules.push_back(r);
        } else if (head_status == body_status) {
            std::cout << r << " ";
        } else {
            std::cout << -r << " ";
        }
    }
    std::cout << "0\n";
    std::cout << "u ";
    for(auto x: undef_rules) {
        std::cout << x << " ";
    }
    std::cout << "0\n";
}