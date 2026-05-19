#include "../include/parser.hpp" 
#include "../include/utils.hpp"
#include "../include/pretty_printer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>

// Helper to convert a literal integer back to a readable string (e.g., -2 -> "¬A2")
std::string lit_to_string(int lit) {
    if (lit == 0) return "UNDEF";
    return (lit < 0 ? "¬A" : "A") + std::to_string(std::abs(lit));
}

// Helper to convert internal truth enum/integers to readable strings
std::string truth_to_string(int val) {
    switch(val) {
        case TRUE:  return "TRUE";
        case FALSE: return "FALSE";
        default:        return "UNDEF";
    }
}



/**
 * Prints the complete PropagatorInput structure along with real-time 
 * evaluation of sums, head truth values, body truth values, and rule satisfaction.
 */
void pretty_print_structure(PropagatorInput& data) {
    std::cout << "=================================================================\n";
    std::cout << "                 LOGIC PROPAGATOR INTERNAL STATE                 \n";
    std::cout << "=================================================================\n\n";

    // 1. METADATA HEADER
    std::cout << "### 1. System Metadata\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  Total Atoms (N): " << data.num_atoms << "\n";
    std::cout << "  Total Rules (P): " << data.num_rules << "\n";
    std::cout << "  Flattened CSR Body Literals: " << data.flat_literals.size() << "\n\n";

    // 2. ASSIGNMENT ARRAY (M)
    std::cout << "### 2. Current Assignment State (M)\n";
    std::cout << "-----------------------------------------------------------------\n";
    
    int determined_count = 0;
    for (int atom = 1; atom <= data.num_atoms; ++atom) {
        int state = data.M[atom];
        std::cout << "  A" << std::left << std::setw(4) << atom << " : " 
                  << truth_to_string(state) << "\n";
        if (state != UNDEF_VAL) {
            determined_count++;
        }
    }
    std::cout << "  Summary: " << determined_count << "/" << data.num_atoms 
              << " atoms currently determined.\n\n";

    // 3. RULES DECODED WITH RUNTIME EVALUATION
    std::cout << "### 3. Logic Rules & Current Evaluation (P)\n";
    std::cout << "-----------------------------------------------------------------\n";
    
    if (data.num_rules == 0) {
        std::cout << "  [No rules loaded]\n";
    }

    int violated_rules = 0;
    int satisfied_rules = 0;

    for (int r = 0; r < data.num_rules; ++r) {
        int head      = data.head[r];
        int bound     = data.bound[r];
        int start_ptr = data.rule_offsets[r];
        int end_ptr   = data.rule_offsets[r + 1];

        // --- A. Print Mathematical Formulation ---
        std::stringstream ss;
        ss << "  Rule " << std::right << std::setw(4) << r << ":  " 
           << std::left << std::setw(6) << lit_to_string(head) 
           << " ↔  " << bound << " ≤ { ";

        int S_sat = 0;
        int S_undef = 0;

        for (int i = start_ptr; i < end_ptr; ++i) {
            int lit = data.flat_literals[i];
            int weight = data.flat_weights[i];
            
            ss << lit_to_string(lit) << ":" << weight;
            if (i < end_ptr - 1) ss << ", ";

            // Compute partial sums dynamically
            int lit_val = get_literal_val(data.M, lit);
            if (lit_val == TRUE) {
                S_sat += weight;
            } else if (lit_val == UNDEF) {
                S_undef += weight;
            }
        }
        ss << " }";
        std::cout << ss.str() << "\n";

        // --- B. Evaluate Sums ---
        int S_max = S_sat + S_undef;
        std::cout << "              Sums   : S_sat = " << std::left << std::setw(4) << S_sat 
                  << " | S_undef = " << std::setw(4) << S_undef 
                  << " | S_max = " << S_max << "\n";

        // --- C. Evaluate Truth Status ---
        std::string head_status = truth_to_string(get_literal_val(data.M, head));
        
        std::string body_status = "UNDEF";
        if (S_sat >= bound) {
            body_status = "TRUE";
        } else if (S_max < bound) {
            body_status = "FALSE";
        }

        std::string rule_status;
        if (head_status == "UNDEF" || body_status == "UNDEF") {
            rule_status = "\033[33mUNDEF\033[0m"; // Yellow text for undefined
        } else if (head_status == body_status) {
            rule_status = "\033[32mTRUE (Satisfied)\033[0m"; // Green text
            satisfied_rules++;
        } else {
            rule_status = "\033[31mFALSE (VIOLATED!)\033[0m"; // Red text alert
            violated_rules++;
        }

        std::cout << "              Status : Head = " << std::left << std::setw(5) << head_status 
                  << " | Body = " << std::setw(5) << body_status 
                  << " | Rule = " << rule_status << "\n\n";
    }

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  Rules Summary: " << satisfied_rules << " Satisfied | " 
              << violated_rules << " Violated | " 
              << (data.num_rules - satisfied_rules - violated_rules) << " Undefined\n";
    std::cout << "=================================================================\n";
}