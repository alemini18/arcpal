#include <bits/stdc++.h>
#include "../include/parser.hpp"
#include "../include/printer.hpp"

using namespace std;

int lit_value(int m_val, int lit) {
    if (m_val == UNDEF) return UNDEF;
    if (lit > 0) return m_val;
    return (m_val == TRUE) ? FALSE : TRUE;
}

bool propagate_rule(DIMACSInput& data, int rule_idx, bool& contradiction) {
    int head = data.head[rule_idx];
    int bound = data.bound[rule_idx];
    int rule_start = data.rule_offsets[rule_idx];
    int rule_end = data.rule_offsets[rule_idx + 1];

    int S_sat = 0;
    int S_undef = 0;

    for (int j = rule_start; j < rule_end; j++) {
        int lit = data.flat_lits[j];
        int weight = data.flat_weights[j];
        int atom = abs(lit);
        int lit_val = lit_value(data.M[atom], lit);

        if (lit_val == TRUE) {
            S_sat += weight;
        } else if (lit_val == UNDEF) {
            S_undef += weight;
        }
    }

    bool changed = false;

    int h_atom = abs(head);
    int h_val = lit_value(data.M[h_atom], head);
    
    if (S_sat >= bound) {
        if (h_val == UNDEF) {
            data.M[h_atom] = (head > 0) ? TRUE : FALSE;
            changed = true;
        } else if (h_val == FALSE) {
            contradiction = true;
        }
    } else if (S_sat + S_undef < bound) {
        if (h_val == UNDEF) {
            data.M[h_atom] = (head > 0) ? FALSE : TRUE;
            changed = true;
        } else if (h_val == TRUE) {
            contradiction = true;
        }
    }
    
    if (contradiction) {
        return changed;
    }

    if (h_val != UNDEF) {
        for (int j = rule_start; j < rule_end; j++) {
            int lit = data.flat_lits[j];
            int weight = data.flat_weights[j];
            int atom = abs(lit);
            int lit_val = lit_value(data.M[atom], lit);

            if (lit_val != UNDEF) continue;

            if (h_val == TRUE && S_sat + S_undef - weight < bound) {
                data.M[atom] = (lit > 0) ? TRUE : FALSE;
                changed = true;
            } else if (h_val == FALSE && S_sat + weight >= bound) {
                data.M[atom] = (lit > 0) ? FALSE : TRUE;
                changed = true;
            }
        }
    }

    return changed;
}

int main() {
    DIMACSInput data = parse_dimacs_input();
    bool changed = true;
    bool contradiction = false;
    int iterations = 0;

    while (changed && !contradiction) {
        changed = false;
        iterations++;

        for (int i = 0; i < data.num_rules; i++) {
            changed |= propagate_rule(data, i, contradiction);
            if (contradiction) {
                break;
            }
        }
    }

    print_structure(data, contradiction, iterations);
    return 0;
}
