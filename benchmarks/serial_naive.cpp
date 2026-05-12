#include <bits/stdc++.h>
#include "../include/parser.hpp"
#include "../include/utils.hpp"
#include "../include/printer.hpp"

using namespace std;



bool infer_head(PropagatorInput& data, int head, int bound, int satisfied_count, int undefined_count, bool& global_contradiction) {
    int head_val = get_literal_val(data.M, head);
    
    if (satisfied_count >= bound) {
        if (head_val == UNDEF_VAL) {
            assign_val_to_literal(data.M, head, TRUE_VAL);
            return true;
        } else if (head_val == FALSE_VAL) {
            global_contradiction = true;
        }
    } else if (satisfied_count + undefined_count < bound) {
        if (head_val == UNDEF_VAL) {
            assign_val_to_literal(data.M, head, FALSE_VAL);
            return true;
        } else if (head_val == TRUE_VAL) {
            global_contradiction = true;
        }
    }
    return false;
}

bool infer_body(PropagatorInput& data, int head, int bound, int rule_start, int rule_end, int satisfied_count, int undefined_count, bool& global_contradiction) {
    bool changed = false;
    int head_val = get_literal_val(data.M, head);
    
    if (head_val == UNDEF_VAL) {
        return false;
    }

    for (int j = rule_start; j < rule_end; j++) {
        int lit = data.flat_literals[j];
        int weight = data.flat_weights[j];
        int lit_val = get_literal_val(data.M, lit);
        
        if (lit_val != UNDEF_VAL) continue;

        if (head_val == TRUE_VAL && satisfied_count + undefined_count - weight < bound) {
            assign_val_to_literal(data.M, lit, TRUE_VAL);
            changed = true;
        } else if (head_val == FALSE_VAL && satisfied_count + weight >= bound) {
            assign_val_to_literal(data.M, lit, FALSE_VAL);
            changed = true;
        }
    }
    return changed;
}

bool propagate_rule(PropagatorInput& data, int rule_idx, bool& global_contradiction) {
    int head = data.head[rule_idx];
    int bound = data.bound[rule_idx];
    int rule_start = data.rule_offsets[rule_idx];
    int rule_end = data.rule_offsets[rule_idx + 1];

    int satisfied_count = 0;
    int undefined_count = 0;

    for (int j = rule_start; j < rule_end; j++) {
        int lit = data.flat_literals[j];
        int weight = data.flat_weights[j];
        int lit_val = get_literal_val(data.M, lit);

        if (lit_val == TRUE_VAL) {
            satisfied_count += weight;
        } else if (lit_val == UNDEF_VAL) {
            undefined_count += weight;
        }
    }

    bool changed = false;

    changed |= infer_head(data, head, bound, satisfied_count, undefined_count, global_contradiction);
    
    if (global_contradiction) {
        return changed;
    }

    changed |= infer_body(data, head, bound, rule_start, rule_end, satisfied_count, undefined_count, global_contradiction);

    return changed;
}

int main() {
    PropagatorInput data = parse_dimacs_input();
    bool global_changed = true;
    bool global_contradiction = false;

    while (global_changed && !global_contradiction) {

        global_changed = false;

        for (int i = 0; i < data.num_rules; i++) {
            bool changed = propagate_rule(data, i, global_contradiction);
            if (changed) {
                global_changed = true;
            }
            if (global_contradiction) {
                break;
            }
        }
    }

    print_structure(data, global_contradiction);
    return 0;
}
