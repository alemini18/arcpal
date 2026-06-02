#include "../include/parser.hpp" 
#include "../include/printer.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

void print_structure(PropagatorInput& data, bool is_contradiction) {

    cout << "s " << (is_contradiction ? "CONTRADICTION" : "SUCCESS") << "\n";
    cout << "v ";
        
    for (int atom = 1; atom <= data.num_atoms; atom++) {
        int state = data.M[atom];
        if (state == TRUE) {
            cout << atom << " ";
        } else if (state == FALSE) {
            cout << -atom << " ";
        }
    }
    cout << "0\n";
    cout << "d ";

    vector<int> undef_rules;

    for (int r = 1; r <= data.num_rules; r++) {
        int head = data.head[r-1];
        int bound = data.bound[r-1];
        int start_ptr = data.rule_offsets[r-1];
        int end_ptr = data.rule_offsets[r];

        int S_sat = 0;
        int S_undef = 0;

        for (int i = start_ptr; i < end_ptr; i++) {
            int lit = data.flat_literals[i];
            int weight = data.flat_weights[i];
            int atom = abs(lit);
            int lit_sat = (lit > 0) ? TRUE : FALSE; 
            int lit_neg = (lit > 0) ? FALSE : TRUE;
            
            int lit_val = (data.M[atom] == TRUE) ? lit_sat : lit_neg;
            if (lit_val == TRUE) {
                S_sat += weight;
            } else if (lit_val == UNDEF) {
                S_undef += weight;
            }
        }

        int S_max = S_sat + S_undef;
        int h_atom = abs(head);
        int h_sat = (head > 0) ? TRUE : FALSE; 
        int h_neg = (head > 0) ? FALSE : TRUE;

        int head_status = (data.M[h_atom] == TRUE) ? h_sat : h_neg;
        
        int body_status = UNDEF;
        if (S_sat >= bound) {
            body_status = TRUE;
        } else if (S_max < bound) {
            body_status = FALSE;
        }

        string rule_status;
        if (head_status == UNDEF || body_status == UNDEF) {
            undef_rules.push_back(r);
        } else if (head_status == body_status) {
            cout << r << " ";
        } else {
            cout << -r << " ";
        }
    }
    cout << "0\n";
    cout << "u ";
    for(auto x: undef_rules) {
        cout << x << " ";
    }
    cout << "0\n";
}