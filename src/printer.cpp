#include "../include/parser.hpp" 
#include "../include/printer.hpp"
#include <iostream>

using namespace std;

void print_structure(DIMACSInput& data, bool contradiction) {

    if(contradiction >= 2){
        cout << "s ERROR\n";
        return; 
    }
    if(contradiction == 1){
        cout << "s CONTRADICTION\n";
        return;
    }

    cout << "s SUCCESS\n";
    cout << "v ";
        
    for (int i = 1; i <= data.num_atoms; i++) {
        int val = data.M[i];
        if (val == TRUE) {
            cout << i << " ";
        } else if (val == FALSE) {
            cout << -i << " ";
        }
    }
    cout << "0" << endl;
    cout << "d ";

    vector<int> undef_rules;

    for (int r = 0 ; r < data.num_rules; r++) {
        int head = data.head[r];
        int bound = data.bound[r];
        int start_idx = data.rule_offsets[r];
        int end_idx = data.rule_offsets[r + 1];

        int S_sat = 0;
        int S_undef = 0;

        for (int i = start_idx; i < end_idx; i++) {
            int lit = data.flat_lits[i];
            int weight = data.flat_weights[i];
            int atom = abs(lit);
            int lit_sat = (lit > 0) ? TRUE : FALSE; 
            int lit_not = (lit > 0) ? FALSE : TRUE;
            
            if(data.M[atom] == UNDEF){
                S_undef += weight;
            } else {
                int lit_val = (data.M[atom] == TRUE) ? lit_sat : lit_not;
                if (lit_val == TRUE) {
                    S_sat += weight;
                }
            }
        }

        int S_max = S_sat + S_undef;
        int h_atom = abs(head);
        int h_sat = (head > 0) ? TRUE : FALSE; 
        int h_not = (head > 0) ? FALSE : TRUE;

        int head_val;
        if(data.M[h_atom] == UNDEF) head_val = UNDEF;
        else head_val = (data.M[h_atom] == TRUE) ? h_sat : h_not;
        
        int body_val = UNDEF;
        if (S_sat >= bound) {
            body_val = TRUE;
        } else if (S_max < bound) {
            body_val = FALSE;
        }

        if (head_val == UNDEF || body_val == UNDEF) {
            undef_rules.push_back(r);
        } else if (head_val == body_val) {
            cout << r << " ";
        } else {
            cout << -r << " ";
        }
    }
    cout << "0" << endl;
    cout << "u ";
    for(auto x: undef_rules) {
        cout << x << " ";
    }
    cout << "0" << endl;
}