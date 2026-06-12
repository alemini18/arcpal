#include "../include/parser.hpp"
#include <iostream>

using namespace std;

DIMACSInput parse_dimacs_input() {
    DIMACSInput data;
    char type;
    cin >> type;
    if (type == 'p'){
        cin >> data.num_atoms >> data.num_rules;

        data.M = vector<int>(data.num_atoms + 1, UNDEF);
        data.head.resize(data.num_rules);
        data.bound.resize(data.num_rules);
        data.rule_offsets.resize(data.num_rules + 1, 0);
        data.rule_offsets[0] = 0;
    
        for(int i = 0; i < data.num_rules; i++){
            int h, b, k;
            cin >> type;
            if (type == 'r'){
                cin >> h >> b >> k;
                data.head[i] = h;
                data.bound[i] = b;
                
                for (int j = 0; j < k; j++) {
                    int lit, weight;
                    cin >> lit >> weight;
                    data.flat_lits.push_back(lit);
                    data.flat_weights.push_back(weight);
                }

                data.rule_offsets[i + 1] = data.flat_lits.size();

            }else{
                cerr << "ERROR: Invalid rule" << i;
                exit(1);
            }
        }
        cin >> type;
        if(type == 'a'){
            int lit;
            while (cin >> lit && lit != 0) {
                int atom = abs(lit);
                data.M[atom] = (lit > 0) ? TRUE : FALSE;
            }
            return data;
        }else{
            cerr << "ERROR: Invalid assignment";
            exit(1);
        }

    }else{
        cerr << "ERROR: File does not start with 'p'";
        exit(1);
    }
}