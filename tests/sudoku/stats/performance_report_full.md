# Sudoku Performance Report (Full Profiling)

This report compares the performance of different Sudoku solver configurations. The times reported represent the average total time across all 9x9 test instances, categorized by component.

## Summary Table

| Configuration      |   Kernel Time (ns) |   Memoria Time (ns) |   Malloc Time (ns) |   CPU Time (ns) |   Total Time (ns) |   Num Tests |
|:-------------------|-------------------:|--------------------:|-------------------:|----------------:|------------------:|------------:|
| rule_packed        |            29909.1 |              267070 |        1.62008e+08 |        107043   |       1.62412e+08 |           9 |
| rule_packed_cg     |            36950.9 |              113656 |        1.62913e+08 |         39207.6 |       1.63103e+08 |           9 |
| rule_cg            |            75231   |              114974 |        1.63052e+08 |         77717.3 |       1.63319e+08 |           9 |
| atom_cg            |           105407   |              167846 |        1.63206e+08 |        178041   |       1.63657e+08 |           9 |
| atom_packed        |            55706.7 |              317061 |        1.63197e+08 |        245741   |       1.63816e+08 |           9 |
| rule_partial_fp_cg |            55084.9 |              116655 |        1.63579e+08 |        108769   |       1.6386e+08  |           9 |
| atom_packed_cg     |           166570   |              168203 |        1.63594e+08 |        316864   |       1.64245e+08 |           9 |
| rule               |            60924.8 |              220550 |        1.64017e+08 |        122600   |       1.64421e+08 |           9 |
| atom               |            70956.6 |              317256 |        1.65176e+08 |        261722   |       1.65826e+08 |           9 |

## Performance Graph

![Performance Comparison](/home/ale/arcpal/tests/sudoku/stats/performance_comparison_full.png)
