# Sudoku Performance Report

This report compares the performance of different Sudoku solver configurations. The times reported represent the average total GPU compute time (excluding memory transfers and CUDA API overhead) across all 9x9 test instances.

## Summary Table

| Configuration      |   Average Compute Time (ns) |   StdDev (ns) |   Num Tests |
|:-------------------|----------------------------:|--------------:|------------:|
| rule_packed        |                     29909.1 |       6864.71 |           9 |
| rule_packed_cg     |                     36950.9 |       8415.93 |           9 |
| rule_partial_fp_cg |                     55084.9 |      12406.2  |           9 |
| rule               |                     60924.8 |      12455.6  |           9 |
| rule_cg            |                     75231   |      15465.7  |           9 |
| atom_packed        |                    123850   |      15714.8  |           9 |
| atom               |                    148859   |      24917.7  |           9 |
| atom_cg            |                    170291   |      16125.8  |           9 |
| atom_packed_cg     |                    235220   |      35146.2  |           9 |

## Performance Graph

![Performance Comparison](/home/ale/arcpal/tests/sudoku/stats/performance_comparison.png)
