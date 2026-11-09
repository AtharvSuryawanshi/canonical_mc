| rule    | name                   | set    | thr     | verdict     | numbers                                                                                                   |
|:--------|:-----------------------|:-------|:--------|:------------|:----------------------------------------------------------------------------------------------------------|
| 1       | no rate effect         | main   | 0.5     | pass        | A3 max 3.288 vs control min 4.515                                                                         |
| 2       | threshold artefact     | main   | 0.3,0.7 | pass        | thr 0.3: A3 max 5.116 vs control min 8.260; thr 0.7: A3 max 2.091 vs control min 2.632                    |
| 3       | accuracy, not budget   | main   | 0.5     | pass        | control acc min 0.939; A3 seeds at/above it: seed 3 acc 0.963 breadth 2.918; control range [4.515, 4.798] |
| 4       | population artefact    | main   | 0.5     | pass        | A3 mean active E 195.0 vs 0.5 x control 101.9                                                             |
| 5       | does not replicate     | second | 0.5     | pass (weak) | feasible only: rate_limited mean 3.388 (n=3) vs control mean 4.920 (n=12)                                 |
| 5 (all) | does not replicate     | second | 0.5     | pass        | all 12, confounded by infeasibility: rate_limited mean 3.359 (n=12) vs control mean 4.920 (n=12)          |
| 6       | wiring does specialize | main   | 0.5     | pass        | B0 max 5.485 vs control min 4.515                                                                         |
