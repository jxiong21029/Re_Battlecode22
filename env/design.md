# Environment Design Specification v0
## Observation Spaces

## Action Spaces

| Entity     | Action Space                                                                 | Notes                                                                           | TODO                  |     |
|------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-----------------------|-----|
| Miner      | Discrete(9) 0-8: movement                                                    | auto mines (to 1 lead)                                                          | leave 1 lead decision |     |
| Builder    | Discrete(10) 0-8: movement 9: spawn lab                                      | auto repairs                                                                    | watchtowers           |     |
| Soldier    | Discrete(9) 0-8: movement                                                    | auto attacks w/ HP heuristic                                                    | attack actions        |     |
| Sage       | Discrete(9) 0-8: movement                                                    | auto attacks w/ HP heuristic                                                    | attack actions        |     |
| Archon     | Discrete(4) 0: idle/repair 1: spawn miner 2: spawn builder 3: spawn "combat" | "combat" spawn attempts to spawn sage when possible, otherwise spawns a soldier | movement?             |     |
| Laboratory | N/A                                                                          | auto converts                                                                   | convert action        |     |
| Watchtower | N/A                                                                          | doesn't get built                                                               | add support           |     |
