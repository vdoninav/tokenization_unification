| dataset | C | horizon | empirical_winner |
| --- | --- | --- | --- |
| ETTh1 | 7 | 96 | patch |
| ETTh1 | 7 | 336 | patch |
| Electricity | 321 | 96 | patch |
| Electricity | 321 | 336 | patch |
| Weather | 21 | 96 | patch |
| Weather | 21 | 336 | point |

**Empirical decision rule:** strategy `patch` is optimal in 5 of 6 observed cells. Exceptions where a different tokenizer was best: `point` (in Weather/H=336). To first order the strategy-selection rule reduces to tau* = `patch` with the exception cells listed explicitly. Six observed cells are too few to tell whether the exceptions reflect structural properties of the data or statistical noise; building a full classification tree would require extending the matrix to additional datasets and horizons.
