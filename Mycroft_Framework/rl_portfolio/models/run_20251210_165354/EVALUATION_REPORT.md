# Mycroft RL Portfolio System - Evaluation Report

## Performance Summary

### RL Agents vs Baselines

| Strategy | Return | Sharpe | Max DD | Volatility | Type |
|----------|--------|--------|--------|------------|------|
| Min-Variance         |  16.47% |  2.020 |  -8.71% |  23.55% | Baseline |
| ValueAgent           |  17.77% |  1.995 | -10.62% |  25.94% | RL Agent |
| RiskAgent            |  16.52% |  1.791 | -10.47% |  27.14% | RL Agent |
| Orchestrator         |  15.93% |  1.707 | -11.15% |  27.58% | RL Agent |
| Buy-and-Hold         |  14.52% |  1.593 | -10.77% |  26.98% | Baseline |
| Equal-Weight         |  14.52% |  1.593 | -10.77% |  26.98% | Baseline |
| GrowthAgent          |  11.45% |  1.198 | -11.90% |  28.97% | RL Agent |
| Momentum             |   8.98% |  0.809 | -13.51% |  37.25% | Baseline |

## Key Findings

**Best Risk-Adjusted Performance:** Min-Variance (Sharpe: 2.020)

**RL Agents Average Sharpe:** 1.673 ± 0.294
**Baselines Average Sharpe:** 1.504 ± 0.437

✓ **RL agents outperform baseline strategies on average**

