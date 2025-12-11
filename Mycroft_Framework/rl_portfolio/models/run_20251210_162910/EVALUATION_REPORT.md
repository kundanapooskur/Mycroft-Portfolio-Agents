# Mycroft RL Portfolio System - Evaluation Report

## Performance Summary

### RL Agents vs Baselines

| Strategy | Return | Sharpe | Max DD | Volatility | Type |
|----------|--------|--------|--------|------------|------|
| Min-Variance         |  16.47% |  2.020 |  -8.71% |  23.55% | Baseline |
| ValueAgent           |  14.77% |  1.820 |  -7.87% |  23.52% | RL Agent |
| Buy-and-Hold         |  14.52% |  1.593 | -10.77% |  26.98% | Baseline |
| Equal-Weight         |  14.52% |  1.593 | -10.77% |  26.98% | Baseline |
| RiskAgent            |  13.09% |  1.479 | -10.75% |  26.16% | RL Agent |
| Orchestrator         |  11.96% |  1.384 | -10.23% |  25.50% | RL Agent |
| GrowthAgent          |   9.14% |  0.939 | -11.85% |  30.08% | RL Agent |
| Momentum             |   8.98% |  0.809 | -13.51% |  37.25% | Baseline |

## Key Findings

**Best Risk-Adjusted Performance:** Min-Variance (Sharpe: 2.020)

**RL Agents Average Sharpe:** 1.405 ± 0.314
**Baselines Average Sharpe:** 1.504 ± 0.437

