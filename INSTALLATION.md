# Mycroft RL Portfolio - Installation & Setup Guide

## Prerequisites
- Python 3.9+
- pip3
- 2GB disk space
- Internet connection (for stock data)

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/nikbearbrown/Mycroft.git
cd Mycroft/Mycroft_Framework/rl_portfolio
```

### 2. Install Dependencies
```bash
pip3 install -r requirements.txt
```

**Required packages:**
- torch>=2.0.0 (PyTorch for neural networks)
- stable-baselines3>=2.0.0 (RL algorithms)
- gymnasium>=0.29.0 (RL environment interface)
- yfinance>=0.2.28 (Market data)
- pandas, numpy, matplotlib (Data processing & viz)

### 3. Verify Installation
```bash
python3 test_setup.py
```

**Expected output:**
```
======================================================================
✓ ALL TESTS PASSED!
======================================================================
```

### 4. Test Run (Quick Demo - 5 minutes)
```bash
python3 train.py --agent-steps 1000 --orch-steps 1000
```

### 5. Full Training (90 minutes)
```bash
python3 train.py --agent-steps 30000 --orch-steps 40000
```

### 6. Evaluate Results
```bash
# Replace TIMESTAMP with your run directory
python3 evaluation/comprehensive_eval.py models/run_TIMESTAMP
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Make sure you're in the project root directory
```bash
cd ~/path/to/Mycroft
python3 Mycroft_Framework/rl_portfolio/train.py
```

### Issue: "KeyError: 'Adj Close'"
**Solution:** yfinance API changed, already fixed in latest code

### Issue: Slow training
**Solution:** Reduce timesteps for demo
```bash
python3 train.py --agent-steps 5000 --orch-steps 5000
```

## Directory Structure After Installation
```
rl_portfolio/
├── models/              # Trained models saved here
│   └── run_TIMESTAMP/   # Each training run
├── logs/               # TensorBoard logs
│   ├── growth/
│   ├── value/
│   ├── risk/
│   └── orchestrator/
└── data/               # Downloaded stock data (cached)
```

## Viewing Training Progress

### TensorBoard (Optional)
```bash
tensorboard --logdir=Mycroft_Framework/rl_portfolio/logs
```

Open browser to `http://localhost:6006`

## System Requirements

**Minimum:**
- 4GB RAM
- 2 CPU cores
- Training time: 2-3 hours

**Recommended:**
- 8GB RAM
- 4+ CPU cores
- GPU (optional, speeds up training 3-5x)
- Training time: 30-60 minutes

## Next Steps

After successful installation:
1. Review `README.md` for architecture overview
2. Read `EXPERIMENTAL_DESIGN.md` for methodology
3. Run training and evaluation
4. Examine results in `models/run_TIMESTAMP/`
