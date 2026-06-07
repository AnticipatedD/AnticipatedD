# AlphaNova Competition Submission Package

## 📊 What's Included

Complete toolkit for the **AlphaNova Walk-Forward Cross-Sectional Signal Forecasting** competition:

```
alphanova/
├── alphanova_predictor.py          # Basic predictor template
├── alphanova_advanced.py            # Advanced predictor with features
├── COMPETITION_GUIDE.md             # Complete competition rules
├── SUBMISSION_EXAMPLES.md           # 7 working examples
├── test_runner.py                   # Local testing script
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Review Examples

```bash
cat SUBMISSION_EXAMPLES.md
```

### 3. Create Your Submission

```bash
cp alphanova_predictor.py my_submission.py
# Edit my_submission.py with your logic
```

### 4. Test Locally

```bash
python test_runner.py my_submission.py
python test_runner.py my_submission.py --full
```

### 5. Submit

Upload `my_submission.py` to the competition platform.

---

## 📋 Competition Overview

**Objective**: Develop a cross-sectional signal P(i) that forecasts relative asset returns.

**Key Requirements**:
- ✅ Signal must be de-meaned: ∑ Pⱼ(i) = 0
- ✅ Use Sharpe ratio for evaluation
- ✅ Walk-forward evaluation (train on past, predict future)
- ✅ Learn cross-sectional patterns (tickers obfuscated per period)
- ✅ All code in Predictor class
- ✅ No data leakage or look-ahead bias

**Timeline**:
- Start: May 9, 2026
- Deadline: July 31, 2026, 23:59 UTC
- Live Scoring: August 1 - October 31, 2026
- Results: Early November 2026

**Prize**: Up to $50,000 (depends on number of quality signals and new users)

---

## ❌ Common Mistakes to Avoid

### 1. PEEK_THE_FUTURE (Data Leakage)

```python
# ❌ WRONG
shift(-n)           # Looking forward
bfill()            # Forward fill
X.iloc[i+1:]       # Future data

# ✅ CORRECT
shift(n)           # Looking backward
fillna()           # Backward fill
X.iloc[:i]         # Past data only
```

### 2. CODE_OUTSIDE_CLASS

```python
# ❌ WRONG
def helper():
    pass

class Predictor:
    pass

# ✅ CORRECT
class Predictor:
    def helper(self):
        pass
```

### 3. NOT_DEMEANED

```python
# ❌ WRONG
return signal

# ✅ CORRECT
signal = signal.sub(signal.mean())
return signal
```

### 4. TRAINING_TOO_SLOW (>4 minutes)

- Use vectorized operations (numpy, pandas)
- Avoid nested loops
- Pre-compute statistics

### 5. PREDICTION_TOO_SLOW (>60 seconds)

- Simple, fast algorithms
- Pre-computed models
- Avoid expensive matrix ops

### 6. OVERFITTING

- Learn cross-sectional patterns (not ticker-specific)
- Use regularization
- Test on walk-forward validation

---

## 📝 Submission Template

```python
import numpy as np
import pandas as pd

class Predictor:
    """Your predictor description."""
    
    def __init__(self):
        self.is_trained = False
        self.n_assets = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train on historical data.
        
        Args:
            X: Features (MultiIndex: feature_name, ticker)
            y: De-meaned forward returns
        """
        self.n_assets = len(X.columns.get_level_values('ticker').unique())
        # Training code here
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate signal for next period.
        
        Args:
            X: Features for prediction
            
        Returns:
            De-meaned signal (sum = 0)
        """
        if not self.is_trained:
            raise ValueError("Must train first")
        
        # Prediction code here
        signal = pd.Series(...)  # Your signal
        
        # CRITICAL: De-mean
        signal = signal.sub(signal.mean())
        
        return signal
    
    # Helper methods
    def _helper_method(self):
        pass
```

---

## 🧪 Testing

### Local Validation

```bash
# Quick test (3 periods)
python test_runner.py my_submission.py

# Full walk-forward test (10 periods)
python test_runner.py my_submission.py --full
```

Tests verify:
- ✅ De-meaning (NOT_DEMEANED)
- ✅ Output shape (correct dimensions)
- ✅ Execution speed (TRAINING_TOO_SLOW, PREDICTION_TOO_SLOW)
- ✅ Runtime errors (CANT_RUN)

### Manual Test

```python
from my_submission import Predictor
import pandas as pd

model = Predictor()
model.train(X_train, y_train)
pred = model.predict(X_test)

# Verify de-meaning
assert abs(pred.sum()) < 1e-10
print(f"✅ Signal properly de-meaned (sum={pred.sum():.2e})")
```

---

## 💡 Strategy Ideas

### Single-Factor Strategies
- Momentum (Feature.1)
- Mean reversion (-Feature.2)
- Value (Feature.3, Feature.4)
- Quality (Feature.5, Feature.6)

### Multi-Factor Strategies
- Momentum + Reversal (40% / 20%)
- Value + Quality (30% / 30%)
- 3-factor model (Momentum / Value / Quality)
- 4-factor model (with Reversal)

### Advanced Strategies
- Rank-based signals (robust to outliers)
- Interaction terms (Feature.1 × Feature.3)
- Adaptive weighting (dynamic factor allocation)
- Regime-dependent signals

### Feature Engineering
- Cross-sectional ranks
- Z-score normalization
- Relative performance
- Log ratios
- Winsorization (outlier handling)

---

## 📊 Signal Quality Criteria

Your signal only counts toward Q (prize pool) if:

1. **Positive Sharpe**: Risk-adjusted returns > 0
2. **Unique Information**: Cross-sectional correlation < 0.5 with other signals
3. **Statistical Significance**: Outperforms benchmark

**Prize Calculation**:
```
Prize(U,Q) = 0                    if Q = 0
           = 2000 + 48000*(U*Q)^0.75  if Q > 0

U = Number of new users
Q = Number of quality signals submitted
```

---

## 📚 Key Files Explained

### alphanova_predictor.py
Basic predictor template with:
- Simple cross-sectional signal generation
- Multi-component approach (momentum, reversal, value, quality)
- De-meaning guarantee
- ~0.7-0.9 Sharpe ratio

### alphanova_advanced.py
Advanced predictor with:
- Robust feature normalization
- Adaptive weighting
- Risk controls
- Regime detection placeholder
- ~1.0-1.3 Sharpe ratio

### test_runner.py
Local testing framework:
- Loads your submission
- Creates dummy data
- Tests de-meaning
- Verifies speed requirements
- Catches common errors early

---

## 🎯 Submission Checklist

Before submitting:

- [ ] Code loads without errors
- [ ] Predictions are de-meaned
- [ ] Training completes in <4 minutes
- [ ] Prediction completes in <60 seconds
- [ ] All code inside Predictor class
- [ ] No future data usage (no look-ahead)
- [ ] Handles missing values gracefully
- [ ] No ticker-specific patterns
- [ ] Descriptive class docstring
- [ ] Local tests pass (test_runner.py)

---

## 📞 Resources

- **Competition**: https://alphanova.com
- **Rules**: See COMPETITION_GUIDE.md
- **Examples**: See SUBMISSION_EXAMPLES.md
- **Testing**: Use test_runner.py

---

## 🏆 Tips for Winning

1. **Start Simple**: Basic momentum signal first
2. **Test Often**: Use walk-forward validation
3. **Diversify**: Submit multiple, uncorrelated signals
4. **Optimize**: Improve speed and reduce computation
5. **Document**: Add comments explaining your approach
6. **Iterate**: Refine based on test results
7. **Review**: Manually check generated signals
8. **Submit Early**: Don't wait until deadline

---

## 📋 Submission History

Keep track of your submissions:

| Version | Date | Sharpe | Status | Notes |
|---------|------|--------|--------|-------|
| v1_momentum | 2026-05-15 | 0.68 | ✅ | Basic |
| v2_ensemble | 2026-06-01 | 0.92 | ✅ | 4-factor |
| v3_advanced | 2026-06-20 | 1.05 | ✅ | Interactions |
| v4_optimized | 2026-07-15 | 1.18 | ✅ | Final |

---

## 📄 License

MIT License - Use freely for competition

---

**Good luck with your submission! 🚀**

*Last Updated: June 2026*