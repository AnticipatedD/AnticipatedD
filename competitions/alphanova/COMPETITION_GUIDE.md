# AlphaNova Competition - Submission Guide

## 🏆 Competition Overview

**Competition**: Walk-Forward Cross-Sectional Signal Forecasting  
**Company**: AlphaNova  
**Timeline**: May 9, 2026 - August 1, 2026  
**Prize Pool**: Up to $50,000  
**Submission Limit**: 10 entries  
**Live Scoring**: August 1 - October 31, 2026

---

## 📋 Competition Requirements

### Signal Specification

You must develop a cross-sectional signal **P(i) = (P₁(i), ..., Pⱼ(i))** that:

1. **Forecasts relative returns** of J assets
2. **De-meaned at every timestamp**: ∑ⱼ₌₁ᴶ Pⱼ(i) = 0
3. **Follows Sharpe ratio metric**: U = E[r(i)] / √Var[r(i)]

### Data Structure

**Input Features**:
- 6 cross-sectional features (Feature.1 through Feature.6)
- 20 assets per period
- MultiIndex DataFrame: (feature_name, ticker)
- Feature.1 = Close returns (also used for backtesting)

**Target**:
- Cross-sectionally de-meaned forward returns
- One-period ahead returns

**Constraints**:
- ❌ NO data leakage (no future information)
- ❌ NO ticker-specific patterns (tickers obfuscated per period)
- ❌ NO look-ahead bias
- ✅ All logic must be in Predictor class
- ✅ All derived features inside the class

---

## 🚫 Common Disqualification Reasons

### 1. PEEK_THE_FUTURE (Data Leakage)
```python
# ❌ WRONG - Uses future data
shift(-n)  # Looking forward
bfill()    # Forward fill with future data
X.iloc[i+1:]  # Accessing future values
target during prediction  # Using y during predict()

# ✅ CORRECT - Use only past data
shift(n)   # Looking backward
fillna()   # Fill with past values
X.iloc[:i]  # Only past data
```

### 2. CODE_OUTSIDE_CLASS
```python
# ❌ WRONG
def helper_function():
    pass

class Predictor:
    pass

# ✅ CORRECT
class Predictor:
    def helper_method(self):
        pass
```

### 3. NOT_DEMEANED
```python
# ❌ WRONG - Signal not de-meaned
return signal

# ✅ CORRECT - De-mean before returning
signal = signal.sub(signal.mean())
return signal
```

### 4. TRAINING_TOO_SLOW
```python
# Must complete within 4 minutes
# Optimize by:
# - Avoiding nested loops
# - Using vectorized operations (numpy, pandas)
# - Limiting complex computations
# - Using efficient algorithms
```

### 5. PREDICTION_TOO_SLOW
```python
# Must complete within 60 seconds
# Optimize by:
# - Pre-computing statistics during training
# - Using simple, fast algorithms during prediction
# - Avoiding expensive matrix operations
```

### 6. OVERFITTING_FAILED
```python
# Avoid overfitting by:
# - Using cross-sectional, not ticker-specific patterns
# - Limiting signal complexity
# - Using regularization
# - Testing walk-forward validation
```

---

## 📝 Submission Template

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Predictor:
    """Your predictor class."""
    
    def __init__(self):
        self.is_trained = False
        self.n_assets = None
        # All state variables here
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train on historical data.
        
        Args:
            X: Features (MultiIndex: feature_name, ticker)
            y: De-meaned forward returns
        """
        self.n_assets = len(X.columns.get_level_values('ticker').unique())
        # Training logic here - must complete in <4 minutes
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate cross-sectional signal.
        
        Args:
            X: Features for prediction period
            
        Returns:
            De-meaned prediction signal (sum to 0)
        """
        # Prediction logic - must complete in <60 seconds
        signal = pd.Series(...)  # Your signal
        
        # CRITICAL: De-mean the signal
        signal = signal.sub(signal.mean())
        
        return signal
    
    # Helper methods inside the class
    def _helper_method(self, X):
        pass
```

---

## 🔍 Testing Checklist

Before submitting:

```bash
# 1. Check for de-meaning
python runner.py your_submission.py

# 2. Full walk-forward validation
python runner.py your_submission.py --full

# 3. Local verification
python -m pytest tests/test_submission.py
```

### Manual Testing

```python
from your_submission import Predictor
import pandas as pd
import numpy as np

# Create dummy data
X_train = pd.DataFrame(...)  # Your training data
y_train = pd.Series(...)    # Your targets
X_test = pd.DataFrame(...)   # Test features

# Test
model = Predictor()
model.train(X_train, y_train)
predictions = model.predict(X_test)

# Verify de-meaning
assert np.abs(predictions.sum()) < 1e-10, "Not de-meaned!"
print(f"Signal sum: {predictions.sum()}")
print(f"Sharpe-like ratio: {predictions.mean() / predictions.std()}")
```

---

## 💡 Strategy Guide

### Feature Engineering Ideas

1. **Cross-Sectional Ranks**
   ```python
   ranks = feature.rank() / len(feature)
   centered = ranks - 0.5
   ```

2. **Z-Score Normalization**
   ```python
   zscore = (feature - feature.mean()) / feature.std()
   ```

3. **Relative Performance**
   ```python
   relative = feature / feature.mean()
   log_relative = np.log(relative)
   ```

4. **Interaction Terms**
   ```python
   interaction = feature1 * feature2
   ```

5. **Momentum & Reversal**
   ```python
   momentum = feature.iloc[-1]  # Recent strength
   reversal = -feature.iloc[-2]  # Opposite of older value
   ```

### Signal Combination

```python
# Ensemble approach
signal = (w1 * momentum + 
          w2 * reversal + 
          w3 * value + 
          w4 * quality)

# Ensure de-meaning
signal = signal - signal.mean()
```

### Quality Signal Selection

Your signal only counts if:
1. ✅ Sharpe > 0 (positive risk-adjusted returns)
2. ✅ Correlation < 0.5 with other selected signals
3. ✅ Brings new, uncorrelated information

---

## 📊 Walk-Forward Evaluation

```
Period 001  →  Train on historical, predict period 002
Period 002  →  Train on [001, 002], predict period 003
...
Period N    →  Train on [001...N], predict period N+1

Final Score = Aggregate Sharpe across all test periods
```

**Important**: Ticker identities change each period. Learn patterns, not specific tickers.

---

## 🎯 Submission Tips

1. **Start Simple**: Begin with basic momentum/reversal
2. **Test Incrementally**: Add features one at a time
3. **Validate Often**: Run walk-forward tests locally
4. **Prevent Overfitting**: Use cross-sectional patterns only
5. **Monitor Performance**: Track Sharpe ratio trends
6. **Optimize Speed**: Profile your code
7. **Add Comments**: Document your approach
8. **Version Control**: Keep all submissions

---

## 📞 Support & Resources

- **Competition Platform**: [AlphaNova](https://alphanova.com)
- **Documentation**: Check official guidelines
- **Community**: Engage in competition discussions
- **Testing Tool**: `runner.py` provided with competition

---

## 🚀 Submission Process

1. **Finalize Code**: Ensure all requirements met
2. **Run Tests**: `python runner.py submission.py --full`
3. **Create Entry**: Name descriptively (e.g., `alphanova_momentum_v1`)
4. **Upload File**: Submit `.py` file
5. **Verify**: Check status on leaderboard
6. **Iterate**: Up to 10 submissions allowed

---

## 💰 Prize Structure

```
Prize(U,Q) = {
  0                           if Q = 0
  2000 + 48000·(U·Q)^0.75    if Q > 0
}

Where:
  U = Number of new users registered
  Q = Number of quality signals submitted
```

---

*Last Updated: June 2026*