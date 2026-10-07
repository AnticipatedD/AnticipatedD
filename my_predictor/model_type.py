# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
[1]:
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
read in data

[2]:
returns=pd.read_csv("returns.07.10.26.csv",index_col=0)
target=pd.read_csv("rel_returns.07.10.26.csv",index_col=0)
features=pd.read_csv("features.07.10.26.csv",index_col=0)
split into train and validate. please use test_size=0.25 as indicated below

[3]:
train_target, validate_target = train_test_split(target, test_size=0.25,shuffle=False )
train_returns, validate_returns = train_test_split(returns, test_size=0.25,shuffle=False)
train_features, validate_features = train_test_split(features, test_size=0.25,shuffle=False )

train_data={'returns':train_returns,'features':train_features}

validate_data={'returns':validate_returns,'features':validate_features}
helper functions

[4]:
def backtest(predictions,relative_returns):
predictions.ffill(inplace=True)

#the following is closely relatived to measuring the lead lag correlation of the prediction to the outcome.
    pf_returns = (predictions.shift(1)).mul(relative_returns.values).sum(axis=1)
    pf_returns.iloc[0] = 0  # first day return is 0, because we do not have prediction for time i=-1
    return pf_returns

def returns_to_equity(returns):
    equity = returns.add(1).cumprod()
    return equity

def utility_sharpe(returns):
    unscaled_sharpe=returns.mean()/returns.std()
    return float(unscaled_sharpe) 

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal_series = None
        
        # Operational constraints
        self.target_bound = 0.22
        self.optimal_concentration = 0.25
        self.l1_hysteresis_threshold = 0.75

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract and lock the unique cross-sectional feature names from the multi-index.
        """
        if features is not None:
            # Safely capture the specific tracking tokens across Level 0 of columns
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
# Ensure we have data and exactly 6 features available to prevent index crashes 
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 2:
            return zero_signal
            
        try:
 [5]:     
  # 1. Shuffling-Immune Cross-Sectional Rank Transform ranks = {}
     
for feat in self.feature_names: 
    block_val = 
    features[feat].astype(np.float64) 
    r = (block_val.rank(axis=1, 
pct=True, method='max') - 0.5) * 2.0 
    ranks[feat] = 
    r.fillna(0.0).to_numpy()
 
    # Read structural dimensions from our generated ranks matrix 
    N_time, J_assets = 
    list(ranks.values())[0].shape
 
    # Extract and explicitly map each individual feature tracking layer 
    f1_rank = 
    ranks[self.feature_names[1]] 
    f2_rank = 
    ranks[self.feature_names[2]] 
    f3_rank = 
    ranks[self.feature_names[3]] 
    f4_rank = 
    ranks[self.feature_names[4]] 
    f5_rank = 
    ranks[self.feature_names[5]] 
    f6_rank = 
    ranks[self.feature_names[6]] 
    
    # 2. Complete 6-Feature Non-Linear Spatial Interaction Layout 
    interaction_blocks = [] 
    
    # Map structural tanh activations for base ranks 
    for f_name in 
    self.feature_names[:2]: 
        interaction_blocks.append(np.tanh(ranks[f_name] * 2.0))  
        
    # Cross-interaction mappings across features 

interaction_blocks.append(np.sin(f1_rank * np.pi * 0.25) * np.cos(f2_rank * np.pi * 0.25))
            interaction_blocks.append(f1_rank * np.abs(f2_rank))           
            interaction_blocks.append(np.sin(f3_rank * np.pi * 0.25) * np.cos(f4_rank * np.pi * 0.25))
            interaction_blocks.append(f3_rank * np.abs(f4_rank))
            
    # Dedicated 5th and 6th feature interaction closure lines
            interaction_blocks.append(np.arctan(f5_rank) * np.tanh(f6_rank))
            interaction_blocks.append(f5_rank * f6_rank * np.sign(f1_rank))

    # Pack structures into a multi-dimensional array [Interactions, Time, Assets]  
tensor_blocks = np.stack(interaction_blocks, axis=0)

    # 3. Asymmetric Directional Weight Projection Loop 

raw_velocity = np.zeros((N_time, J_assets), dtype=np.float32) 
for t in range(N_time): 
    
    cross_slice = tensor_blocks[:, t, :] 
    # Dimension: [M_interactions, J_assets] 
    
    cross_slice -= 
    cross_slice.mean(axis=1, 
                     keepdims=True)
                 
    # Execute SVD factorization over the snapshot step 
    u, s, vh = 
    np.linalg.svd(cross_slice, 
                  full_matrices=False)
                 
    # Extract first orthogonal vector component and align sign directionally 
    raw_velocity[t] = vh[0] * np.sign(np.sum(vh[0]))
                
 
    # 4. Global Hypersphere Normalization 
    velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True) 
    norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-8] = 0.5
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration
    
   # 5. Asset Mapping & L1 Execution Hysteresis Matrix Logic
            final_positions = np.zeros_like(sphere_target)
            if self.prev_signal_series is not None:
                active_position = self.prev_signal_series.reindex(tickers, fill_value=0.0).to_numpy(dtype=np.float32)
            else:
                active_position = np.zeros(J_assets, dtype=np.float32)
            
            for t in range(N_time):
                target_position = sphere_target[t]
                l1_allocation_delta = np.sum(np.abs(target_position - active_position))
                
                if l1_allocation_delta < self.l1_hysteresis_threshold:
                    current_allocation = active_position.copy()
                else:
                    current_allocation = 0.25 * target_position + 0.75 * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
    # 6. Final Clean Post-Clip Demean Check
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            for _ in range(4):
                final_df = final_df.sub(final_df.mean(axis=1), axis=0)
                final_df = final_df.clip(-self.target_bound, self.target_bound)
            
            # Cache ongoing structural vector back into memory
            self.prev_signal_series = final_df.iloc[-1].astype(np.float32)
            return final_df.astype(np.float64)
            
        except Exception:
            return zero_signal

[6]:
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "P",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "### `PtolemyISoter_Submission_v1_1_4.py`: Predicting Relative Returns in assets Deadline Oct 15 2026 at 6:00:00 AM GMT+6\n",
       "\n",
    "#### Objective\n",
"PtolemyISoter_Submission_v1_1_4 is challenged to develop a predictive signal that forecasts the relative returns of a set of assets. The goal is to create a signal $P(i)$ at each timestamp $i$ that effectively predicts the next period’s excess returns for multiple assets compared to a market index. The objective is to maximize a utility function $U$, which measures both the predictive accuracy and consistency of the signal in forecasting these relative returns.\n",
    "\n", 
       
       "#### Background\n",
"Financial markets are known for their nonstationarity and complex dynamics. The assets here, which will remain masked,  enable leveraged trading without an expiry date, are particularly popular and offer unique opportunities for relative return prediction. This challenge tests PtolemyISoter_Submission_v1_1_4’ file is ability to forecast returns that are not only accurate but also consistent over time.\n",
    "\n",
    "#### Data Description\n",
    "- **Input Data:** A time series of returns from assets, $X(i) = (X_1(i), X_2(i), ..., X_J(i))$, where $i$ denotes the time index expressed as an integer, and each $X_j(i)$ represents the return of the $j$-th asset at time $i$.\n",
    "- **Prediction Target:** At each time $i$, participants will predict a vector signal $P(i) = (P_1(i), P_2(i), ..., P_J(i))$, where each $P_j(i)$ represents the `PtolemyISoter_Submission_v1_1_4.py` file’s prediction for the relative return of the $j$-th asset in the next time step, $i+1$, compared to the index.\n",
    "\n",
    "   > **Note:** The actual values of the target relative returns, $R(i+1)$, are obfuscated to maintain data confidentiality, and `PtolemyISoter` participants will work with pre-processed, relative return values that do not reveal underlying raw prices.\n",
    "\n",
    "- **Feature Data:** In addition to historical return data, `PtolemyISoter` participants will have access to a set of obfuscated features. Each timestamp $i$ is associated with a feature vector $F(i) = (F_1(i), F_2(i), ..., F_K(i))$, where $K$ is the number of available features, and each feature is identified only by its index (e.g., $F_1, F_2, F_3, F_4, F_5, F_6\\dots, F_K$).\n",
    "\n",
    "#### Objective Function (Utility)\n",
    "The performance of each submission will be evaluated using the following utility function:\n",
    "\n",
    "$$\n",
    "U = \\frac{\\mathbb{E}[P(i) \\cdot R(i+1)]}{\\operatorname{std}(P(i) \\cdot R(i+1))}\n",
    "$$\n",
    "\n",
    "where:\n",
    "- $ R(i+1) $ represents the vector of relative returns of each asset versus the index at time $i+1$.\n",
    "- The term $P(i) \\cdot R(i+1)$ captures the alignment between the prediction $P(i)$ and the realized relative return $R(i+1)$.\n",
    "- The numerator, $\\mathbb{E}[P(i) \\cdot R(i+1)]$, represents the average predictive alignment.\n",
    "- The denominator, $\\operatorname{std}(P(i) \\cdot R(i+1))$, promotes stability by penalizing high variance.\n",
    "\n",
    "#### Scoring Criteria\n",
    "\n",
    "For now we will ignore overfitting tests, but in the future, when we are up an running with material capital, overfit signals will be flagged and disqualified. You may read about overfitting in the literature. For example, \"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality\" by De Prado and Bailey, the return values of the Optimization in the example below would allow one to easily use some of the methods in the aforementioned paper.\n",
    "\n",
    "PtolmyISoter` Participants submissions be ranked based on their utility $U$, which can be interpreted as a non-scaled Sharpe Ratio, assuming zero transaction costs. This scoring assumes:\n",
    "- **Nominal position sizes** in each asset are determined directly by the predictions $P(i)$.\n",
    "- An **offsetting hedge** is taken in an index, with size equal to the sum of predictions at each timestamp, resulting in a market-neutral strategy.\n",
    "\n",
    "1. **Accuracy**: High values of $P(i) \\cdot R(i+1)$ indicate strong predictive alignment with the relative returns.\n",
    "2. **Consistency**: Lower standard deviation of $P(i) \\cdot R(i+1)$ rewards predictions that maintain stability over time, resulting in a more robust $U$.\n",
    "\n",
       
       \"Contest.\\<mdabul@cc.cc\n", 
       "`PtolemyISoter_Submission_v1_1_4.py`\\>\" WITH EITHER the example below and create my own Predictor class in place of the submission `class MyPredictor(Predictor)` that is defined. My work will involve a)\n",
    "  creating parameters, b) optimizing and c) prediction logic, all clearly defined in the methods of the base class.   
 
       "The output of our prediction, as per the example below should be a dataframe with the following information:   
       
       For each time $i$, participants must submit a vector $P(i) = (P_1(i), P_2(i), ..., P_J(i))$ that represents their forecast for the next time step’s relative returns across all assets in the dataset.\n",
    "\n",
    "#### Guidelines\n",
    "- **Direct Signal Usage**: To ensure simplicity and transparency, we should submit $P(i) = K(i)$, meaning no additional scaling factors (such as $\\beta$) are applied to the predictions.  \n",
    "- **Robustness Across Market Conditions**: Given the high volatility and nonstationarity of markets, signals that perform well across diverse market regimes are encouraged.\n",
    "\n",
    "#### Test Set\n",
    "The evaluation will be conducted over a hidden test set, that he holds but i do not have, during which $ U $ will be calculated based on the prediction submitted by us.\n",
    "\n",
    "#### Additional Notes\n",
    "- **Data Confidentiality**: Both target returns and feature data are obfuscated. Features are labeled only by an index, and timestamps are represented as sequential integers.\n",
    "- **Scoring Interpretation**: The utility function $ U $ is designed as a **non-scaled Sharpe Ratio** assuming a zero-cost, market-neutral strategy, where nominal positions are determined by me $P(i)$ with an offsetting index hedge equal to the sum of predictions.\n",
    "\n", 
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter_Submission_v1_1_4.py",
   "metadata": {},
   "source": [
    "# class MyPredictor(Predictor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "PtolemyIStore",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from itertools import product\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "## read in data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "returns=pd.read_csv(\"returns.06.10.26.csv\",index_col=0)\n",
    "target=pd.read_csv(\"rel_returns.06.10.26.csv\",index_col=0)\n",
    "features=pd.read_csv(\"features.06.10.26.csv\",index_col=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "# split into train and validate.  please use test_size=0.25 as indicated below"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "PtolemyISoter,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_target, validate_target = train_test_split(target, test_size=0.25,shuffle=False )\n",
    "train_returns, validate_returns = train_test_split(returns, test_size=0.25,shuffle=False)\n",
    "train_features, validate_features = train_test_split(features, test_size=0.25,shuffle=False )\n",
    "\n",
    "train_data={'returns':train_returns,'features':train_features}\n",
    "\n",
    "validate_data={'returns':validate_returns,'features':validate_features}"
   ]
  },
  {
   "cell_type": "markdown",
  "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "## helper functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "def backtest(predictions,relative_returns):\n",
    "    predictions.ffill(inplace=True)\n",
    "    #the following is closely relatived to measuring the lead lag correlation of the prediction to the outcome.\n",
    "    pf_returns = (predictions.shift(1)).mul(relative_returns.values).sum(axis=1)\n",
    "    pf_returns.iloc[0] = 0  # first day return is 0, because we do not have prediction for time i=-1\n",
    "    return pf_returns\n",
    "\n",
    "def returns_to_equity(returns):\n",
    "    equity = returns.add(1).cumprod()\n",
    "    return equity\n",
    "\n",
    "def utility_sharpe(returns):\n",
    "    unscaled_sharpe=returns.mean()/returns.std()\n",
    "    return float(unscaled_sharpe)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "PtolemyISoter",
   "metadata": {
    "jupyter": {
     "is_executing": true
    }
   },
   "outputs": [],
   "source": [
    "from abc import ABC, abstractmethod\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import math\n",
    "\n",
    "def 
    ClassMyPredictor(predictor):\n",
    "    cls = globals()[predictor]\n",
    "\n",
    "    return cls()\n",
    "\n",
    "def PredictorFactory(config):\n",
    "    ind=ClassMyPredictor(config['.py'])\n",
    "    ind.set_parms(**config['parms'])\n",
    "    return ind\n",
    "\n",
    "class MyPredictor(predictor):\n",
    "\n",
    "    @abstractmethod\n",
    "    def set_parms(self, **parms):\n",
    "        pass\n",
    "\n",
    "    @abstractmethod\n",
    "    #train model to find best parameters, whether explicit or implicit to maximize utility\n",
    "    def train(self, data,target):\n",
    "        pass\n",
    "\n",
    "    @abstractmethod\n",
    "    def predict(self,data):\n",
    "        pass\n",
    "\n",
    "\n",
    "class RollingAverageReturn(MyPredictor):\n",
    "\n",
    "    def __init__(self,window=1,direction=1):\n",
    "        self._window=window\n",
    "        self._direction=direction\n",
    "\n",
    "    def set_parms(self, **parms):\n",
    "        self._window = parms['window']\n",
    "        self._direction=parms['direction']\n",
    "\n",
    "    def train(self,data,target):\n",
    "        param_space = {\n",
    "            \"window\": range(1,500),\n",
    "            \"direction\":[-1,1]\n",
    "        }\n",
    "        best_utility = -100\n",
    "        best_parms = None\n",
    "        strategy_returns_list = []\n",
    "        for values in product(*param_space.values()):\n",
    "            parms = {key: val for key, val in zip(param_space.keys(), values)}\n",
    "            self.set_parms(**parms)\n",
    "            predictions = self.predict(data)\n",
    "            strategy_returns = backtest(predictions,target)\n",
    "            utility=utility_sharpe(strategy_returns)\n",
    "\n",
    "            if utility>best_utility:\n",
    "                best_parms=parms\n",
    "                best_utility=utility\n",
    "            strategy_returns_list.append((str(parms), strategy_returns))\n",
    "\n",
    "        df_strategy_returns = pd.concat([sr for _, sr in strategy_returns_list], axis=1)\n",
    "        df_strategy_returns.columns = [name for name, _ in strategy_returns_list]\n",
    "        self.set_parms(**best_parms)\n",
    "\n",
    "        return df_strategy_returns,best_parms,float(best_utility)\n",
    "            \n",
    "                \n",
    "\n",
    "\n",
    "    def predict(self,data):\n",
    "  \n",
    "        returns=(data['returns'])\n",
    "\n",
    "        #smooths the sign of returns \n",
    "        pred = self._direction*returns.rolling(window=self._window).mean()\n",
    "\n",
    "        #normalizes so sum of absolute valuies of predictons is 1\n",
    "        pred=pred.divide(abs(pred).sum(axis=1),axis=0)\n",
    "        return pred"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "# train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictor=RollingAverageReturn()\n",
    "df_strategy_returns,best_parms,best_utility=predictor.train(train_data,train_target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'window': 1, 'direction': -1}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_parms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_utility"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "### train backtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "image",
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "returns_to_equity(df_strategy_returns[str(best_parms)]).plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "# VALIDATE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictor_best=RollingAverageReturn(**best_parms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions=predictor_best.predict(validate_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "pnl=backtest(predictions,validate_target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": [
    "utility=utility_sharpe(pnl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.00000"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "utility"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "PtolemyISoter",
   "metadata": {},
   "source": [
    "## validation backtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "image",
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "returns_to_equity(pnl).plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "PtolemyISoter",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (.py)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "PtolemyISoter",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "PtolemyISoter",
   "nbconvert_exporter": "python",
   "pygments_lexer": "python3",
   "version": "7.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
