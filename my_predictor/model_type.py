
[1]:
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
read in data
[2]:
returns=pd.read_csv("returns.06.10.26.csv",index_col=0)
target=pd.read_csv("rel_returns.06.10.26.csv",index_col=0)
features=pd.read_csv("features.06.10.26.csv",index_col=0)
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

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "Partner",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "### Challenge Title: Predicting Relative Returns in assets Deadline Dec 6 2024 midnight UTC\n",
    "\n",
    "#### Objective\n",
    "Participants are challenged to develop a predictive signal that forecasts the relative returns of a set of assets. The goal is to create a signal $P(i)$ at each timestamp $i$ that effectively predicts the next period’s excess returns for multiple assets compared to a market index. The objective is to maximize a utility function $U$, which measures both the predictive accuracy and consistency of the signal in forecasting these relative returns.\n",
    "\n",
    "#### Background\n",
    "Financial markets are known for their nonstationarity and complex dynamics. The assets here, which will remain masked,  enable leveraged trading without an expiry date, are particularly popular and offer unique opportunities for relative return prediction. This challenge tests participants’ ability to forecast returns that are not only accurate but also consistent over time.\n",
    "\n",
    "#### Data Description\n",
    "- **Input Data:** A time series of returns from assets, $X(i) = (X_1(i), X_2(i), ..., X_J(i))$, where $i$ denotes the time index expressed as an integer, and each $X_j(i)$ represents the return of the $j$-th asset at time $i$.\n",
    "- **Prediction Target:** At each time $i$, participants will predict a vector signal $P(i) = (P_1(i), P_2(i), ..., P_J(i))$, where each $P_j(i)$ represents the participant’s prediction for the relative return of the $j$-th asset in the next time step, $i+1$, compared to the index.\n",
    "\n",
    "   > **Note:** The actual values of the target relative returns, $R(i+1)$, are obfuscated to maintain data confidentiality, and participants will work with pre-processed, relative return values that do not reveal underlying raw prices.\n",
    "\n",
    "- **Feature Data:** In addition to historical return data, participants will have access to a set of obfuscated features. Each timestamp $i$ is associated with a feature vector $F(i) = (F_1(i), F_2(i), ..., F_K(i))$, where $K$ is the number of available features, and each feature is identified only by its index (e.g., $F_1, F_2, \\dots, F_K$).\n",
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
    "For now we will ignore overfitting tests, but in the future, when we are up an running with material capital, overfit signals will be flagged and disqualified. You may read about overfitting in the literature. For example, \"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality\" by De Prado and Bailey. We encourage you to do your own overfitting tests. the return values of the Optimization in the example below would allow one to easily use some of the methods in the aforementioned paper.\n",
    "\n",
    "Participants submissions be ranked based on their utility $U$, which can be interpreted as a non-scaled Sharpe Ratio, assuming zero transaction costs. This scoring assumes:\n",
    "- **Nominal position sizes** in each asset are determined directly by the predictions $P(i)$.\n",
    "- An **offsetting hedge** is taken in an index, with size equal to the sum of predictions at each timestamp, resulting in a market-neutral strategy.\n",
    "\n",
    "1. **Accuracy**: High values of $P(i) \\cdot R(i+1)$ indicate strong predictive alignment with the relative returns.\n",
    "2. **Consistency**: Lower standard deviation of $P(i) \\cdot R(i+1)$ rewards predictions that maintain stability over time, resulting in a more robust $U$.\n",
    "\n",
    "#### Submission Format\n",
    "The submission should be a copy of this python code files, or a .ipynb file with same content as this template, except for your specific predictor. PLEASE NAME YOUR NOTEBOOK IN THE FORM \"Contest.\\<your email\n",
    " address\\>\" WITH EITHER THE RESPECTIVE NOTEBOOK OR .PY SUFFIX. Follow the example below and create your own Predictor class in place of the example class that is defined. Your work will involve a)\n",
    "  creating parameters, b) optimizing and c) prediction logic, all clearly defined in the methods of the base class.   You do *not* have to follow the example's optimization approach, which is rudimentary brute force search over a defined grid of parameters.\n",
    "\n",
    "The output of our prediction, as per the example below should be a dataframe with the following information:  For each time $i$, participants must submit a vector $P(i) = (P_1(i), P_2(i), ..., P_J(i))$ that represents their forecast for the next time step’s relative returns across all assets in the dataset.\n",
    "\n",
    "#### Guidelines\n",
    "- **Direct Signal Usage**: To ensure simplicity and transparency, we should submit $P(i) = K(i)$, meaning no additional scaling factors (such as $\\beta$) are applied to the predictions.  \n",
    "- **Robustness Across Market Conditions**: Given the high volatility and nonstationarity of markets, signals that perform well across diverse market regimes are encouraged.\n",
    "\n",
    "#### Winning Criteria\n",
    "We are with the highest utility $U$ over the Test Set will be declared the winner. In the event that multiple users by chance have the same winning utility, we will reward the full prize to the contestant of our submitted solution *first*.\n",
    "\n",
    "#### Test Set\n",
    "The evaluation will be conducted over a hidden test set, that he hold but you do not have, during which $ U $ will be calculated based on the prediction submitted by us \n",
    "\n",
    "#### Additional Notes\n",
    "- **Data Confidentiality**: Both target returns and feature data are obfuscated. Features are labeled only by an index, and timestamps are represented as sequential integers.\n",
    "- **Scoring Interpretation**: The utility function $ U $ is designed as a **non-scaled Sharpe Ratio** assuming a zero-cost, market-neutral strategy, where nominal positions are determined by me $P(i)$ with an offsetting index hedge equal to the sum of predictions.\n",
    "\n",
    "#### Submission Deadline\n",
    "This will be detailed in the email in which this was attached.\n",
    "\n",
    "#### Important Note for the avoidance of doubt\n",
    "The only things we need to change in our version of this python file are a) possible additional imports of modules and b) replace the example Prediction class with its own parameters, optimize and predict methods.\n",
    "\n",
    "#### We have provided a requirements.txt, which is very short, but this is due to feedback on pandas/numpy incompatibility.  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "myid",
   "metadata": {},
   "source": [
    "# class MyPredictor(Predictor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "myID",
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
   "id": "myID",
   "metadata": {},
   "source": [
    "## read in data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "myID",
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
   "id": "myid",
   "metadata": {},
   "source": [
    "# split into train and validate.  please use test_size=0.25 as indicated below"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "myid,
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
  "id": "myid",
   "metadata": {},
   "source": [
    "## helper functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "myid",
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
   "id": "myid",
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
    "def ClassMyPredictor(predictor):\n",
    "    cls = globals()[predictor]\n",
    "\n",
    "    return cls()\n",
    "\n",
    "def PredictorFactory(config):\n",
    "    ind=ClassMyPredictor(config['type'])\n",
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
   "id": "myid",
   "metadata": {},
   "source": [
    "# train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "myid",
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
   "id": "myid",
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
   "id": "myid",
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
   "id": "myid",
   "metadata": {},
   "source": [
    "### train backtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "myid",
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
   "id": "myid",
   "metadata": {},
   "source": [
    "# VALIDATE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "myid",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictor_best=RollingAverageReturn(**best_parms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "myid",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions=predictor_best.predict(validate_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "myid",
   "metadata": {},
   "outputs": [],
   "source": [
    "pnl=backtest(predictions,validate_target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "myid",
   "metadata": {},
   "outputs": [],
   "source": [
    "utility=utility_sharpe(pnl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "myid",
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
   "id": "myid",
   "metadata": {},
   "source": [
    "## validation backtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "myid",
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
   "id": "myid",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "myid",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "myid",
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
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "python3",
   "version": "6.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
