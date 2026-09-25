"""
Ames Housing Price Prediction Pipeline
Author: harigov63
Repository: https://github.com
Description: End-to-end robust regression model combining automated regularized 
             linear models and gradient-boosted trees for the Kaggle Ames Housing competition.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# 1. DATA LOADING & PATH HANDLING
# ==========================================
print("[INFO] Loading datasets...")
try:
    # Kaggle environment pathing
    train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
    test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
except FileNotFoundError:
    # Local fallback pathing (assumes data files are in your local repository folder)
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

# ==========================================
# 2. DATA CLEANING & OUTLIER REMOVAL
# ==========================================
# Dean De Cock (dataset author) recommends removing homes with >4000 sq ft living area
print("[INFO] Handling extreme dataset outliers...")
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Isolate target variables and IDs
test_ids = test['Id']
y = np.log1p(train['SalePrice'])  # Aligning metric optimization to Log-RMSE
X = train.drop(['Id', 'SalePrice'], axis=1)
X_test = test.drop(['Id'], axis=1)

# ==========================================
# 3. FEATURE ENGINEERING INTERSECTIONS
# ==========================================
print("[INFO] Running feature engineering transformations...")
for df in [X, X_test]:
    # Total combined living and structural square footage
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Combined bathroom metric across floors
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    
    # Combined porch and deck relaxation footprint
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    
    # Chronological age metrics relative to sale period
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    
    # Textural flags transformed to intuitive binaries
    df['IsNew'] = df['YearBuilt'].apply(lambda x: 1 if x == 2010 else 0)
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

# ==========================================
# 4. PREPROCESSING PIPELINE INFRASTRUCTURE
# ==========================================
# Grouping statistical types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Robust scaling protects against lingering numerical outliers using IQR range boundaries
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Structural explicit 'None' category assignment handles missing pool/basement/garage rows safely
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# ==========================================
# 5. AUTOMATED ALPHAS & MODEL BUILDING
# ==========================================
print("[INFO] Training base architectures...")
alphas_space = [0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0]

models = {
    'RidgeCV': RidgeCV(alphas=alphas_space, cv=5),
    'LassoCV': LassoCV(alphas=[0.0001, 0.0003, 0.0005, 0.001, 0.005], cv=5, max_iter=15000),
    'ElasticNetCV': ElasticNetCV(alphas=[0.0001, 0.0003, 0.0005, 0.001], l1_ratio=[0.8, 0.85, 0.9, 0.95], cv=5, max_iter=15000),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, subsample=0.7, colsample_bytree=0.7, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=15, max_depth=4, random_state=42, verbose=-1)
}

# ==========================================
# 6. MODEL CROSS-VALIDATION
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
trained_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='negative_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-scores).mean()
    print(f" -> {name} Log-RMSE Strategy Score: {rmse:.5f}")
    
    # Train fully on full distribution space
    pipeline.fit(X, y)
    trained_pipelines[name] = pipeline

# ==========================================
# 7. BLENDED ENSEMBLE METRIC EXPORT
# ==========================================
print("[INFO] Composing final weighted submission model...")

preds_ridge = np.expm1(trained_pipelines['RidgeCV'].predict(X_test))
preds_lasso = np.expm1(trained_pipelines['LassoCV'].predict(X_test))
preds_enet = np.expm1(trained_pipelines['ElasticNetCV'].predict(X_test))
preds_gbr = np.expm1(trained_pipelines['GradientBoosting'].predict(X_test))
preds_xgb = np.expm1(trained_pipelines['XGBoost'].predict(X_test))
preds_lgb = np.expm1(trained_pipelines['LightGBM'].predict(X_test))

# Blending mathematical configurations
final_predictions = (
    (0.20 * preds_ridge) + 
    (0.20 * preds_lasso) + 
    (0.10 * preds_enet) + 
    (0.15 * preds_gbr) + 
    (0.20 * preds_xgb) + 
    (0.15 * preds_lgb)
)

# Output generation format mapping
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
submission.to_csv('submission.csv', index=False)
print("[SUCCESS] Complete file 'submission.csv' compiled and ready for Kaggle evaluation.")
