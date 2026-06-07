#!/usr/bin/env python
"""
Local Testing Script for AlphaNova Competition Submission

Usage:
  python test_runner.py alphanova_predictor.py
  python test_runner.py alphanova_predictor.py --full
"""

import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import time


class TestRunner:
    """Test and validate AlphaNova submissions locally."""
    
    def __init__(self, submission_path: str):
        self.submission_path = Path(submission_path)
        self.predictor = None
        self.errors = []
        self.warnings = []
        
    def load_submission(self) -> bool:
        """Load and validate submission file."""
        try:
            spec = __import__('importlib.util').util.spec_from_file_location(
                'submission', self.submission_path
            )
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'Predictor'):
                self.errors.append("❌ Predictor class not found")
                return False
            
            self.predictor = module.Predictor()
            print("✅ Submission loaded successfully")
            return True
            
        except Exception as e:
            self.errors.append(f"❌ Failed to load submission: {e}")
            traceback.print_exc()
            return False
    
    def create_dummy_data(self, n_periods: int = 5, n_assets: int = 20, 
                         n_features: int = 6) -> tuple:
        """Create dummy data for testing."""
        np.random.seed(42)
        
        data = []
        targets = []
        
        for period in range(n_periods):
            # Create feature data
            feature_data = np.random.randn(n_assets, n_features) * 0.1
            
            # Create returns (targets) and de-mean them
            returns = np.random.randn(n_assets) * 0.05
            returns = returns - returns.mean()
            targets.append(returns)
            
            # Create MultiIndex DataFrame
            tickers = [f'ticker.{i}' for i in range(n_assets)]
            features = [f'Feature.{i+1}' for i in range(n_features)]
            
            df_list = []
            for feat_idx, feat in enumerate(features):
                for asset_idx, ticker in enumerate(tickers):
                    df_list.append({
                        'period': period,
                        'feature_name': feat,
                        'ticker': ticker,
                        'value': feature_data[asset_idx, feat_idx]
                    })
            
            data.extend(df_list)
        
        # Create DataFrames
        data_df = pd.DataFrame(data)
        X = data_df.set_index(['period', 'feature_name', 'ticker'])['value']
        X = X.unstack('feature_name')
        
        y = pd.Series(np.concatenate(targets))
        
        return X, y
    
    def test_demeaning(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Test if predictions are de-meaned."""
        try:
            print("\n📊 Testing de-meaning...")
            
            # Train
            print("  Training...")
            start = time.time()
            self.predictor.train(X, y)
            train_time = time.time() - start
            print(f"  ✅ Training completed in {train_time:.2f}s")
            
            # Predict
            print("  Predicting...")
            start = time.time()
            predictions = self.predictor.predict(X)
            pred_time = time.time() - start
            print(f"  ✅ Prediction completed in {pred_time:.2f}s")
            
            # Check de-meaning
            pred_sum = predictions.sum()
            print(f"  Signal sum: {pred_sum:.2e}")
            
            if np.abs(pred_sum) > 1e-8:
                self.errors.append(f"❌ NOT_DEMEANED: Sum is {pred_sum:.2e}")
                return False
            
            print("  ✅ Signal properly de-meaned")
            return True
            
        except Exception as e:
            self.errors.append(f"❌ CANT_RUN: {e}")
            traceback.print_exc()
            return False
    
    def test_speed(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Test if training and prediction are fast enough."""
        try:
            print("\n⏱️  Testing speed...")
            
            # Training speed
            print("  Testing training speed...")
            start = time.time()
            self.predictor.train(X, y)
            train_time = time.time() - start
            
            if train_time > 240:  # 4 minutes
                self.errors.append(f"❌ TRAINING_TOO_SLOW: {train_time:.1f}s")
                return False
            
            print(f"  ✅ Training: {train_time:.2f}s (limit: 240s)")
            
            # Prediction speed
            print("  Testing prediction speed...")
            start = time.time()
            predictions = self.predictor.predict(X)
            pred_time = time.time() - start
            
            if pred_time > 60:  # 60 seconds
                self.errors.append(f"❌ PREDICTION_TOO_SLOW: {pred_time:.1f}s")
                return False
            
            print(f"  ✅ Prediction: {pred_time:.2f}s (limit: 60s)")
            return True
            
        except Exception as e:
            self.errors.append(f"❌ CANT_RUN: {e}")
            return False
    
    def test_output_shape(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Test output shape and type."""
        try:
            print("\n📐 Testing output shape...")
            
            self.predictor.train(X, y)
            predictions = self.predictor.predict(X)
            
            # Check type
            if not isinstance(predictions, (pd.Series, np.ndarray)):
                self.errors.append(f"❌ Invalid output type: {type(predictions)}")
                return False
            
            # Check shape
            expected_shape = X.columns.get_level_values('ticker').nunique()
            actual_shape = len(predictions) if isinstance(predictions, pd.Series) else predictions.shape[0]
            
            if actual_shape != expected_shape:
                self.errors.append(f"❌ Wrong shape: got {actual_shape}, expected {expected_shape}")
                return False
            
            print(f"  ✅ Output shape correct: {actual_shape}")
            return True
            
        except Exception as e:
            self.errors.append(f"❌ CANT_RUN: {e}")
            return False
    
    def run_all_tests(self, full: bool = False) -> bool:
        """Run all tests."""
        print("🧪 AlphaNova Competition - Local Test Runner\n")
        print(f"Submission: {self.submission_path}")
        print("=" * 50)
        
        # Load submission
        if not self.load_submission():
            self.print_report()
            return False
        
        # Create test data
        print("\n📥 Creating test data...")
        if full:
            X, y = self.create_dummy_data(n_periods=10, n_assets=20, n_features=6)
            print("  ✅ Created 10-period dataset")
        else:
            X, y = self.create_dummy_data(n_periods=3, n_assets=20, n_features=6)
            print("  ✅ Created 3-period dataset")
        
        # Run tests
        tests = [
            ("De-meaning", lambda: self.test_demeaning(X, y)),
            ("Output Shape", lambda: self.test_output_shape(X, y)),
            ("Speed", lambda: self.test_speed(X, y)),
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            if not test_func():
                all_passed = False
        
        self.print_report()
        return all_passed
    
    def print_report(self):
        """Print test report."""
        print("\n" + "=" * 50)
        print("📋 TEST REPORT")
        print("=" * 50)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  {error}")
        else:
            print("\n✅ All tests passed!")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        print("\n" + "=" * 50)
        print("💡 Tips:")
        print("  - Ensure predict() de-means output: signal.sub(signal.mean())")
        print("  - All code must be inside Predictor class")
        print("  - No look-ahead bias (no future data usage)")
        print("  - Keep training < 4 minutes, prediction < 60 seconds")
        print("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py <submission.py> [--full]")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    full_test = "--full" in sys.argv
    
    runner = TestRunner(submission_file)
    success = runner.run_all_tests(full=full_test)
    
    sys.exit(0 if success else 1)
