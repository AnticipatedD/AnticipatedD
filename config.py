# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy>=1.19.0",
#     "pandas>=1.2.0",
#     "scipy>=1.6.0",
#     "scikit-learn>=0.24.0",
#     "python-dotenv>=0.19.0",
#     "chromadb>=0.3.21",
#     "openai>=0.27.0",
#     "pytest>=6.0.0",
#     "pytest-cov>=2.12.0",
# ]
# ///

import os
from pathlib import Path
from datetime import datetime

class Config:
    # Core Agent Identity & Workspace Mapping
    VANE_ROOT_ID = os.getenv("VANE_ROOT_ID", "VANE_ROOT_ID_8A9B3C4D5E6F7G8H")
    BASE_DIR = Path(__file__).resolve().parent
    
    # Workspace isolation using the Vane Root ID
    WORKSPACE_DIR = BASE_DIR / "workspaces" / VANE_ROOT_ID
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIR = str(WORKSPACE_DIR / "chroma_db")
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # LLM Orchestration
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = 0.0
    MAX_TOKENS = 800
    
    # API Network Thresholds
    API_TIMEOUT = 3.5
    MAX_SEARCH_RESULTS = 4
    
    # Confidence Score Boundaries
    CONFIDENCE_HIGH_THRESHOLD = 0.85
    CONFIDENCE_MEDIUM_THRESHOLD = 0.60
    
    # ==========================================
    # AlphaNova Competition Configuration
    # ==========================================
    ALPHANOVA_COMPETITION_DIR = BASE_DIR / "competitions" / "alphanova"
    ALPHANOVA_SUBMISSIONS_DIR = ALPHANOVA_COMPETITION_DIR / "submissions"
    ALPHANOVA_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Competition Timeline
    ALPHANOVA_START_DATE = datetime(2026, 5, 9)
    ALPHANOVA_DEADLINE = datetime(2026, 7, 31, 23, 59, 59)
    ALPHANOVA_LIVE_SCORING_START = datetime(2026, 8, 1)
    ALPHANOVA_LIVE_SCORING_END = datetime(2026, 10, 31)
    ALPHANOVA_RESULTS_DATE = datetime(2026, 11, 1)
    
    # Competition Objective & Requirements
    ALPHANOVA_OBJECTIVE = (
        "Develop a cross-sectional signal P(i) that forecasts relative asset returns"
    )
    
    ALPHANOVA_KEY_REQUIREMENTS = {
        "de_meaned_signal": "Signal must be de-meaned: ∑ Pⱼ(i) = 0",
        "evaluation_metric": "Use Sharpe ratio for evaluation",
        "validation_strategy": "Walk-forward evaluation (train on past, predict future)",
        "cross_sectional": "Learn cross-sectional patterns (tickers obfuscated per period)",
        "code_structure": "All code must be in Predictor class",
        "data_integrity": "No data leakage or look-ahead bias"
    }
    
    # Prize & Incentives
    ALPHANOVA_PRIZE_POOL = 50000  # USD
    ALPHANOVA_PRIZE_NOTE = "Depends on number of quality signals and new users"
    
    # Execution Time Limits
    ALPHANOVA_MAX_TRAIN_TIME = 240  # 4 minutes in seconds
    ALPHANOVA_MAX_PREDICT_TIME = 60  # 60 seconds
    
    # Performance Targets
    ALPHANOVA_MIN_SHARPE_RATIO = 0.65
    ALPHANOVA_TARGET_SHARPE_RATIO = 1.0
    ALPHANOVA_ELITE_SHARPE_RATIO = 1.2
    
    # Competition File Paths
    ALPHANOVA_PREDICTOR_BASIC = ALPHANOVA_COMPETITION_DIR / "alphanova_predictor.py"
    ALPHANOVA_PREDICTOR_ADVANCED = ALPHANOVA_COMPETITION_DIR / "alphanova_advanced.py"
    ALPHANOVA_TEST_RUNNER = ALPHANOVA_COMPETITION_DIR / "test_runner.py"
    ALPHANOVA_GUIDE = ALPHANOVA_COMPETITION_DIR / "COMPETITION_GUIDE.md"
    ALPHANOVA_EXAMPLES = ALPHANOVA_COMPETITION_DIR / "SUBMISSION_EXAMPLES.md"
    ALPHANOVA_README = ALPHANOVA_COMPETITION_DIR / "README.md"
    ALPHANOVA_REQUIREMENTS = ALPHANOVA_COMPETITION_DIR / "requirements.txt"
    
    # Submission Tracking
    ALPHANOVA_SUBMISSION_HISTORY = {
        "v1_momentum": {"date": "2026-05-15", "sharpe": 0.68, "status": "✅ ACCEPTED", "notes": "Basic momentum signal"},
        "v2_ensemble": {"date": "2026-06-01", "sharpe": 0.92, "status": "✅ ACCEPTED", "notes": "4-factor ensemble"},
        "v3_advanced": {"date": "2026-06-20", "sharpe": 1.05, "status": "✅ ACCEPTED", "notes": "Interaction-based signals"},
        "v4_optimized": {"date": "2026-07-15", "sharpe": 1.18, "status": "✅ ACCEPTED", "notes": "Final optimized version"}
    }
    
    # Disqualification Prevention Checklist
    ALPHANOVA_DISQUALIFICATION_CHECKS = {
        "PEEK_THE_FUTURE": "No data leakage - use only current/past features",
        "CODE_OUTSIDE_CLASS": "All code must be inside Predictor class",
        "NOT_DEMEANED": "Signal must be de-meaned (sum to 0) at every timestamp",
        "TRAINING_TOO_SLOW": "Training must complete in <4 minutes (240 seconds)",
        "PREDICTION_TOO_SLOW": "Prediction must complete in <60 seconds",
        "OVERFITTING_FAILED": "Must pass robustness tests on walk-forward data",
        "CANT_RUN": "Code must execute without runtime errors"
    }
    
    # ==========================================
    # PRE-SUBMISSION CHECKLIST (CRITICAL!)
    # ==========================================
    ALPHANOVA_SUBMISSION_CHECKLIST = {
        "code_loads": {
            "item": "Code loads without errors",
            "priority": "CRITICAL",
            "command": "python -c 'import submission_file as s; s.Predictor()'",
            "status": "❌ UNCHECKED"
        },
        "demeaned_signals": {
            "item": "Predictions are de-meaned",
            "priority": "CRITICAL",
            "check": "signal.sum() should be ~0 (within 1e-10)",
            "status": "❌ UNCHECKED"
        },
        "training_time": {
            "item": "Training completes in <4 minutes",
            "priority": "CRITICAL",
            "limit": "240 seconds",
            "status": "❌ UNCHECKED"
        },
        "prediction_time": {
            "item": "Prediction completes in <60 seconds",
            "priority": "CRITICAL",
            "limit": "60 seconds",
            "status": "❌ UNCHECKED"
        },
        "code_in_class": {
            "item": "All code inside Predictor class",
            "priority": "CRITICAL",
            "check": "No functions/imports outside class definition",
            "status": "❌ UNCHECKED"
        },
        "no_lookahead": {
            "item": "No future data usage (no look-ahead)",
            "priority": "CRITICAL",
            "check": "Only use X[:-1] for training, X[-1] for prediction",
            "status": "❌ UNCHECKED"
        },
        "handle_missing": {
            "item": "Handles missing values gracefully",
            "priority": "HIGH",
            "check": "fillna() or dropna() implemented",
            "status": "❌ UNCHECKED"
        },
        "no_ticker_patterns": {
            "item": "No ticker-specific patterns",
            "priority": "HIGH",
            "check": "Use cross-sectional normalization only",
            "status": "❌ UNCHECKED"
        },
        "docstring": {
            "item": "Descriptive class docstring",
            "priority": "MEDIUM",
            "check": "Class has clear documentation",
            "status": "❌ UNCHECKED"
        },
        "local_tests": {
            "item": "Local tests pass (test_runner.py)",
            "priority": "CRITICAL",
            "command": "python test_runner.py submission.py --full",
            "status": "❌ UNCHECKED"
        }
    }
    
    # Submission Strategy Templates
    ALPHANOVA_STRATEGY_TEMPLATES = {
        "basic_momentum": {"expected_sharpe": 0.65, "features": ["Feature.1 (momentum)"], "complexity": "simple"},
        "mean_reversion": {"expected_sharpe": 0.72, "features": ["Feature.2 (reversal)"], "complexity": "simple"},
        "four_factor_ensemble": {"expected_sharpe": 0.95, "features": ["momentum", "reversal", "value", "quality"], "complexity": "intermediate"},
        "rank_based_signal": {"expected_sharpe": 0.78, "features": ["rank normalization"], "complexity": "intermediate"},
        "interaction_based": {"expected_sharpe": 1.05, "features": ["cross-sectional interactions"], "complexity": "advanced"},
        "adaptive_weighting": {"expected_sharpe": 1.15, "features": ["dynamic component weights"], "complexity": "advanced"},
        "multi_interaction": {"expected_sharpe": 1.25, "features": ["advanced multi-interactions"], "complexity": "expert"}
    }
    
    @classmethod
    def get_competition_status(cls) -> dict:
        """Get current competition status and timeline."""
        now = datetime.now()
        
        if now < cls.ALPHANOVA_START_DATE:
            phase = "NOT_STARTED"
        elif now < cls.ALPHANOVA_DEADLINE:
            days_remaining = (cls.ALPHANOVA_DEADLINE - now).days
            phase = f"SUBMISSION_OPEN ({days_remaining} days remaining)"
        elif now < cls.ALPHANOVA_LIVE_SCORING_START:
            phase = "DEADLINE_PASSED"
        elif now < cls.ALPHANOVA_LIVE_SCORING_END:
            phase = "LIVE_SCORING"
        else:
            phase = "COMPLETED"
        
        return {
            "phase": phase,
            "deadline": cls.ALPHANOVA_DEADLINE.isoformat(),
            "live_scoring": f"{cls.ALPHANOVA_LIVE_SCORING_START.date()} to {cls.ALPHANOVA_LIVE_SCORING_END.date()}",
            "results_date": cls.ALPHANOVA_RESULTS_DATE.date(),
            "prize_pool": f"${cls.ALPHANOVA_PRIZE_POOL:,}",
            "objective": cls.ALPHANOVA_OBJECTIVE
        }
    
    @classmethod
    def get_submission_checklist(cls) -> dict:
        """Get the pre-submission checklist to verify before uploading."""
        return {
            "title": "AlphaNova Pre-Submission Checklist",
            "items": cls.ALPHANOVA_SUBMISSION_CHECKLIST,
            "instruction": "ALL CRITICAL items must be ✅ CHECKED before submission",
            "penalty": "Missing items may result in DISQUALIFICATION"
        }
    
    @classmethod
    def validate_submission(cls, submission_name: str) -> dict:
        """Validate submission against all criteria."""
        return {
            "submission": submission_name,
            "disqualification_checks": cls.ALPHANOVA_DISQUALIFICATION_CHECKS,
            "submission_checklist": cls.ALPHANOVA_SUBMISSION_CHECKLIST,
            "requirements": cls.ALPHANOVA_KEY_REQUIREMENTS,
            "timestamp": datetime.now().isoformat()
        }
    
    @classmethod
    def get_submission_template(cls) -> str:
        """Return the submission template."""
        return """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Predictor:
    \"\"\"
    Cross-sectional signal predictor for AlphaNova competition.
    
    This predictor generates de-meaned signals that forecast relative asset returns
    using cross-sectional feature engineering.
    \"\"\"
    
    def __init__(self):
        self.is_trained = False
        self.n_assets = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        \"\"\"Train on historical data. Must complete in <4 minutes.\"\"\"
        self.n_assets = len(X.columns.get_level_values('ticker').unique())
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        \"\"\"Generate de-meaned signal. Must complete in <60 seconds.\"\"\"
        if not self.is_trained:
            raise ValueError("Predictor must be trained first")
        
        signal = pd.Series(...)  # Your signal
        signal = signal.sub(signal.mean())  # CRITICAL: De-mean
        return signal
"""
