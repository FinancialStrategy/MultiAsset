import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import scipy.stats as stats
import warnings
import streamlit as st
import sys

warnings.filterwarnings('ignore')

# Portfolio optimization imports with better error handling
try:
    from pypfopt import expected_returns, risk_models, EfficientFrontier
    from pypfopt import objective_functions
    from pypfopt import CLA
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
    import cvxpy
    CVXPY_AVAILABLE = True
except ImportError as e:
    st.warning(f"""
    ⚠️ **Optimization Libraries Not Fully Loaded**
    
    Some advanced optimization methods require additional packages.
    Please install them with:
    ```
    pip install cvxpy ecos osqp scs
    ```
    
    Error: {e}
    """)
    CVXPY_AVAILABLE = False

from scipy.optimize import minimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# ============================================================
# ENHANCED CONFIGURATION WITH DYNAMIC PARAMETERS
# ============================================================

class EnhancedPortfolioConfig:
    """Extended configuration with dynamic parameters and validation"""

    def __init__(self, initial_capital: float, custom_assets: Dict[str, List[str]]):
        # Core parameters
        self.INITIAL_CAPITAL = initial_capital
        self.ANNUAL_TRADING_DAYS = 252

        # Enhanced asset universe with categorization
        self.ASSET_UNIVERSE = custom_assets

        # Flatten asset list with metadata
        self.ASSETS = []
        self.ASSET_CATEGORIES = {}
        for category, assets in self.ASSET_UNIVERSE.items():
            self.ASSETS.extend(assets)
            for asset in assets:
                self.ASSET_CATEGORIES[asset] = category

        # Benchmark - using ^GSPC (S&P 500 Index) or a regional index like XU100.IS
        self.BENCHMARK = '^GSPC' if 'US_Equities' in custom_assets else 'XU100.IS'

        # Dynamic date parameters
        self.END_DATE = datetime.now().strftime('%Y-%m-%d')
        self.START_DATE = (datetime.now() - timedelta(days=6 * 365)).strftime('%Y-%m-%d')  # 6 years

        # Optimization parameters
        self.MIN_WEIGHT = 0.01
        self.MAX_WEIGHT = 0.20
        self.MAX_SECTOR_WEIGHT = 0.35

        # Advanced risk parameters
        self.VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
        
        # New: Covariance Model Choices
        self.COV_MODELS = {
            "Sample Covariance": "sample_cov",
            "Ledoit-Wolf Shrinkage": "ledoit_wolf",
            "Oracle Approximating Shrinkage (OAS)": "oas",
            "Shrunk Covariance (Custom)": "shrunk_cov",
        }
        
        # Risk-free rate (depends on benchmark)
        self.RISK_FREE_RATE = self._get_current_risk_free_rate()

    def _get_current_risk_free_rate(self) -> float:
        """Dynamically fetch current risk-free rate"""
        try:
            # Use a high rate for a high-inflation economy like Turkey
            if self.BENCHMARK == 'XU100.IS':
                return 0.25  # Conservative 25% for Turkey
            
            # For USD portfolios, use 10-year Treasury yield
            ticker = yf.Ticker('^TNX')
            hist = ticker.history(period='5d')  # 5 days to ensure we get data
            if not hist.empty and 'Close' in hist.columns:
                current_yield = hist['Close'].iloc[-1] / 100
                # Clamp to reasonable band
                return max(0.02, min(0.08, current_yield))
        except Exception as e:
            print(f"Failed to fetch risk-free rate: {e}")
            
        # Fallback rates based on benchmark
        if self.BENCHMARK == 'XU100.IS':
            return 0.25  # Turkey fallback
        else:
            return 0.0425  # USD fallback to 4.25%

# ============================================================
# COMPREHENSIVE HISTORICAL SCENARIOS (RETAINED)
# ============================================================

class ComprehensiveHistoricalScenarios:
    SCENARIOS = {
        "COVID-19 Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
        "Q4 2018 Bear Market": ("2018-10-01", "2018-12-31"),
        "Russia-Ukraine War (Feb-Mar 2022)": ("2022-02-24", "2022-03-31"),
        "Lehman Collapse (Sep-Oct 2008)": ("2008-09-15", "2008-10-15"),
        "2022 Final Selloff (Sep-Oct 2022)": ("2022-09-01", "2022-10-31"),
    }

    def get_scenarios_in_range(self, start_date: str, end_date: str) -> Dict[str, Tuple[str, str]]:
        """Filter scenarios to include only those fully within the data range."""
        valid_scenarios = {}
        data_start = pd.to_datetime(start_date)
        data_end = pd.to_datetime(end_date)

        for name, (start, end) in self.SCENARIOS.items():
            scenario_start = pd.to_datetime(start)
            scenario_end = pd.to_datetime(end)
            if scenario_start >= data_start and scenario_end <= data_end:
                valid_scenarios[name] = (start, end)
        return valid_scenarios

# ============================================================
# ADVANCED DATA MANAGER WITH REAL HISTORICAL DATA (RETAINED/FIXED)
# ============================================================

class AdvancedDataManager:
    def __init__(self, config: EnhancedPortfolioConfig):
        self.config = config
        self.asset_prices = pd.DataFrame()
        self.asset_returns = pd.DataFrame()
        self.benchmark_prices = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.risk_free_rate = config.RISK_FREE_RATE
        self.log = []

    def _log(self, message: str):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(_self) -> bool:
        """Load and validate all required data - FIXED VERSION"""
        self = _self
        self.log.clear()
        self._log("📥 Loading comprehensive market data...")

        try:
            successful_assets = []
            asset_dataframes = []
            
            # Set yfinance timezone (improves date consistency)
            yf.set_tz_cache_location("America/New_York")

            # Download assets with progress bar
            progress_bar = st.progress(0)
            total_assets = len(self.config.ASSETS)
            
            for idx, asset in enumerate(self.config.ASSETS):
                try:
                    self._log(f"Downloading data for {asset}...")
                    
                    ticker = yf.Ticker(asset)
                    
                    # Try to download data with retry logic
                    data = None
                    for attempt in range(3):
                        try:
                            data = ticker.history(
                                start=self.config.START_DATE,
                                end=self.config.END_DATE,
                                auto_adjust=True,
                                progress=False
                            )
                            if not data.empty:
                                break
                        except Exception:
                            if attempt == 2:
                                raise
                    
                    if data is None or data.empty:
                        self._log(f"No data for {asset}")
                        continue

                    if 'Close' in data.columns:
                        close_prices = data['Close']
                        close_prices.index = close_prices.index.tz_localize(None)
                        close_prices = close_prices.rename(asset)
                        asset_dataframes.append(close_prices)
                        successful_assets.append(asset)
                        self._log(f"✅ Successfully loaded {asset}")
                    else:
                        self._log(f"No Close price for {asset}")

                except Exception as e:
                    self._log(f"Failed to download {asset}: {str(e)[:100]}...")
                    continue
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total_assets)

            progress_bar.empty()

            if len(successful_assets) < 3:
                self._log(f"❌ Insufficient assets downloaded: {len(successful_assets)} (minimum 3 required)")
                return False

            # Combine all asset prices
            if asset_dataframes:
                self.asset_prices = pd.concat(asset_dataframes, axis=1)
                self.asset_prices = self.asset_prices.ffill().dropna()
                
                # Ensure we have enough data points
                if len(self.asset_prices) < 50:
                    self._log(f"❌ Insufficient data points: {len(self.asset_prices)} (minimum 50 required)")
                    return False
                    
                # Calculate returns
                self.asset_returns = self.asset_prices.pct_change().dropna()
            else:
                self._log("❌ No asset dataframes to concatenate")
                return False

            # Download benchmark data with real historical OHLC
            benchmark_success = self._load_benchmark_data_real()
            if not benchmark_success:
                self._log("❌ Failed to load benchmark data")
                return False

            # Align dates between assets and benchmark
            self._align_data_dates()

            self._log(f"✅ Data loaded successfully: {len(self.asset_prices.columns)} assets, {len(self.asset_returns)} periods")
            return True

        except Exception as e:
            self._log(f"❌ Data loading failed: {str(e)}")
            return False

    def _load_benchmark_data_real(self) -> bool:
        """Load benchmark data using real historical OHLC data"""
        try:
            self._log(f"Downloading benchmark data for {self.config.BENCHMARK}...")
            
            benchmark_ticker = yf.Ticker(self.config.BENCHMARK)
            
            # Try to download benchmark data
            benchmark_data = None
            for attempt in range(3):
                try:
                    benchmark_data = benchmark_ticker.history(
                        start=self.config.START_DATE,
                        end=self.config.END_DATE,
                        auto_adjust=True,
                        progress=False
                    )
                    if not benchmark_data.empty:
                        break
                except Exception:
                    if attempt == 2:
                        raise
            
            if benchmark_data is None or benchmark_data.empty:
                self._log(f"❌ No benchmark data available for {self.config.BENCHMARK}")
                return False
            
            if 'Close' in benchmark_data.columns:
                # Store both prices and returns
                self.benchmark_prices = benchmark_data['Close'].copy()
                self.benchmark_prices.index = self.benchmark_prices.index.tz_localize(None)
                
                # Calculate returns
                self.benchmark_returns = self.benchmark_prices.pct_change().dropna()
                
                self._log(f"✅ Successfully loaded benchmark {self.config.BENCHMARK}")
                self._log(f"    Benchmark data points: {len(self.benchmark_prices)}")
                self._log(f"    Benchmark return periods: {len(self.benchmark_returns)}")
                
                return True
            else:
                self._log(f"❌ No Close price in benchmark data for {self.config.BENCHMARK}")
                return False
                
        except Exception as e:
            self._log(f"❌ Benchmark download failed: {str(e)}")
            return False

    def _align_data_dates(self):
        """Align dates between datasets using real data"""
        if self.asset_returns.empty or self.benchmark_returns.empty:
            self._log("⚠️ Cannot align dates: asset_returns or benchmark_returns is empty")
            return

        common_dates = self.asset_returns.index.intersection(self.benchmark_returns.index)

        if len(common_dates) > 50:  # Need sufficient overlapping data
            self.asset_returns = self.asset_returns.loc[common_dates]
            self.benchmark_returns = self.benchmark_returns.loc[common_dates]
            self.asset_prices = self.asset_prices.loc[common_dates]
            self.benchmark_prices = self.benchmark_prices.loc[common_dates]
            self._log(f"✅ Aligned data to {len(common_dates)} common dates")
        else:
            self._log(f"⚠️ Insufficient common dates: {len(common_dates)} (minimum 50 required)")
            # Try to use available data without alignment
            pass

# ============================================================
# ENHANCED PORTFOLIO OPTIMIZER - COMPLETE IMPLEMENTATION
# ============================================================

class AdvancedPortfolioOptimizer:
    """Enhanced portfolio optimization with multiple methodologies and advanced features"""

    def __init__(
        self,
        expected_returns_series: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float,
        config: EnhancedPortfolioConfig,
        log_func,
        asset_returns: pd.DataFrame = None
    ):
        self.mu = expected_returns_series
        self.S = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.config = config
        self.asset_categories = config.ASSET_CATEGORIES
        self.asset_prices = None
        self.asset_returns = asset_returns  # For momentum calculation
        self.log = log_func

    def set_asset_prices(self, asset_prices: pd.DataFrame):
        self.asset_prices = asset_prices

    def optimize_all_strategies(self) -> Dict[str, Dict]:
        """Optimize using multiple enhanced strategies with advanced techniques"""
        if not CVXPY_AVAILABLE:
            self.log("⚠️ cvxpy not available, using simplified strategies only")
            return self._get_basic_strategies()
            
        strategies = {}

        optimization_methods = [
            ('Max Sharpe', self.max_sharpe_enhanced),
            ('Min Volatility', self.min_volatility_enhanced),
            ('Risk Parity', self.risk_parity_enhanced),
            ('Max Diversification', self.max_diversification_enhanced),
            ('Equal Risk Contribution', self.equal_risk_contribution),
            ('Black-Litterman', self.black_litterman_optimization),
            ('Robust Optimization', self.robust_optimization),
            ('Hierarchical Risk Parity', self.hierarchical_risk_parity),
            ('Momentum-Adj. Max Sharpe', self.momentum_adjusted_max_sharpe),
            ('Low-Vol Factor Tilt', self.factor_tilted_max_sharpe),
        ]

        for name, method in optimization_methods:
            try:
                weights = method()
                if weights is not None and self._validate_weights(weights):
                    strategies[name] = {
                        'weights': weights,
                        'description': self._get_strategy_description(name),
                        'optimization_method': name
                    }
                    self.log(f"✅ {name} optimization completed successfully")
                else:
                    self.log(f"⚠️ {name} optimization returned invalid weights")
            except Exception as e:
                self.log(f"⚠️ {name} optimization failed: {str(e)[:100]}...")

        strategies.update(self._add_benchmark_strategies())

        self.log(f"🎯 Successfully optimized {len(strategies)} portfolio strategies")
        return strategies

    def _get_basic_strategies(self) -> Dict[str, Dict]:
        """Fallback strategies when cvxpy is not available"""
        strategies = {}
        
        basic_methods = [
            ('Equal Weight', self.equal_weight),
            ('Inverse Volatility', self.inverse_volatility),
            ('Min Volatility Basic', self.min_volatility_fallback),
            ('Risk Parity Basic', self.risk_parity_basic),
        ]
        
        for name, method in basic_methods:
            try:
                weights = method()
                if weights is not None and self._validate_weights(weights):
                    strategies[name] = {
                        'weights': weights,
                        'description': self._get_strategy_description(name),
                        'optimization_method': name
                    }
                    self.log(f"✅ {name} completed successfully")
                else:
                    self.log(f"⚠️ {name} returned invalid weights")
            except Exception as e:
                self.log(f"⚠️ {name} failed: {str(e)[:100]}...")
        
        return strategies

    # -------------------- NEW STRATEGY 1: MOMENTUM --------------------
    def momentum_adjusted_max_sharpe(self) -> Optional[Dict[str, float]]:
        """Max Sharpe using expected returns adjusted by 12-month momentum."""
        if self.asset_returns is None or len(self.asset_returns) < 252:
            self.log("Momentum calculation requires at least 1 year of data.")
            return self.max_sharpe_enhanced()

        try:
            lookback = 252
            skip = 21

            if len(self.asset_returns) < lookback + skip:
                self.log(
                    f"Not enough data for 12-month momentum ({lookback} days + {skip} skip days)."
                )
                return self.max_sharpe_enhanced()

            returns_for_mom = self.asset_returns.iloc[-(lookback + skip):-skip]
            momentum_returns = (1 + returns_for_mom).prod() - 1
            momentum_returns = momentum_returns.reindex(self.mu.index).fillna(0)

            # Simple scaling of expected returns based on momentum score
            mu_mom = self.mu * (1 + momentum_returns)

            # Normalize to current magnitude
            if mu_mom.mean() != 0:
                mu_mom = mu_mom * (self.mu.mean() / mu_mom.mean())
            else:
                mu_mom = self.mu

            ef = EfficientFrontier(mu_mom, self.S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            return self._apply_weight_rounding(weights)

        except Exception as e:
            self.log(f"Momentum-Adj. optimization failed, falling back: {e}")
            return self.max_sharpe_enhanced()

    # -------------------- NEW STRATEGY 2: FACTOR EXPOSURE --------------------
    def factor_tilted_max_sharpe(self) -> Optional[Dict[str, float]]:
        """Max Sharpe with a constraint on low volatility factor exposure (Factor Tilt)."""
        try:
            ef = EfficientFrontier(self.mu, self.S)

            # Inverse volatility as proxy for low-vol factor
            factor_exposure = 1 / np.sqrt(np.diag(self.S.values))
            factor_exposure_series = pd.Series(factor_exposure, index=self.mu.index)

            # Benchmark factor exposure
            factor_benchmark = factor_exposure_series.mean()
            tilt_target = factor_benchmark * 1.01  # 1% tilt to low-vol

            factor_vec = factor_exposure_series.values

            # pypfopt constraint requires the function to return a scalar indicating constraint violation
            def low_vol_tilt_constraint(w):
                # Constraint: w @ factor_vec >= tilt_target --> tilt_target - w @ factor_vec <= 0 (ineq)
                return w @ factor_vec - tilt_target

            # Apply standard constraints
            self._apply_enhanced_constraints(ef)
            # Apply factor tilt constraint (pypfopt assumes inequality constraints are >= 0 if type not specified)
            ef.add_constraint(low_vol_tilt_constraint, 'ineq') 

            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            # Logging exposure
            w_array = np.array([weights.get(t, 0.0) for t in self.mu.index], dtype=float)
            if w_array.sum() > 0:
                w_array = w_array / w_array.sum()
            final_factor_exposure = float(w_array @ factor_vec)
            self.log(
                f"Factor Tilt: Target Low Vol Exp >= {tilt_target:.4f}, "
                f"Actual: {final_factor_exposure:.4f}"
            )

            return self._apply_weight_rounding(weights)

        except Exception as e:
            self.log(f"Factor Tilt optimization failed, falling back: {e}")
            return self.max_sharpe_enhanced()

    # --- EXISTING STRATEGIES START ---
    def max_sharpe_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()
            return self._apply_weight_rounding(weights)
        except Exception:
            return self.max_sharpe_fallback()

    def max_sharpe_fallback(self) -> Optional[Dict[str, float]]:
        try:
            cla = CLA(self.mu, self.S)
            cla.max_sharpe()
            weights = cla.clean_weights()
            return weights
        except Exception:
            return self.inverse_volatility()

    def min_volatility_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.05)
            self._apply_enhanced_constraints(ef)
            ef.min_volatility()
            weights = ef.clean_weights()
            return self._apply_weight_rounding(weights)
        except Exception:
            return self.min_volatility_fallback()

    def min_volatility_fallback(self) -> Dict[str, float]:
        try:
            variances = np.diag(self.S.values)
            inverse_variance = 1 / np.maximum(variances, 1e-12)
            weights = inverse_variance / inverse_variance.sum()
            weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
            return self._apply_basic_constraints(weight_dict)
        except Exception:
            return self.equal_weight()

    def risk_parity_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            initial_weights = np.ones(len(volatilities)) / len(volatilities)
            weights = self._iterative_risk_parity(initial_weights)
            return self._apply_weight_rounding(weights)
        except Exception:
            return self.risk_parity_basic()

    def _iterative_risk_parity(self, initial_weights: np.ndarray, max_iter: int = 100) -> Dict[str, float]:
        weights = initial_weights.copy()
        n_assets = len(weights)

        for _ in range(max_iter):
            portfolio_variance = weights @ self.S.values @ weights
            if portfolio_variance <= 0:
                break

            marginal_risk = self.S.values @ weights
            risk_contributions = weights * marginal_risk / portfolio_variance

            target_risk = 1.0 / n_assets
            risk_ratios = risk_contributions / target_risk

            adjustment = 1.0 / np.sqrt(np.maximum(risk_ratios, 1e-12))
            weights = weights * adjustment
            weights = weights / weights.sum()

            if np.max(np.abs(risk_ratios - 1.0)) < 0.01:
                break

        weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
        return self._apply_basic_constraints(weight_dict)

    def max_diversification_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))

            def diversification_ratio(w):
                weighted_avg_vol = w @ volatilities
                portfolio_vol = np.sqrt(w @ self.S.values @ w)
                return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0

            n_assets = len(self.mu)
            initial_weights = np.ones(n_assets) / n_assets
            bounds = [(self.config.MIN_WEIGHT, self.config.MAX_WEIGHT) for _ in range(n_assets)]

            def objective(weights):
                return -diversification_ratio(weights)

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            result = minimize(objective, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            if result.success:
                weights = {asset: float(w) for asset, w in zip(self.mu.index, result.x)}
                return self._apply_weight_rounding(weights)
            else:
                return self.max_diversification_basic()
        except Exception:
            return self.max_diversification_basic()

    def max_diversification_basic(self) -> Dict[str, float]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            diversification_weights = 1 / np.maximum(volatilities, 1e-12)
            weights = diversification_weights / diversification_weights.sum()
            weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
            return self._apply_basic_constraints(weight_dict)
        except Exception:
            return self.equal_weight()

    def equal_risk_contribution(self) -> Dict[str, float]:
        try:
            n_assets = len(self.mu)
            initial_weights = np.ones(n_assets) / n_assets

            def risk_contribution_variance(weights):
                w = np.array(weights)
                portfolio_variance = w @ self.S.values @ w
                if portfolio_variance <= 0:
                    return 1.0
                marginal_risk = self.S.values @ w
                risk_contributions = w * marginal_risk / portfolio_variance
                return np.var(risk_contributions)

            bounds = [(self.config.MIN_WEIGHT, self.config.MAX_WEIGHT) for _ in range(n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            result = minimize(risk_contribution_variance, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = {asset: float(w) for asset, w in zip(self.mu.index, result.x)}
                return self._apply_weight_rounding(weights)
            else:
                return self.risk_parity_basic()
        except Exception:
            return self.risk_parity_basic()

    def black_litterman_optimization(self) -> Dict[str, float]:
        try:
            # Equal-weight proxy for market caps
            market_caps = pd.Series(1.0, index=self.mu.index)

            market_prior = market_implied_prior_returns(
                market_caps=market_caps,
                cov_matrix=self.S,
                risk_aversion=1.0
            )

            # Equilibrium-only BL (no explicit views)
            bl = BlackLittermanModel(self.S, pi=market_prior)
            bl_returns = bl.bl_returns()

            ef = EfficientFrontier(bl_returns, self.S)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            return self._apply_weight_rounding(weights)
        except Exception as e:
            self.log(f"Black-Litterman optimization failed, falling back: {e}")
            return self.max_sharpe_enhanced()

    def robust_optimization(self) -> Dict[str, float]:
        try:
            if self.asset_prices is not None:
                sample_cov = risk_models.sample_cov(self.asset_prices)
                ledoit_cov = risk_models.CovarianceShrinkage(self.asset_prices).ledoit_wolf()
                robust_cov = (sample_cov + ledoit_cov) / 2
                robust_cov = robust_cov.loc[self.mu.index, self.mu.index]
            else:
                robust_cov = self.S

            ef = EfficientFrontier(self.mu, robust_cov)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            return self._apply_weight_rounding(weights)
        except Exception:
            return self.max_sharpe_enhanced()

    def hierarchical_risk_parity(self) -> Dict[str, float]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            weights = np.ones(len(self.mu)) / len(self.mu)

            if len(self.mu) > 1 and self.asset_prices is not None:
                corr_matrix = risk_models.sample_cov(self.asset_prices).corr()
                corr_matrix = corr_matrix.loc[self.mu.index, self.mu.index].values

                dist = squareform(1 - corr_matrix)
                linkage_matrix = hierarchy.linkage(dist, method='single')
                clusters = hierarchy.fcluster(linkage_matrix, t=0.5, criterion='distance')

                for cluster_id in np.unique(clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    if len(cluster_indices) > 0:
                        cluster_vols = volatilities[cluster_indices]
                        cluster_weights = 1 / np.maximum(cluster_vols, 1e-12)
                        cluster_weights = cluster_weights / cluster_weights.sum()
                        weights[cluster_indices] = cluster_weights

                weights = weights / weights.sum()

            weight_dict = {asset: float(w) for asset, w in zip(self.mu.index, weights)}
            return self._apply_basic_constraints(weight_dict)
        except Exception as e:
            self.log(f"HRP optimization failed: {e}")
            return self.risk_parity_enhanced()

    def equal_weight(self) -> Dict[str, float]:
        n_assets = len(self.mu)
        weight = 1.0 / n_assets
        return {asset: weight for asset in self.mu.index}

    def inverse_volatility(self) -> Dict[str, float]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            inverse_vol = 1 / np.maximum(volatilities, 1e-12)
            weights = inverse_vol / inverse_vol.sum()
            weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
            return weight_dict
        except Exception:
            return self.equal_weight()

    def risk_parity_basic(self) -> Dict[str, float]:
        return self.inverse_volatility()

    def _apply_enhanced_constraints(self, ef: EfficientFrontier):
        ef.add_constraint(lambda w: w >= self.config.MIN_WEIGHT)
        ef.add_constraint(lambda w: w <= self.config.MAX_WEIGHT)

        if hasattr(self, 'asset_categories') and self.asset_categories:
            self._apply_sector_constraints(ef)

    def _apply_sector_constraints(self, ef: EfficientFrontier):
        try:
            categories = {}
            for asset, category in self.asset_categories.items():
                if asset in self.mu.index:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(asset)
            
            # Apply sector constraints if we have multiple categories
            for category, assets in categories.items():
                if len(assets) > 1:
                    # Sum of weights in this category <= MAX_SECTOR_WEIGHT
                    ef.add_constraint(lambda w, a=assets: sum(w[list(a)]) <= self.config.MAX_SECTOR_WEIGHT)
        except Exception as e:
            self.log(f"Sector constraint application failed: {e}")

    def _apply_basic_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        try:
            constrained_weights = {}
            total_weight = sum(weights.values())

            if total_weight <= 0:
                return self.equal_weight()

            min_weight = getattr(self.config, 'MIN_WEIGHT', 0.01)
            max_weight = getattr(self.config, 'MAX_WEIGHT', 0.20)

            for asset, weight in weights.items():
                normalized_weight = weight / total_weight
                constrained_weight = max(min_weight, min(max_weight, normalized_weight))
                constrained_weights[asset] = constrained_weight

            total_constrained = sum(constrained_weights.values())
            if total_constrained > 0:
                constrained_weights = {k: v / total_constrained for k, v in constrained_weights.items()}

            return constrained_weights
        except Exception:
            return self.equal_weight()

    def _apply_weight_rounding(self, weights: Dict[str, float]) -> Dict[str, float]:
        try:
            rounded_weights = {asset: round(weight, 4) for asset, weight in weights.items()}
            min_weight = getattr(self.config, 'MIN_WEIGHT', 0.01)
            cleaned_weights = {
                asset: weight for asset, weight in rounded_weights.items()
                if weight >= min_weight
            }
            total = sum(cleaned_weights.values())
            if total > 0:
                final_weights = {asset: weight / total for asset, weight in cleaned_weights.items()}
                return final_weights
            else:
                return self.equal_weight()
        except Exception:
            return self.equal_weight()

    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        if not weights:
            return False
        try:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                return False
            if any(weight < -0.001 for weight in weights.values()):
                return False
            return True
        except Exception:
            return False

    def _get_strategy_description(self, strategy_name: str) -> str:
        descriptions = {
            'Max Sharpe': 'Maximum risk-adjusted returns (Sharpe ratio) with constraints',
            'Min Volatility': 'Minimum volatility portfolio with diversification objectives',
            'Risk Parity': 'Equal risk contribution across all assets using iterative methods',
            'Max Diversification': 'Maximum diversification ratio optimization',
            'Equal Risk Contribution': 'True equal risk contribution portfolio minimizing risk concentration',
            'Black-Litterman': 'Market equilibrium model incorporating investor views (equilibrium only here)',
            'Robust Optimization': 'Optimization using a blended (sample + shrinkage) covariance estimate',
            'Hierarchical Risk Parity': 'Clustering-based risk parity for improved diversification',
            'Momentum-Adj. Max Sharpe': 'Max Sharpe using expected returns adjusted by intermediate-term (12-1 month) momentum.',
            'Low-Vol Factor Tilt': 'Max Sharpe optimization subject to a minimum exposure constraint to the Low Volatility Factor.',
            'Equal Weight': 'Equal allocation across all assets (naive diversification benchmark)',
            'Inverse Volatility': 'Risk-based allocation weighted by inverse of volatility',
            'Min Volatility Basic': 'Basic minimum volatility using inverse variance weighting',
            'Risk Parity Basic': 'Basic risk parity using inverse volatility weighting'
        }
        return descriptions.get(strategy_name, 'Advanced portfolio optimization strategy')

    def _add_benchmark_strategies(self) -> Dict[str, Dict]:
        return {
            'Equal Weight': {
                'weights': self.equal_weight(),
                'description': self._get_strategy_description('Equal Weight'),
                'optimization_method': 'Benchmark'
            },
            'Inverse Volatility': {
                'weights': self.inverse_volatility(),
                'description': self._get_strategy_description('Inverse Volatility'),
                'optimization_method': 'Benchmark'
            }
        }

# ============================================================
# COMPREHENSIVE METRICS CALCULATOR (RETAINED/FIXED)
# ============================================================

class ComprehensiveMetricsCalculator:
    @staticmethod
    def calculate_performance_metrics(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict[str, float]:
        if portfolio_returns.empty or len(portfolio_returns) < 20:
            return {}

        # Calculate returns
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        cumulative_return = total_return

        if not benchmark_returns.empty:
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
        else:
            benchmark_total_return = 0
            benchmark_annual_return = 0

        # Risk-adjusted metrics
        excess_returns = portfolio_returns - risk_free_rate / 252
        std = portfolio_returns.std()
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / std if std > 0 else 0

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

        # Alpha and Beta
        if not benchmark_returns.empty and len(benchmark_returns) > 0 and benchmark_returns.var() > 0:
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252) if len(active_returns) > 0 else 0
            information_ratio = (annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
        else:
            beta = 1.0
            alpha = 0.0
            tracking_error = 0.0
            information_ratio = 0.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
        }

    @staticmethod
    def calculate_risk_metrics(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        if portfolio_returns.empty or len(portfolio_returns) < 20:
            return {}

        # Volatility & Drawdown
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0

        # VaR and CVaR at multiple confidence levels
        var_90 = -np.percentile(portfolio_returns, 10) if len(portfolio_returns) > 10 else 0
        var_95 = -np.percentile(portfolio_returns, 5) if len(portfolio_returns) > 5 else 0
        var_99 = -np.percentile(portfolio_returns, 1) if len(portfolio_returns) > 1 else 0
        
        cvar_95 = -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() if len(portfolio_returns) > 5 else var_95
        cvar_99 = -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean() if len(portfolio_returns) > 1 else var_99

        # Higher moments
        skewness = portfolio_returns.skew() if len(portfolio_returns) > 2 else 0
        kurtosis = portfolio_returns.kurtosis() if len(portfolio_returns) > 3 else 0

        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        calmar_ratio = -annual_volatility / max_drawdown if max_drawdown < 0 else 0

        return {
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'var_90': var_90,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_std': downside_std,
            'calmar_ratio': calmar_ratio
        }

    @staticmethod
    def calculate_var_metrics(
        portfolio_returns: pd.Series,
        confidence_levels: List[float]
    ) -> Dict[str, float]:
        var_metrics = {}
        for confidence in confidence_levels:
            var_key = f'var_{int(confidence * 100)}'
            cvar_key = f'cvar_{int(confidence * 100)}'
            
            if len(portfolio_returns) > 10:
                var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
                tail_returns = portfolio_returns[portfolio_returns <= -var]
                cvar = -tail_returns.mean() if len(tail_returns) > 0 else var
            else:
                var = 0
                cvar = 0
                
            var_metrics[var_key] = var
            var_metrics[cvar_key] = cvar
        return var_metrics

# ============================================================
# ENHANCED STRESS TEST CALCULATOR (RETAINED)
# ============================================================

class EnhancedStressTestCalculator:
    @staticmethod
    def calculate_stress_metrics(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        try:
            start_dt = pd.to_datetime(start_date).tz_localize(None)
            end_dt = pd.to_datetime(end_date).tz_localize(None)

            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)

            mask = (portfolio_returns.index >= start_dt) & (portfolio_returns.index <= end_dt)
            stress_portfolio = portfolio_returns.loc[mask]
            stress_benchmark = benchmark_returns.loc[mask]

            if len(stress_portfolio) < 5:
                return {}

            portfolio_return = (1 + stress_portfolio).prod() - 1
            benchmark_return = (1 + stress_benchmark).prod() - 1
            relative_return = portfolio_return - benchmark_return

            portfolio_cumulative = (1 + stress_portfolio).cumprod()
            portfolio_peak = portfolio_cumulative.expanding().max()
            portfolio_drawdown = (portfolio_cumulative - portfolio_peak) / portfolio_peak
            max_drawdown = portfolio_drawdown.min()

            volatility = stress_portfolio.std() * np.sqrt(252) if len(stress_portfolio) > 1 else 0
            var_95 = -np.percentile(stress_portfolio, 5) if len(stress_portfolio) > 5 else 0
            cvar_95 = -stress_portfolio[
                stress_portfolio <= np.percentile(stress_portfolio, 5)
            ].mean() if len(stress_portfolio) > 5 else 0

            if len(stress_portfolio) > 1 and len(stress_benchmark) > 1 and stress_benchmark.var() > 0:
                covariance = stress_portfolio.cov(stress_benchmark)
                benchmark_variance = stress_benchmark.var()
                stress_beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            else:
                stress_beta = 1.0

            return {
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'relative_return': relative_return,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'stress_beta': stress_beta,
                'days_in_stress': len(stress_portfolio),
                'start_date': start_date,
                'end_date': end_date
            }

        except Exception as e:
            print(f"Stress test calculation failed: {e}")
            return {}

# ============================================================
# ADVANCED VISUALIZATION ENGINE WITH STREAMLIT OUTPUT (ENHANCED)
# ============================================================

class AdvancedVisualizationEngine:
    COLORS = {
        'performance': '#34495E',
        'risk': '#1ABC9C',
        'var': '#2980B9',
        'stress': '#9B59B6',
        'strategy': '#E67E22',
        'positive': '#27AE60',
        'negative': '#E74C3C',
        'neutral': '#95A5A6'
    }

    def _format_dataframe(
        self,
        df: pd.DataFrame,
        header_color: str,
        percentage_cols: List[str],
        ratio_cols: List[str]
    ) -> st.delta_generator.DeltaGenerator:
        """Formats a DataFrame for Streamlit display with conditional styling."""

        format_dict = {}
        for col in percentage_cols:
            if col in df.columns:
                format_dict[col] = '{:+.2%}'
        for col in ratio_cols:
            if col in df.columns:
                format_dict[col] = '{:.3f}'

        df_styled = df.style.format(format_dict).set_table_styles([
            {'selector': 'th', 'props': [('background-color', header_color), ('color', 'white')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}
        ])

        return st.dataframe(df_styled, use_container_width=True)
    
    def create_comprehensive_performance_table(
        self,
        data: Dict[str, Dict[str, float]]
    ) -> st.delta_generator.DeltaGenerator:
        df = pd.DataFrame(data).T
        performance_metrics = [
            'annual_return',
            'cumulative_return',
            'sharpe_ratio',
            'sortino_ratio',
            'information_ratio',
            'alpha',
            'beta',
            'tracking_error'
        ]
        available_metrics = [m for m in performance_metrics if m in df.columns]
        df = df[available_metrics]

        display_names = {
            'annual_return': 'Annual Return',
            'cumulative_return': 'Cumulative Return',
            'sharpe_ratio': 'Sharpe Ratio',
            'sortino_ratio': 'Sortino Ratio',
            'information_ratio': 'Info Ratio',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'tracking_error': 'Tracking Error'
        }
        df = df.rename(columns=display_names)

        percentage_cols = ['Annual Return', 'Cumulative Return', 'Tracking Error', 'Alpha']
        ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Info Ratio', 'Beta']

        return self._format_dataframe(df, self.COLORS['performance'], percentage_cols, ratio_cols)

    def create_comprehensive_risk_table(
        self,
        data: Dict[str, Dict[str, float]]
    ) -> st.delta_generator.DeltaGenerator:
        df = pd.DataFrame(data).T
        risk_metrics = [
            'annual_volatility',
            'downside_std',
            'max_drawdown',
            'skewness',
            'kurtosis',
            'var_95',
            'cvar_95',
            'calmar_ratio'
        ]
        available_metrics = [m for m in risk_metrics if m in df.columns]
        df = df[available_metrics]

        display_names = {
            'annual_volatility': 'Annual Vol',
            'downside_std': 'Downside Vol',
            'max_drawdown': 'Max Drawdown',
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis',
            'var_95': 'VaR 95%',
            'cvar_95': 'CVaR 95%',
            'calmar_ratio': 'Calmar Ratio'
        }
        df = df.rename(columns=display_names)

        percentage_cols = ['Annual Vol', 'Downside Vol', 'Max Drawdown', 'VaR 95%', 'CVaR 95%']
        ratio_cols = ['Skewness', 'Kurtosis', 'Calmar Ratio']

        return self._format_dataframe(df, self.COLORS['risk'], percentage_cols, ratio_cols)

    def create_advanced_var_table(
        self,
        data: Dict[str, Dict[str, float]],
        confidence_levels: List[float]
    ) -> st.delta_generator.DeltaGenerator:
        df = pd.DataFrame(data).T
        var_metrics = [
            f'var_{int(conf * 100)}' for conf in confidence_levels
        ] + [
            f'cvar_{int(conf * 100)}' for conf in confidence_levels
        ]
        available_metrics = [m for m in var_metrics if m in df.columns]
        df = df[available_metrics]

        display_names = {f'var_{int(c * 100)}': f'VaR {int(c * 100)}%' for c in confidence_levels}
        display_names.update({f'cvar_{int(c * 100)}': f'CVaR {int(c * 100)}%' for c in confidence_levels})
        df = df.rename(columns=display_names)

        percentage_cols = df.columns.tolist()

        return self._format_dataframe(df, self.COLORS['var'], percentage_cols, [])

    def create_historical_stress_analysis_table(
        self,
        stress_results: Dict[str, Dict[str, Any]]
    ) -> Optional[st.delta_generator.DeltaGenerator]:
        if not stress_results:
            st.info("No stress test data available within the analysis period.")
            return None

        df = pd.DataFrame(stress_results).T
        stress_metrics = [
            'portfolio_return',
            'benchmark_return',
            'relative_return',
            'max_drawdown',
            'volatility',
            'var_95',
            'cvar_95',
            'stress_beta',
            'days_in_stress'
        ]
        available_metrics = [m for m in stress_metrics if m in df.columns]
        df = df[available_metrics]

        display_names = {
            'portfolio_return': 'Port. Ret',
            'benchmark_return': 'Bench. Ret',
            'relative_return': 'Relative Ret',
            'max_drawdown': 'Max DD',
            'volatility': 'Stress Vol',
            'var_95': 'VaR 95%',
            'cvar_95': 'CVaR 95%',
            'stress_beta': 'Stress Beta',
            'days_in_stress': 'Duration Days'
        }
        df = df.rename(columns=display_names)

        percentage_cols = ['Port. Ret', 'Bench. Ret', 'Relative Ret', 'Max DD', 'Stress Vol', 'VaR 95%', 'CVaR 95%']
        ratio_cols = ['Stress Beta', 'Duration Days']

        return self._format_dataframe(df, self.COLORS['stress'], percentage_cols, ratio_cols)

    def create_advanced_efficient_frontier(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        strategies: Dict[str, Dict],
        risk_free_rate: float
    ) -> go.Figure:
        try:
            # --- Robust Input Checks ---
            mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
            S = S.loc[mu.index, mu.index]
            if len(mu) < 2:
                st.warning("Fewer than two assets after sanitization.")
                return go.Figure().add_annotation(
                    text="Insufficient assets for efficient frontier",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )

            # --- Calculate Frontier and Key Points ---
            ef_min = EfficientFrontier(mu, S)
            min_ret, min_vol, _ = ef_min.min_volatility().portfolio_performance()

            ef_ms = EfficientFrontier(mu, S)
            ms_ret, ms_vol, ms_sharpe = ef_ms.max_sharpe(
                risk_free_rate=risk_free_rate
            ).portfolio_performance()

            target_returns = np.linspace(min_ret, ms_ret * 1.5, 60)
            efficient_vols, efficient_rets = [], []
            for tr in target_returns:
                try:
                    ef_p = EfficientFrontier(mu, S)
                    ef_p.efficient_return(target_return=tr)
                    r, v, _ = ef_p.portfolio_performance()
                    efficient_rets.append(r)
                    efficient_vols.append(v)
                except Exception:
                    continue

            fig = go.Figure()

            # 1. Efficient Frontier Line
            if efficient_vols:
                fig.add_trace(go.Scatter(
                    x=efficient_vols,
                    y=efficient_rets,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color=self.COLORS['performance'], width=3),
                    hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))

            # 2. Individual Assets
            individual_volatilities = np.sqrt(np.diag(S.values))
            fig.add_trace(go.Scatter(
                x=individual_volatilities,
                y=mu.values,
                mode='markers',
                name='Individual Assets',
                marker=dict(size=10, color=self.COLORS['risk'], symbol='circle'),
                text=list(mu.index),
                hovertemplate='<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))

            # 3. Strategy Points
            for strategy_name, strategy_info in strategies.items():
                weights = strategy_info.get('weights', {})
                if weights:
                    w = np.array([weights.get(t, 0.0) for t in mu.index], dtype=float)
                    w = w / np.sum(w) if np.sum(w) > 0 else w
                    portfolio_return = float(np.dot(w, mu.values))
                    portfolio_vol = float(np.sqrt(w @ S.values @ w))

                    fig.add_trace(go.Scatter(
                        x=[portfolio_vol],
                        y=[portfolio_return],
                        mode='markers',
                        name=strategy_name,
                        marker=dict(
                            size=16,
                            symbol='star',
                            color=self.COLORS['strategy'],
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=(
                            f'<b>{strategy_name}</b>'
                            '<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                        )
                    ))

            # 4. Capital Market Line
            if ms_vol and ms_vol > 0:
                sharpe_ratio = (ms_ret - risk_free_rate) / ms_vol
                x_max = max(
                    (efficient_vols if efficient_vols else [0])
                    + individual_volatilities.tolist()
                ) * 1.25
                x_range = np.linspace(0, x_max, 50)
                cml_returns = risk_free_rate + sharpe_ratio * x_range

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=cml_returns,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color=self.COLORS['var'], dash='dash', width=2),
                    hoverinfo='skip'
                ))

            fig.update_layout(
                title_text='📈 Advanced Efficient Frontier Analysis',
                xaxis_title="Annual Volatility (Risk)",
                yaxis_title="Annual Return",
                hovermode='closest',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                width=None,
                height=650,
                template="plotly_white",
                xaxis=dict(tickformat='.0%'),
                yaxis=dict(tickformat='.0%')
            )
            return fig

        except Exception as e:
            st.error(f"Efficient frontier plotting failed: {e}")
            return go.Figure().add_annotation(
                text="Efficient frontier not available",
                x=0.5,
                y=0.5,
                showarrow=False
            )

    def create_weight_allocation_chart(self, strategies: Dict[str, Dict]) -> go.Figure:
        if not strategies:
            return go.Figure().add_annotation(
                text="No strategies available",
                x=0.5,
                y=0.5,
                showarrow=False
            )

        all_assets = sorted(list(set(
            a for info in strategies.values() for a in info.get('weights', {}).keys()
        )))

        data = []
        for strategy_name, strategy_info in strategies.items():
            weights = strategy_info.get('weights', {})
            weight_values = [weights.get(asset, 0) for asset in all_assets]

            data.append(go.Bar(
                name=strategy_name,
                x=all_assets,
                y=weight_values,
                hovertemplate=(
                    '<b>%{x}</b><br>Strategy: ' + strategy_name +
                    '<br>Weight: %{y:.2%}<extra></extra>'
                )
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            title_text='📊 Portfolio Allocation by Strategy',
            barmode='group',
            xaxis=dict(title="Assets"),
            yaxis=dict(title="Weight Allocation", tickformat='.0%'),
            width=None,
            height=600,
            template="plotly_white"
        )
        return fig

    # -------- Institutional Strategy Factsheet (New Panel) --------
    def create_strategy_factsheet(
        self,
        strategy_name: str,
        performance_data: Dict[str, Dict[str, float]],
        risk_data: Dict[str, Dict[str, float]],
        var_data: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ):
        if not strategy_name:
            st.info("No optimal strategy selected.")
            return

        perf = performance_data.get(strategy_name, {})
        risk = risk_data.get(strategy_name, {})
        var = var_data.get(strategy_name, {})

        if not perf or not risk:
            st.info("Insufficient metrics to build strategy factsheet.")
            return

        st.subheader(f"🏦 Institutional Strategy Factsheet – {strategy_name}")

        # --- Metrics Table ---
        metrics = {
            'Annual Return': perf.get('annual_return', np.nan),
            'Annual Volatility': risk.get('annual_volatility', np.nan),
            'Sharpe Ratio': perf.get('sharpe_ratio', np.nan),
            'Sortino Ratio': perf.get('sortino_ratio', np.nan),
            'Max Drawdown': risk.get('max_drawdown', np.nan),
            'VaR 95%': risk.get('var_95', np.nan),
            'CVaR 95%': risk.get('cvar_95', np.nan),
        }

        metrics_df = pd.DataFrame(metrics, index=[strategy_name])

        percentage_cols = ['Annual Return', 'Annual Volatility', 'Max Drawdown', 'VaR 95%', 'CVaR 95%']
        ratio_cols = ['Sharpe Ratio', 'Sortino Ratio']

        st.markdown("#### Key Risk–Return Metrics")
        self._format_dataframe(metrics_df, self.COLORS['performance'], percentage_cols, ratio_cols)

        # --- Weights Table ---
        if weights:
            w_series = pd.Series(weights, name='Weight').sort_values(ascending=False)
            w_df = w_series.to_frame()
            st.markdown("#### Asset Allocation (Weights)")
            st.dataframe(
                w_df.style.format({'Weight': '{:.2%}'}).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', self.COLORS['strategy']), ('color', 'white')]},
                    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}
                ]),
                use_container_width=True
            )

    # -------- New: Monte Carlo Simulation Chart --------
    def plot_monte_carlo_results(self, mc_data: Dict[str, Any]) -> go.Figure:
        if not mc_data:
            return go.Figure().add_annotation(text="No Monte Carlo data available", x=0.5, y=0.5, showarrow=False)

        simulations = mc_data['simulations']
        percentiles = mc_data['percentiles']
        
        # Calculate daily cumulative returns for plotting
        sim_df = pd.DataFrame(simulations)
        sim_cum_returns = sim_df.apply(lambda x: (1 + x).cumprod(), axis=0)

        fig = go.Figure()
        
        # 1. Plot all paths (faded)
        for col in sim_cum_returns.columns[:100]:  # Limit to 100 paths for performance
            fig.add_trace(go.Scatter(
                y=sim_cum_returns[col],
                line=dict(width=0.5, color='rgba(52, 152, 219, 0.1)'), # Light Blue/Faded
                showlegend=False,
                hoverinfo='skip'
            ))

        # 2. Plot key percentiles (Median, 5th, 95th)
        median_path = sim_cum_returns.median(axis=1)
        p5_path = sim_cum_returns.quantile(0.05, axis=1)
        p95_path = sim_cum_returns.quantile(0.95, axis=1)

        fig.add_trace(go.Scatter(
            y=median_path, name=f'Median Return ({percentiles["median_final"]:.2%})',
            line=dict(color=self.COLORS['performance'], width=3),
            hovertemplate='Day: %{x}<br>Median Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            y=p95_path, name=f'95th Percentile ({percentiles["p95_final"]:.2%})',
            line=dict(color=self.COLORS['positive'], width=1.5, dash='dash'),
            hovertemplate='Day: %{x}<br>95th Value: %{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            y=p5_path, name=f'5th Percentile (Worst Case) ({percentiles["p5_final"]:.2%})',
            line=dict(color=self.COLORS['negative'], width=1.5, dash='dash'),
            hovertemplate='Day: %{x}<br>5th Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title_text=f'🔮 Monte Carlo Simulation ({mc_data["n_sims"]} Paths over 1 Year)',
            xaxis_title="Simulation Day (Next 252 Days)",
            yaxis_title="Portfolio Value (Initial Value = 1.0)",
            template="plotly_white", height=650
        )
        return fig

# ============================================================
# MAIN PORTFOLIO ENGINE (ENHANCED)
# ============================================================

class QFAPortfolioEngine:
    def __init__(self, config: EnhancedPortfolioConfig, log_func):
        self.config = config
        self.log_func = log_func
        self.data_manager = AdvancedDataManager(self.config)
        self.visualizer = AdvancedVisualizationEngine()
        self.metrics_calculator = ComprehensiveMetricsCalculator()
        self.stress_calculator = EnhancedStressTestCalculator()

        self.asset_prices = None
        self.asset_returns = None
        self.benchmark_returns = None
        self.risk_free_rate = None
        self.cov_model_name = "Ledoit-Wolf Shrinkage" # Default
        self.monte_carlo_results = {}

        self.strategies = {}
        self.comprehensive_performance_data = {}
        self.comprehensive_risk_data = {}
        self.var_data = {}
        self.stress_test_results = {}

    def set_cov_model(self, model_name: str):
        self.cov_model_name = model_name
        self.log_func(f"Covariance model set to: {model_name}")

    @staticmethod
    def _ensure_psd(S: pd.DataFrame, min_eig: float = 1e-10) -> pd.DataFrame:
        try:
            A = S.values
            A = 0.5 * (A + A.T)
            vals, vecs = np.linalg.eigh(A)
            vals_clipped = np.clip(vals, min_eig, None)
            A_psd = (vecs @ np.diag(vals_clipped) @ vecs.T)
            A_psd = 0.5 * (A_psd + A_psd.T)
            return pd.DataFrame(A_psd, index=S.index, columns=S.columns)
        except Exception:
            return S.copy()

    def _get_covariance_matrix(self, model_name: str) -> pd.DataFrame:
        """Dynamically select and compute the covariance matrix."""
        method = self.config.COV_MODELS.get(model_name, "ledoit_wolf")
        
        self.log_func(f"Calculating Covariance using {model_name}...")

        if method == "sample_cov":
            S = risk_models.sample_cov(self.asset_prices)
        elif method == "oas":
            try:
                # Try to use scikit-learn's OAS
                from sklearn.covariance import OAS
                oas = OAS()
                returns_array = self.asset_prices.pct_change().dropna().values
                oas.fit(returns_array)
                S = pd.DataFrame(oas.covariance_, 
                                index=self.asset_prices.columns, 
                                columns=self.asset_prices.columns)
            except ImportError:
                # Fallback to Ledoit-Wolf if scikit-learn not available
                self.log_func("⚠️ scikit-learn not available, using Ledoit-Wolf instead of OAS")
                S = risk_models.CovarianceShrinkage(self.asset_prices).ledoit_wolf()
        elif method == "shrunk_cov":
            S = risk_models.CovarianceShrinkage(self.asset_prices, shrinkage=0.5).shrunk_covariance() # Custom 50% shrinkage
        elif method == "ledoit_wolf":
            S = risk_models.CovarianceShrinkage(self.asset_prices).ledoit_wolf()
        else:
            S = risk_models.CovarianceShrinkage(self.asset_prices).ledoit_wolf()
            
        return S

    def _compute_sanitized_mu_cov(self) -> Tuple[pd.Series, pd.DataFrame]:
        try:
            mu = expected_returns.mean_historical_return(self.asset_prices)
        except Exception:
            mu = self.asset_returns.mean() * self.config.ANNUAL_TRADING_DAYS

        try:
            S = self._get_covariance_matrix(self.cov_model_name)
        except Exception as e:
            self.log_func(f"Error calculating covariance: {e}. Falling back to Sample Cov.")
            S = risk_models.sample_cov(self.asset_prices)

        mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
        S = S.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').dropna(axis=1, how='any')

        keep = mu.index.intersection(S.index)
        mu = mu.loc[keep]
        S = S.loc[keep, keep]

        diag = np.diag(S.values)
        valid = diag > 1e-12
        if not np.all(valid):
            mu = mu.iloc[valid]
            S = S.iloc[valid, valid]

        S = self._ensure_psd(S)

        if len(mu) < 2:
            raise ValueError("Insufficient valid assets after sanitization (need at least 2).")

        self.log_func(f"Sanitized asset set for EF: {list(mu.index)}")
        return mu, S

    def _run_monte_carlo_simulation(self, weights: Dict[str, float], n_sims: int = 1000, n_days: int = 252):
        """Enhanced Monte Carlo simulation with better error handling"""
        
        self.log_func(f"🔮 Running Monte Carlo Simulation ({n_sims} paths)...")
        
        try:
            # 1. Prepare inputs with validation
            mu, S = self._compute_sanitized_mu_cov()
            
            # Filter weights to match sanitized assets
            weights_array = np.array([weights.get(t, 0.0) for t in mu.index], dtype=float)
            
            # Check if weights sum to approximately 1
            weight_sum = weights_array.sum()
            if abs(weight_sum - 1.0) > 0.01:
                self.log_func(f"⚠️ Weights sum to {weight_sum:.2f}, normalizing")
                weights_array = weights_array / weight_sum
            
            # Annualized portfolio metrics
            port_mu = np.dot(weights_array, mu.values)
            port_sigma = np.sqrt(weights_array @ S.values @ weights_array)
            
            # Check for valid parameters
            if port_sigma <= 1e-12:
                self.log_func("⚠️ Portfolio sigma too low for meaningful MC simulation")
                return
            
            # Daily metrics
            daily_mu = port_mu / self.config.ANNUAL_TRADING_DAYS
            daily_sigma = port_sigma / np.sqrt(self.config.ANNUAL_TRADING_DAYS)
            
            # 2. Generate simulations
            np.random.seed(42)  # For reproducibility
            simulations = np.random.normal(
                daily_mu, 
                daily_sigma, 
                size=(n_days, n_sims)
            )
            
            # 3. Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + simulations, axis=0)
            
            # 4. Calculate statistics
            final_values = cumulative_returns[-1, :] - 1  # Convert to returns
            final_percentiles = {
                'p5_final': np.percentile(final_values, 5),
                'median_final': np.percentile(final_values, 50),
                'p95_final': np.percentile(final_values, 95)
            }
            
            # Calculate worst/best paths
            worst_path_idx = np.argmin(final_values)
            best_path_idx = np.argmax(final_values)
            
            self.monte_carlo_results = {
                'simulations': simulations,  # Daily returns
                'cumulative_returns': cumulative_returns,  # Cumulative values
                'n_sims': n_sims,
                'n_days': n_days,
                'percentiles': final_percentiles,
                'worst_path': cumulative_returns[:, worst_path_idx],
                'best_path': cumulative_returns[:, best_path_idx],
                'expected_return': port_mu,
                'expected_volatility': port_sigma
            }
            
            self.log_func("✅ Monte Carlo Simulation Complete.")
            
        except Exception as e:
            self.log_func(f"❌ Monte Carlo simulation failed: {str(e)}")
            st.error(f"Monte Carlo simulation failed: {str(e)}")

    def run_analysis(self):
        self.log_func("🚀 Starting Comprehensive Portfolio Analysis...")

        try:
            if not CVXPY_AVAILABLE:
                self.log_func("⚠️ cvxpy not available. Some optimization methods will be limited.")
                st.warning("⚠️ cvxpy not properly installed. Some advanced optimization methods will be limited. Please ensure cvxpy is installed.")

            # Validate data loading
            if not self.data_manager.load_data():
                st.error("❌ Data loading failed. Please check asset tickers and try again.")
                return

            self.asset_prices = self.data_manager.asset_prices
            self.asset_returns = self.data_manager.asset_returns
            self.benchmark_returns = self.data_manager.benchmark_returns
            self.risk_free_rate = self.data_manager.risk_free_rate

            # Check if we have sufficient data
            if self.asset_returns.empty or len(self.asset_returns) < 50:
                st.error(f"❌ Insufficient data for analysis. Only {len(self.asset_returns) if not self.asset_returns.empty else 0} periods available.")
                return

            mu, S = self._compute_sanitized_mu_cov()

            optimizer = AdvancedPortfolioOptimizer(
                mu,
                S,
                self.risk_free_rate,
                self.config,
                self.log_func,
                asset_returns=self.asset_returns
            )

            optimizer.set_asset_prices(self.asset_prices)
            self.strategies = optimizer.optimize_all_strategies()

            self._calculate_comprehensive_metrics()
            self._run_enhanced_stress_tests()
            
            # Automatically run MC for the Max Sharpe strategy
            optimal_strategy = self.get_optimal_strategy("sharpe")
            if optimal_strategy and optimal_strategy in self.strategies:
                self._run_monte_carlo_simulation(self.strategies[optimal_strategy]['weights'])

            self.log_func("✅ Comprehensive Portfolio Analysis Complete!")

        except ValueError as e:
            if "Insufficient valid assets" in str(e):
                st.error("""
                ❌ **Insufficient Data for Analysis**
                
                Please check:
                1. All ticker symbols are correct
                2. Assets have sufficient historical data
                3. Try removing assets that might not be publicly traded
                """)
            else:
                st.error(f"❌ Analysis failed: {str(e)}")
            self.log_func(f"❌ Analysis failed: {str(e)}")
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            self.log_func(f"❌ Analysis failed: {str(e)}")

    def _get_portfolio_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns without re-normalizing weights"""
        if self.asset_returns is None or self.asset_returns.empty:
            return pd.Series(dtype=float)
        
        # Create empty series with correct index
        portfolio_returns = pd.Series(0.0, index=self.asset_returns.index)
        
        # Filter assets that exist in returns data
        valid_assets = [
            asset for asset in weights.keys() 
            if asset in self.asset_returns.columns and weights[asset] > 0
        ]
        
        if not valid_assets:
            return pd.Series(dtype=float)
        
        # Calculate weighted returns without re-normalization
        for asset in valid_assets:
            portfolio_returns += self.asset_returns[asset] * weights[asset]
        
        return portfolio_returns.dropna()

    def _run_enhanced_stress_tests(self):
        self.log_func("📉 Running enhanced historical stress tests...")
        self.stress_test_results = {}
        if not self.strategies:
            return

        primary_strategy = self.get_optimal_strategy("sharpe")
        if not primary_strategy or primary_strategy not in self.strategies:
            primary_strategy = list(self.strategies.keys())[0] # Fallback to first one
            
        portfolio_returns = self._get_portfolio_returns(self.strategies[primary_strategy]['weights'])

        if portfolio_returns.empty:
            return

        available_scenarios = ComprehensiveHistoricalScenarios().get_scenarios_in_range(
            self.asset_returns.index.min().strftime('%Y-%m-%d'),
            self.asset_returns.index.max().strftime('%Y-%m-%d')
        )

        for scenario_name, (start_date, end_date) in available_scenarios.items():
            stress_metrics = self.stress_calculator.calculate_stress_metrics(
                portfolio_returns,
                self.benchmark_returns,
                start_date,
                end_date
            )
            if stress_metrics:
                self.stress_test_results[scenario_name] = stress_metrics
                self.log_func(f"✅ Stress test completed: {scenario_name}")

    def _calculate_comprehensive_metrics(self):
        self.log_func("📊 Calculating comprehensive metrics...")
        self.comprehensive_performance_data = {}
        self.comprehensive_risk_data = {}
        self.var_data = {}

        for strategy_name, strategy_info in self.strategies.items():
            portfolio_returns = self._get_portfolio_returns(strategy_info['weights'])

            if portfolio_returns.empty or len(portfolio_returns) < 20:
                self.log_func(f"⚠️ Insufficient data for {strategy_name} metrics")
                continue

            performance_metrics = self.metrics_calculator.calculate_performance_metrics(
                portfolio_returns,
                self.benchmark_returns,
                self.risk_free_rate
            )
            risk_metrics = self.metrics_calculator.calculate_risk_metrics(
                portfolio_returns,
                self.benchmark_returns
            )
            var_metrics = self.metrics_calculator.calculate_var_metrics(
                portfolio_returns,
                self.config.VAR_CONFIDENCE_LEVELS
            )

            self.comprehensive_performance_data[strategy_name] = performance_metrics
            self.comprehensive_risk_data[strategy_name] = risk_metrics
            self.var_data[strategy_name] = var_metrics

        # Add benchmark metrics if available
        if not self.benchmark_returns.empty and len(self.benchmark_returns) >= 20:
            benchmark_performance = self.metrics_calculator.calculate_performance_metrics(
                self.benchmark_returns,
                self.benchmark_returns,
                self.risk_free_rate
            )
            benchmark_risk = self.metrics_calculator.calculate_risk_metrics(
                self.benchmark_returns,
                self.benchmark_returns
            )
            benchmark_var = self.metrics_calculator.calculate_var_metrics(
                self.benchmark_returns,
                self.config.VAR_CONFIDENCE_LEVELS
            )

            self.comprehensive_performance_data['Benchmark'] = benchmark_performance
            self.comprehensive_risk_data['Benchmark'] = benchmark_risk
            self.var_data['Benchmark'] = benchmark_var

    def get_optimal_strategy(self, criteria: str = "sharpe") -> str:
        """Get optimal strategy based on specified criteria"""
        if not self.comprehensive_performance_data:
            return ""

        best_strategy = ""
        best_value = -float('inf')

        for strategy, metrics in self.comprehensive_performance_data.items():
            if strategy == 'Benchmark':
                continue

            if criteria == "return":
                value = metrics.get('annual_return', -float('inf'))
            elif criteria == "risk_adjusted":
                value = metrics.get('sortino_ratio', -float('inf'))
            else:  # sharpe
                value = metrics.get('sharpe_ratio', -float('inf'))

            if value > best_value:
                best_value = value
                best_strategy = strategy

        return best_strategy

# ============================================================
# STREAMLIT UI INTEGRATION
# ============================================================

def get_default_global_assets():
    return {
        'US_Equities': ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA'],
        'International_Equities': ['VEA', 'VWO', 'EWJ', 'EZU'],
        'Fixed_Income': ['BND', 'TLT', 'IEF', 'LQD', 'MUB', 'HYG'],
        'Real_Assets': ['GLD', 'SLV', 'VNQ', 'GSG', 'DBB'],
        'Sectors': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU']
    }


def get_turkish_equities():
    # Optimized list for better performance on Streamlit Cloud
    return {
        'Turkish_Equities': [
            'GARAN.IS', 'HALKB.IS', 'AKBNK.IS', 'THYAO.IS', 'EREGL.IS',
            'TUPRS.IS', 'SAHOL.IS', 'ASELS.IS', 'BIMAS.IS', 'FROTO.IS',
            'KCHOL.IS', 'SISE.IS', 'YKBNK.IS', 'KOZAA.IS', 'TCELL.IS'
        ]
    }


def run_streamlit_tab(tab_title: str, config: EnhancedPortfolioConfig):
    st.header(f"🏛️ {tab_title}")

    # Simple logging display
    log_placeholder = st.empty()

    # Simple logger function to update the placeholder
    def update_log(message):
        if 'log_history' not in st.session_state:
            st.session_state['log_history'] = []
        st.session_state['log_history'].append(message)
        log_placeholder.code('\n'.join(st.session_state['log_history'][-10:]), language='text')

    if 'log_history' not in st.session_state:
        st.session_state['log_history'] = []

    engine = QFAPortfolioEngine(config, update_log)

    st.sidebar.markdown(f"### {tab_title} Configuration")
    
    # NEW: Covariance Model Selection
    cov_model_choice = st.sidebar.selectbox(
        "Select Covariance Model (Robustness)",
        options=list(config.COV_MODELS.keys()),
        index=1, # Ledoit-Wolf
        key=f'cov_model_{tab_title}'
    )
    engine.set_cov_model(cov_model_choice)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Initial Capital:** ${config.INITIAL_CAPITAL:,.0f}")
    st.sidebar.markdown(f"**Risk-Free Rate:** {config.RISK_FREE_RATE:.2%}")
    st.sidebar.markdown(f"**Benchmark:** {config.BENCHMARK}")
    st.sidebar.markdown(f"**Assets:** {len(config.ASSETS)}")


    if st.button(f"✨ Run {tab_title} Analysis", key=f"run_button_{tab_title}"):
        # Clear previous MC results before running a new analysis
        engine.monte_carlo_results = {}
        with st.spinner("Analyzing portfolio, please wait..."):
            engine.run_analysis()

    if engine.strategies:
        st.success("✅ Analysis complete!")

        optimal_sharpe = engine.get_optimal_strategy("sharpe")

        st.markdown("---")

        st.subheader("🎯 Optimal Strategy Insight (Max Sharpe)")
        if optimal_sharpe:
            st.info(f"The **Maximum Sharpe Ratio** strategy is: **{optimal_sharpe}**.")
            sharpe_value = engine.comprehensive_performance_data[optimal_sharpe].get('sharpe_ratio', 0)
            st.markdown(f"**Sharpe Ratio:** `{sharpe_value:.3f}`")
        else:
            st.info("No optimal strategy identified.")

        # --- Tabbed Results ---
        tab_eff, tab_mc, tab_weights, tab_perf, tab_risk, tab_stress, tab_fact = st.tabs([
            "Efficient Frontier",
            "Monte Carlo Risk",
            "Asset Allocation",
            "Performance Metrics",
            "Risk Metrics (VaR/CVaR)",
            "Stress Testing",
            "Institutional Factsheet"
        ])

        with tab_eff:
            st.subheader("📈 Advanced Efficient Frontier Analysis")
            mu, S = engine._compute_sanitized_mu_cov()
            fig_eff = engine.visualizer.create_advanced_efficient_frontier(
                mu,
                S,
                engine.strategies,
                engine.risk_free_rate
            )
            st.plotly_chart(fig_eff, use_container_width=True)

            st.subheader("Strategy Characteristics Comparison")
            strategy_data = {}
            for strategy_name, strategy_info in engine.strategies.items():
                weights = strategy_info.get('weights', {})
                total_assets = len([w for w in weights.values() if w > 0.001])
                max_weight = max(weights.values()) if weights else 0
                concentration = sum(w ** 2 for w in weights.values())

                strategy_data[strategy_name] = {
                    'Description': strategy_info.get('description', ''),
                    'Assets Count': total_assets,
                    'Max Weight': f"{max_weight:.1%}",
                    'Concentration Index (HHI)': f"{concentration:.3f}",
                    'Diversification Score': f"{1 - concentration:.3f}"
                }
            st.dataframe(pd.DataFrame(strategy_data).T, use_container_width=True)
            
        with tab_mc:
            st.subheader(f"🔮 1-Year Monte Carlo Simulation for {optimal_sharpe or 'Optimal'} Portfolio")
            if engine.monte_carlo_results:
                mc_data = engine.monte_carlo_results
                st.markdown(f"""
                    **Expected Terminal Return (Median):** `{mc_data['percentiles']['median_final']:.2%}`  
                    **95% Confidence Worst Case (5th Percentile):** `{mc_data['percentiles']['p5_final']:.2%}`
                """)
                fig_mc = engine.visualizer.plot_monte_carlo_results(mc_data)
                st.plotly_chart(fig_mc, use_container_width=True)
                st.markdown("---")
                st.caption("This simulation assumes a Normal distribution of daily portfolio returns based on the Max Sharpe portfolio's calculated historical mean and volatility.")
            else:
                st.info("Monte Carlo results not available. Run analysis first.")

        with tab_weights:
            st.subheader("📊 Portfolio Weight Allocation by Strategy")
            fig_weights = engine.visualizer.create_weight_allocation_chart(engine.strategies)
            st.plotly_chart(fig_weights, use_container_width=True)

        with tab_perf:
            st.subheader("📈 Performance Metrics Comparison")
            engine.visualizer.create_comprehensive_performance_table(
                engine.comprehensive_performance_data
            )

        with tab_risk:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⚠️ Core Risk Metrics")
                engine.visualizer.create_comprehensive_risk_table(
                    engine.comprehensive_risk_data
                )
            with col2:
                st.subheader("🛡️ Value at Risk (VaR) & CVaR")
                engine.visualizer.create_advanced_var_table(
                    engine.var_data,
                    config.VAR_CONFIDENCE_LEVELS
                )

        with tab_stress:
            st.subheader("🌪️ Historical Stress Testing Results")
            engine.visualizer.create_historical_stress_analysis_table(
                engine.stress_test_results
            )

        with tab_fact:
            if optimal_sharpe and optimal_sharpe in engine.strategies:
                engine.visualizer.create_strategy_factsheet(
                    optimal_sharpe,
                    engine.comprehensive_performance_data,
                    engine.comprehensive_risk_data,
                    engine.var_data,
                    engine.strategies[optimal_sharpe]['weights']
                )
            else:
                st.info("No optimal strategy factsheet available.")

# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main_app():
    st.set_page_config(
        layout="wide",
        page_title="QFA Institutional Portfolio Optimization",
        page_icon="🏆"
    )

    st.title("🏆 QFA Institutional Portfolio Optimization")
    st.markdown("### Advanced Optimization and Risk Analysis for Institutional Mandates")
    st.markdown("---")
    
    # Check for cvxpy installation
    global CVXPY_AVAILABLE
    if not CVXPY_AVAILABLE:
        st.warning("""
        ⚠️ **cvxpy is not fully loaded!**
        
        Some advanced optimization methods will be limited. Please ensure `cvxpy` and its dependencies (`ecos`) are installed.
        """)

    # --- Global Configuration Sidebar ---
    st.sidebar.title("Global Configuration")

    # USD Portfolio Config
    usd_capital = st.sidebar.number_input(
        "USD Portfolio Capital ($):",
        min_value=1_000_000,
        max_value=1_000_000_000,
        value=100_000_000,
        step=5_000_000,
        key="usd_cap"
    )

    # TRY Portfolio Config
    try_capital = st.sidebar.number_input(
        "TRY Portfolio Capital (₺):",
        min_value=1_000_000,
        max_value=100_000_000,
        value=10_000_000,
        step=1_000_000,
        key="try_cap"
    )

    # Data source info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Source")
    st.sidebar.markdown("All data is sourced from **Yahoo Finance** using real historical OHLC prices.")
    st.sidebar.markdown("**Note:** Analysis uses 6 years of historical data.")

    # --- Main Tabs ---
    tab_usd, tab_try = st.tabs([
        "🌍 Global Mandate (USD)",
        "🇹🇷 Turkish Equities Mandate (TRY)"
    ])

    with tab_usd:
        usd_config = EnhancedPortfolioConfig(usd_capital, get_default_global_assets())
        run_streamlit_tab("Global Mandate (USD)", usd_config)

    with tab_try:
        try_config = EnhancedPortfolioConfig(try_capital, get_turkish_equities())
        run_streamlit_tab("Turkish Equities Mandate (TRY)", try_config)
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>📊 <b>QFA Institutional Portfolio Optimization</b> | Powered by PyPortfolioOpt, yfinance, and Streamlit</p>
    <p>⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only. Past performance is not indicative of future results.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main_app()