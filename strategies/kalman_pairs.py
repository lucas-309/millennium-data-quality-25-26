"""Kalman filter pairs trading with dynamic hedge ratio.

Replaces the rolling OLS hedge ratio in classic pairs trading with a proper
online state-space estimator. The Kalman filter adapts faster to regime changes
and handles time-varying cointegration relationships naturally.

State: [alpha_t, beta_t] (intercept and hedge ratio)
Observation: price_a_t = alpha_t + beta_t * price_b_t + epsilon_t

Reference: Halls-Moore "Successful Algorithmic Trading" Ch 17.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from strategies.order_generator import OrderGenerator


class KalmanPairsOrderGenerator(OrderGenerator):
    """Kalman filter-based pairs trading.

    At each step, update the hedge ratio online via Kalman recursion. Trade the
    spread when z-score exceeds entry_zscore, exit at exit_zscore buffer.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 4.0,
        allocation_per_leg: float = 0.10,
        obs_noise: float = 0.001,
        process_noise: float = 1e-5,
    ):
        self.pairs = pairs
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.allocation_per_leg = allocation_per_leg
        self.obs_noise = obs_noise
        self.process_noise = process_noise

    def _run_kalman(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Run Kalman filter with state [alpha, beta]."""
        n = len(y)
        delta = self.process_noise
        Vw = delta / (1 - delta) * np.eye(2)  # process noise covariance
        Ve = self.obs_noise                    # observation noise variance

        theta = np.zeros(2)
        P = np.eye(2) * 1e6  # prior state covariance (diffuse)

        alphas = np.full(n, np.nan)
        betas = np.full(n, np.nan)
        residuals = np.full(n, np.nan)
        resid_var = np.full(n, np.nan)

        y_vals = y.values
        x_vals = x.values

        for t in range(n):
            if np.isnan(y_vals[t]) or np.isnan(x_vals[t]):
                continue

            H = np.array([1.0, x_vals[t]])  # observation matrix row
            yhat = H @ theta
            residual = y_vals[t] - yhat
            P_pred = P + Vw
            S = H @ P_pred @ H + Ve  # innovation variance
            K = P_pred @ H / S       # Kalman gain
            theta = theta + K * residual
            P = P_pred - np.outer(K, H @ P_pred)

            alphas[t] = theta[0]
            betas[t] = theta[1]
            residuals[t] = residual
            resid_var[t] = S

        return pd.DataFrame({
            "alpha": alphas,
            "beta": betas,
            "residual": residuals,
            "resid_var": resid_var,
        }, index=y.index)

    def generate_orders(self, data: pd.DataFrame):
        orders = []

        for ticker_a, ticker_b in self.pairs:
            if ticker_a not in data.columns or ticker_b not in data.columns:
                continue

            y = data[ticker_a]
            x = data[ticker_b]
            valid = y.notna() & x.notna()
            y = y[valid]
            x = x[valid]
            if len(y) < 50:
                continue

            kalman = self._run_kalman(y, x)
            resid = kalman["residual"]
            var = kalman["resid_var"].clip(lower=1e-12)
            zscore = resid / np.sqrt(var)

            position = None
            for date in y.index:
                z = zscore.get(date)
                if pd.isna(z):
                    continue

                if position is None:
                    if z > self.entry_zscore:
                        # a expensive → short a, long b
                        orders.append({"date": date, "type": "BUY", "ticker": ticker_b,
                                       "quantity": self.allocation_per_leg})
                        orders.append({"date": date, "type": "SELL", "ticker": ticker_a,
                                       "quantity": self.allocation_per_leg})
                        position = "long_b"
                    elif z < -self.entry_zscore:
                        orders.append({"date": date, "type": "BUY", "ticker": ticker_a,
                                       "quantity": self.allocation_per_leg})
                        orders.append({"date": date, "type": "SELL", "ticker": ticker_b,
                                       "quantity": self.allocation_per_leg})
                        position = "long_a"
                else:
                    should_exit = False
                    if position == "long_b":
                        if z <= self.exit_zscore or z > self.stop_loss_zscore:
                            should_exit = True
                    elif position == "long_a":
                        if z >= -self.exit_zscore or z < -self.stop_loss_zscore:
                            should_exit = True

                    if should_exit:
                        # Exit: fully close both legs. Float qty <= 1.0 means
                        # "fraction of current holdings", so 1.0 = close 100%.
                        if position == "long_b":
                            orders.append({"date": date, "type": "SELL", "ticker": ticker_b,
                                           "quantity": 1.0})
                            orders.append({"date": date, "type": "BUY", "ticker": ticker_a,
                                           "quantity": 1.0})
                        else:
                            orders.append({"date": date, "type": "SELL", "ticker": ticker_a,
                                           "quantity": 1.0})
                            orders.append({"date": date, "type": "BUY", "ticker": ticker_b,
                                           "quantity": 1.0})
                        position = None

        orders.sort(key=lambda o: (o["date"], o["ticker"], o["type"]))
        return orders
