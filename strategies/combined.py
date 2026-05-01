"""Combined strategy — run several OrderGenerators side-by-side and merge.

Engine quantity semantics (backtester/backtest_engine.py):
  - float in (0, 1.0]  → fraction of current portfolio value (BUY) or fraction
                         of current holdings (SELL on long; new short on flat)
  - everything else    → absolute share count

Combining strategies that all size BUYs as ``q * portfolio_value`` cannot
just be a concatenation: every child sees the same start-of-day PV, so
naively merging 4–5 fractional BUYs over-commits cash. ``CombinedOrderGenerator``
gives each child a weight (default 1/N) and scales orders so the total
target capital across children stays at ~100% of PV. The one exception is
``SELL 1.0`` — every child uses that as a "close my entire position"
sentinel, so it passes through unchanged; scaling it down would leave
residuals across rebalances. Fractional SELLs below 1.0 *are* scaled
because the engine reads them as new shorts when a child happens to be
flat at execution (e.g. PairsTrading's short leg), and an unscaled short
would blow past the child's allocated capital. Integer share counts
(CustomerSupplierMomentum's delta-style book) are scaled on both sides
because they're absolute, not position-relative.

Known cross-talk: the engine maintains one shared book per ticker, so if
two children both touch the same name (e.g. Momentum buys AAPL while
Mean Reversion later issues SELL AAPL 1.0), the second child's "close
all" closes both children's stakes. Run the bundle on a wide enough
universe that name-level overlap is rare, or curate the children so they
trade disjoint slices.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .order_generator import OrderGenerator


class CombinedOrderGenerator(OrderGenerator):
    """Merge orders from multiple sub-strategies under a per-child capital weight."""

    def __init__(
        self,
        generators: Sequence[OrderGenerator],
        weights: Optional[Sequence[float]] = None,
        names: Optional[Sequence[str]] = None,
    ):
        if not generators:
            raise ValueError("CombinedOrderGenerator needs at least one sub-strategy")
        self.generators: List[OrderGenerator] = list(generators)

        n = len(self.generators)
        if weights is None:
            weights_list = [1.0 / n] * n
        else:
            weights_list = [float(w) for w in weights]
            if len(weights_list) != n:
                raise ValueError(
                    f"weights length ({len(weights_list)}) must match generators length ({n})"
                )
            if any(w < 0 for w in weights_list):
                raise ValueError("weights must be non-negative")
            total = sum(weights_list)
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            weights_list = [w / total for w in weights_list]
        self.weights: List[float] = weights_list

        if names is None:
            self.names: List[str] = [g.__class__.__name__ for g in self.generators]
        else:
            names_list = list(names)
            if len(names_list) != n:
                raise ValueError("names length must match generators length")
            self.names = names_list

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        all_orders: List[Dict[str, Any]] = []
        for gen, weight, name in zip(self.generators, self.weights, self.names):
            sub_orders = gen.generate_orders(data)
            for order in sub_orders:
                scaled = self._scale_order(order, weight)
                if scaled is None:
                    continue
                scaled.setdefault("source", name)
                all_orders.append(scaled)

        all_orders.sort(
            key=lambda o: (
                o["date"],
                o.get("source", ""),
                o["ticker"],
                o["type"],
            )
        )
        return all_orders

    @staticmethod
    def _scale_order(order: Dict[str, Any], weight: float) -> Optional[Dict[str, Any]]:
        q = order["quantity"]
        # Treat numpy scalars as their Python counterparts.
        if hasattr(q, "item"):
            try:
                q = q.item()
            except Exception:
                pass

        is_fractional = isinstance(q, float) and 0.0 < q <= 1.0
        # bool is a subclass of int — exclude it so True/False can't sneak in as 1/0 shares.
        is_integer_shares = isinstance(q, int) and not isinstance(q, bool)
        # A float >1.0 is malformed for the engine (it doesn't match either branch
        # cleanly), but scale it as absolute shares to stay close to caller intent.
        if isinstance(q, float) and q > 1.0:
            is_integer_shares = True

        side = order["type"]
        new_q: Any = q

        if side == "BUY":
            if is_fractional:
                new_q = float(q) * weight
            elif is_integer_shares:
                scaled = int(round(float(q) * weight))
                if scaled <= 0:
                    return None
                new_q = scaled
        elif side == "SELL":
            if is_fractional:
                # SELL 1.0 is the universal "close my entire position" sentinel —
                # never scale it, otherwise the child can't fully exit. Anything
                # smaller is either a partial trim or a new short, and both
                # should be weighted to the child's slice of capital.
                if float(q) < 1.0:
                    new_q = float(q) * weight
            elif is_integer_shares:
                scaled = int(round(float(q) * weight))
                if scaled <= 0:
                    return None
                new_q = scaled

        out = dict(order)
        out["quantity"] = new_q
        return out


def build_default_combined(
    *,
    pairs: Optional[Sequence[tuple]] = None,
    capiq_path: Optional[str] = None,
    capiq_initial_cash: float = 100_000.0,
    weights: Optional[Sequence[float]] = None,
) -> CombinedOrderGenerator:
    """Equal-weight bundle of all five sub-strategies.

    CustomerSupplierMomentum is included only when ``capiq_path`` is provided
    — it requires a relationship file. Without it, the combined generator
    falls back to four sub-strategies and weights renormalize to 1/4.
    """
    from .customer_supplier_momentum import CustomerSupplierMomentumOrderGenerator
    from .mean_reversion import MeanReversionOrderGenerator
    from .momentum_strategy import MomentumOrderGenerator
    from .moving_avg import MovingAverageOrderGenerator
    from .pairs_trading import PairsTradingOrderGenerator

    if pairs is None:
        pairs = [("KO", "PEP"), ("V", "MA")]

    generators: List[OrderGenerator] = [
        MeanReversionOrderGenerator(window=100, position_size=0.5),
        MomentumOrderGenerator(window_days=125, threshold=0.02),
        MovingAverageOrderGenerator(),
        PairsTradingOrderGenerator(pairs=list(pairs), lookback_window=60),
    ]
    names = ["MeanReversion", "Momentum", "MovingAverage", "PairsTrading"]

    if capiq_path:
        n_other = len(generators)
        # Pre-scale CustomerSupplier's internal book so its delta-style sizing
        # already targets its slice of capital — post-hoc scaling of integer
        # SELL deltas keeps it consistent rebalance-to-rebalance.
        cs_weight = 1.0 / (n_other + 1)
        generators.append(
            CustomerSupplierMomentumOrderGenerator(
                capiq_path=capiq_path,
                initial_cash=capiq_initial_cash * cs_weight,
            )
        )
        names.append("CustomerSupplierMomentum")

    return CombinedOrderGenerator(generators=generators, weights=weights, names=names)
