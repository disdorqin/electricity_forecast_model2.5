"""
Daily Ledger GEF (Generalized Ensemble Fusion) weight learner.

Learns fusion weights from the past 30 days of prediction + actual
ledger data. Updates weights day by day from D-1 (most recent) to
D-30 (oldest), using day_gate to weight recency.

Algorithm: BGEW (Bounded Generalized Exponentiated Weighting)

For each (task, period):
  1. Start with equal weights for all models.
  2. For each day in [D-1, D-2, ..., D-30]:
     a. Compute per-model loss for that day + period.
     b. Normalize loss by median of available models.
     c. Update: w_m *= exp(-eta * day_gate * normalized_loss_m)
     d. Clip: w_m = max(w_m, weight_floor)
     e. Renormalize: sum(w) = 1
  3. Apply evidence shrinkage to prevent overfitting on sparse data.

No validation tap / rolling OOF / online validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Metrics
# ===========================================================================

def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    SMAPE-floor50 (correct formula per docs/metrics_calculation.md).

    Clips individual y_true and y_pred to floor=50 BEFORE computing SMAPE.
    This is the authoritative formula for 2.1.

    y_clip  = max(y_true, 50)
    pred_clip = max(y_pred, 50)
    SMAPE = mean(|pred_clip - y_clip| / ((|pred_clip| + |y_clip|) / 2)) * 100
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if mask.sum() == 0:
        return np.nan

    yt = yt[mask]
    yp = yp[mask]

    # Clip each value to floor=50 (per docs: clip per value, not per pair sum)
    yt_clip = np.maximum(yt, 50.0)
    yp_clip = np.maximum(yp, 50.0)

    denom = (np.abs(yp_clip) + np.abs(yt_clip)) / 2.0
    smape = np.mean(np.abs(yp_clip - yt_clip) / denom) * 100.0
    return float(smape)


def mae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE as percentage: 100 * MAE / max(median(|y_true_clip|), 50).

    Designed to be on the same 0-100 scale as SMAPE-floor50
    for composite loss blending.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if mask.sum() == 0:
        return np.nan

    yt = yt[mask]
    yp = yp[mask]

    mae = np.mean(np.abs(yt - yp))
    denominator = max(np.median(np.abs(np.maximum(yt, 50.0))), 50.0)
    return float(100.0 * mae / denominator)


def compute_daily_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_type: str = "composite",
) -> float:
    """
    Compute loss for a single model on a single day + period.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values for the hours in this period.
    loss_type : str
        "smape" or "composite" (0.7*smape_floor50 + 0.3*mae_percent).
        Both on 0-100 scale.

    Returns
    -------
    float loss value (lower is better).
    """
    if loss_type == "smape":
        return smape_floor50(y_true, y_pred)
    else:
        s = smape_floor50(y_true, y_pred)
        m = mae_percent(y_true, y_pred)
        if np.isnan(s):
            return m
        if np.isnan(m):
            return s
        # Both on 0-100 scale, so balanced blending
        return 0.7 * s + 0.3 * m


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class GEFConfig:
    """Configuration for the Daily Ledger GEF learner."""

    # Window
    window_days: int = 30

    # BGEW parameters
    eta: float = 0.8                # Learning rate
    weight_floor: float = 0.03      # Minimum weight per model
    day_gate_recent: float = 0.7    # Weight for D-1
    day_gate_oldest: float = 0.3    # Weight for D-30
    normalized_loss_min: float = 0.25
    normalized_loss_max: float = 4.0

    # Evidence shrinkage
    evidence_prior: float = 5.0     # Prior pseudo-count
    use_evidence_shrinkage: bool = True

    # Loss
    loss_type: str = "composite"    # "smape" or "composite"

    # Periods（默认 24 点三段；96 点由 resolution 覆盖）
    periods: tuple = ("1_8", "9_16", "17_24")
    # 每天每模型期望行数（默认 24；96 点 96）
    n_expected_per_day: int = 24
    # resolution：若非 None，__post_init__ 自动覆盖 periods / n_expected_per_day
    resolution: Optional[object] = None

    def __post_init__(self) -> None:
        if self.resolution is not None:
            self.periods = tuple(self.resolution.period_names)
            self.n_expected_per_day = self.resolution.slots_per_day


@dataclass
class WeightTraceRow:
    """Single row in the dynamic weight trace."""
    task: str
    period: str
    target_day: str
    age_days: int
    day_gate: float
    model_name: str
    weight_before: float
    loss: float
    normalized_loss: float
    weight_after: float


# ===========================================================================
# Main learner
# ===========================================================================

class DailyLedgerGEF:
    """
    Daily Ledger GEF weight learner.

    Learns per-(task, period) weights from the prediction + actual ledger
    using the BGEW algorithm with day-gated temporal decay.
    """

    def __init__(self, config: Optional[GEFConfig] = None):
        self.config = config or GEFConfig()
        self.weights_: dict = {}         # (task, period) → {model: weight}
        self.trace_: list[WeightTraceRow] = []

    def fit(
        self,
        training_table: pd.DataFrame,
    ) -> dict:
        """
        Learn weights from the training table.

        Parameters
        ----------
        training_table : pd.DataFrame
            Must contain: task, model_name, target_day, business_day,
            hour_business, period, y_pred, y_true, age_days, day_gate.

        Returns
        -------
        dict mapping (task, period) → {model_name: weight}
        """
        cfg = self.config
        self.trace_ = []

        # Get unique tasks and periods
        tasks = sorted(training_table["task"].unique())
        models = sorted(training_table["model_name"].unique())

        logger.info(
            f"DailyLedgerGEF.fit: {len(tasks)} tasks, {len(models)} models, "
            f"{len(training_table)} training rows"
        )

        weights = {}

        for task in tasks:
            task_df = training_table[training_table["task"] == task]

            for period in cfg.periods:
                period_df = task_df[task_df["period"] == period]
                key = (task, period)

                # Get sorted unique days for this period
                days = sorted(period_df["target_day"].unique())
                if len(days) == 0:
                    continue

                # Start with equal weights
                w = {m: 1.0 / len(models) for m in models}

                # Sort days by age_days ascending (D-1 first, D-30 last)
                day_info = (
                    period_df[["target_day", "age_days", "day_gate"]]
                    .drop_duplicates()
                    .sort_values("age_days")
                )

                for _, day_row in day_info.iterrows():
                    day = day_row["target_day"]
                    age = int(day_row["age_days"])
                    gate = float(day_row["day_gate"])

                    day_period_df = period_df[period_df["target_day"] == day]

                    if len(day_period_df) == 0:
                        continue

                    # Compute loss per model — each model's y_pred is aligned
                    # to its own y_true rows (sorted by hour_business, dropping NaN).
                    losses = {}
                    shape_errors = []
                    for m in models:
                        m_df = (
                            day_period_df[day_period_df["model_name"] == m]
                            # 96 点按 business_period 排序保证 y_true/y_pred 对齐；
                            # hourly 无该列则回退 hour_business
                            .sort_values("business_period" if "business_period" in day_period_df.columns else "hour_business")
                            .dropna(subset=["y_true", "y_pred"])
                        )

                        if len(m_df) == 0:
                            losses[m] = np.nan
                            continue

                        y_true_m = m_df["y_true"].values
                        y_pred_m = m_df["y_pred"].values

                        if len(y_true_m) != len(y_pred_m):
                            err = (
                                f"Loss shape mismatch for {task}/{period}/{day}/{m}: "
                                f"y_true={len(y_true_m)}, y_pred={len(y_pred_m)}"
                            )
                            shape_errors.append(err)
                            losses[m] = np.nan
                            continue

                        losses[m] = compute_daily_loss(y_true_m, y_pred_m, cfg.loss_type)

                    if shape_errors:
                        for err in shape_errors:
                            logger.error(err)
                        raise ValueError(
                            f"Shape mismatch in {task}/{period}/{day}: "
                            f"{len(shape_errors)} models affected. "
                            f"First error: {shape_errors[0]}"
                        )

                    # Available models
                    available = [m for m in models if not np.isnan(losses[m])]
                    if len(available) < 2:
                        # Not enough models to compare — skip this day
                        for m in models:
                            self.trace_.append(WeightTraceRow(
                                task=task, period=period, target_day=day,
                                age_days=age, day_gate=gate,
                                model_name=m,
                                weight_before=w.get(m, 0),
                                loss=losses.get(m, np.nan),
                                normalized_loss=np.nan,
                                weight_after=w.get(m, 0),
                            ))
                        continue

                    # Median loss of available models
                    available_losses = [losses[m] for m in available]
                    median_loss = float(np.median(available_losses))
                    if median_loss < 1e-6:
                        median_loss = 1e-6

                    # Update weights
                    for m in models:
                        w_before = w.get(m, cfg.weight_floor)

                        if m not in available:
                            # Missing model: keep weight, no update
                            self.trace_.append(WeightTraceRow(
                                task=task, period=period, target_day=day,
                                age_days=age, day_gate=gate,
                                model_name=m,
                                weight_before=w_before,
                                loss=np.nan,
                                normalized_loss=np.nan,
                                weight_after=w_before,
                            ))
                            continue

                        loss_m = losses[m]
                        norm_loss = loss_m / median_loss
                        norm_loss = float(np.clip(norm_loss, cfg.normalized_loss_min, cfg.normalized_loss_max))

                        # BGEW update
                        decay = np.exp(-cfg.eta * gate * norm_loss)
                        w_new = w_before * decay
                        w_new = max(w_new, cfg.weight_floor)

                        w[m] = w_new

                        self.trace_.append(WeightTraceRow(
                            task=task, period=period, target_day=day,
                            age_days=age, day_gate=gate,
                            model_name=m,
                            weight_before=w_before,
                            loss=loss_m,
                            normalized_loss=norm_loss,
                            weight_after=w_new,
                        ))

                    # Renormalize available model weights
                    total = sum(w[m] for m in models)
                    if total > 0:
                        for m in models:
                            w[m] = w[m] / total

                # --- Evidence shrinkage ---
                if cfg.use_evidence_shrinkage:
                    w = self._apply_evidence_shrinkage(
                        w, models, period_df, key
                    )

                weights[key] = dict(w)

        self.weights_ = weights

        logger.info(
            f"Learned weights for {len(weights)} (task, period) combinations"
        )

        return weights

    def _apply_evidence_shrinkage(
        self,
        w: dict,
        models: list[str],
        period_df: pd.DataFrame,
        key: tuple,
    ) -> dict:
        """
        Apply evidence shrinkage to prevent weights from overfitting
        when a model has too few observations.

        w_final = confidence * w_learned + (1 - confidence) * w_prior
        confidence = evidence_mass / (evidence_mass + evidence_prior)
        """
        cfg = self.config
        w_prior = {m: 1.0 / len(models) for m in models}

        for m in models:
            m_df = period_df[period_df["model_name"] == m]
            if len(m_df) == 0:
                w[m] = w_prior[m]
                continue

            evidence_mass = m_df["day_gate"].sum()
            confidence = evidence_mass / (evidence_mass + cfg.evidence_prior)
            w[m] = confidence * w.get(m, w_prior[m]) + (1.0 - confidence) * w_prior[m]

        # Renormalize
        total = sum(w.values())
        if total > 0:
            w = {m: v / total for m, v in w.items()}

        return w

    # =========================================================================
    # Output
    # =========================================================================

    def get_weights_df(self) -> pd.DataFrame:
        """Return weights as a DataFrame."""
        rows = []
        for (task, period), wdict in self.weights_.items():
            for model, weight in wdict.items():
                rows.append({
                    "task": task,
                    "period": period,
                    "model_name": model,
                    "weight": round(weight, 6),
                })
        return pd.DataFrame(rows)

    def get_trace_df(self) -> pd.DataFrame:
        """Return the dynamic weight trace as a DataFrame."""
        if not self.trace_:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self.trace_])

    def get_coverage_report(
        self,
        training_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate coverage report per model per day. expected=24 rows."""
        if training_table.empty:
            return pd.DataFrame()

        coverage = (
            training_table
            .groupby(["task", "target_day", "model_name"])
            .size()
            .reset_index(name="n_pred")
        )
        coverage["n_expected"] = cfg.n_expected_per_day
        coverage["coverage_pct"] = (
            coverage["n_pred"] / coverage["n_expected"] * 100
        ).round(1)

        # Status: ok if n_pred == n_expected_per_day, else incomplete
        coverage["status"] = coverage["n_pred"].apply(
            lambda x: "ok" if x == cfg.n_expected_per_day else "incomplete"
        )

        return coverage

    def get_candidate_metrics(self, training_table: pd.DataFrame) -> pd.DataFrame:
        """Compute per-model metrics from the training table."""
        rows = []
        for (task, model), grp in training_table.groupby(["task", "model_name"]):
            if len(grp) == 0:
                continue
            smape = smape_floor50(grp["y_true"].values, grp["y_pred"].values)
            mp = mae_percent(grp["y_true"].values, grp["y_pred"].values)
            mae_raw = float(np.mean(np.abs(grp["y_true"].values - grp["y_pred"].values)))

            learner_loss = np.nan
            if not np.isnan(smape) and not np.isnan(mp):
                learner_loss = round(0.7 * smape + 0.3 * mp, 4)
            elif not np.isnan(smape):
                learner_loss = round(smape, 4)
            elif not np.isnan(mp):
                learner_loss = round(mp, 4)

            rows.append({
                "task": task,
                "model_name": model,
                "n_samples": len(grp),
                "learner_loss": learner_loss,
                "smape_floor50": round(smape, 4) if not np.isnan(smape) else None,
                "mae": round(mae_raw, 4),
                "mae_percent": round(mp, 4) if not np.isnan(mp) else None,
            })
        return pd.DataFrame(rows)


# ===========================================================================
# NNLSGEF — 稀疏非负最小二乘融合权重学习器（2026-08-16 实证：优于 BGEW）
# ===========================================================================

@dataclass
class NNLSConfig:
    """NNLSGEF 配置。实证（96 点 ledger 2025-12~2026-07）：
    scipy.nnls 21 天 OOF → 段1 29.15 / 段3 21.43（等权 38.13/26.66），
    赢等权占比 68.6%，相对提升 +8.1%，多处超越单模型最优。
    SLSQP 版会退化等权（局部最优），scipy.nnls 天然稀疏无此问题。
    """

    window_days: int = 21                 # OOF 窗口（天）
    weight_floor: float = 0.02            # 单模型权重下界（防归零但允许强模型主导）
    use_ada_hedge_fallback: bool = True   # NNLS 失败/冷启动时回退 AdaHedge 在线更新
    ada_eta: float = 0.5
    loss_type: str = "composite"
    resolution: Optional[object] = None
    granularity: str = "period"           # "period"(3段) / "hour"(24组, 96点专属, 每小时块独立权重)

    periods: tuple = ("1_8", "9_16", "17_24")
    n_expected_per_day: int = 24

    def __post_init__(self) -> None:
        if self.resolution is not None:
            self.periods = tuple(self.resolution.period_names)
            self.n_expected_per_day = self.resolution.slots_per_day
            if self.granularity == "hour":
                # 24 小时块命名 h1..h24
                self.periods = tuple(f"h{h}" for h in range(1, 25))


class NNLSGEF:
    """Sparse non-negative least-squares fusion weight learner.

    对每个 (task, period)：
      1. 用最近 window_days 天 OOF 预测拼成设计矩阵 X（每列一个模型）、实际拼 y。
      2. scipy.optimize.nnls(X, y) 学非负系数，归一化为权重。
      3. 权重下界 weight_floor + 重归一（保留稀疏性，允许强模型主导）。
      4. 样本不足 / NNLS 退化时回退 AdaHedge 在线更新（或等权）。

    输出与 DailyLedgerGEF 完全兼容（weights.csv 同格式）。
    """

    def __init__(self, config: Optional[NNLSConfig] = None):
        self.config = config or NNLSConfig()
        self.weights_: dict = {}
        self.trace_: list[dict] = []

    def fit(self, training_table: pd.DataFrame) -> dict:
        cfg = self.config
        self.trace_ = []
        tasks = sorted(training_table["task"].unique())
        models = sorted(training_table["model_name"].unique())

        # 槽列（96 点用 business_period，24 点回退 hour_business）
        slot_col = "business_period" if "business_period" in training_table.columns else "hour_business"

        weights: dict = {}

        for task in tasks:
            task_df = training_table[training_table["task"] == task]
            # granularity: hour 粒度按 hour_business 分组（每小时 4 点一组，96 点专属）
            if cfg.granularity == "hour" and "hour_business" in training_table.columns:
                group_col = "hour_business"
                group_name = lambda g: f"h{int(g)}"
                groups = sorted(task_df["hour_business"].dropna().unique())
            else:
                group_col = "period"
                group_name = lambda g: str(g)
                groups = list(cfg.periods)

            for grp in groups:
                if group_col == "period":
                    period_df = task_df[task_df["period"] == grp]
                    period_label = str(grp)
                else:
                    period_df = task_df[task_df["hour_business"] == grp]
                    period_label = group_name(grp)
                key = (task, period_label)

                # 构造 X（行=样本点，列=模型）、y
                # 每个目标日每模型在该 period 有 n_expected 个点
                days = sorted(period_df["target_day"].unique())
                if not days:
                    continue
                # 只用最近 window_days 天（若有更多）
                recent_days = days[-cfg.window_days:]

                X_parts, y_parts = [], []
                for day in recent_days:
                    day_df = period_df[period_df["target_day"] == day]
                    # 宽表：index=slot, columns=model, 值=y_pred；y_true 任取一模型行
                    piv = day_df.pivot_table(index=slot_col, columns="model_name",
                                             values="y_pred", aggfunc="first")
                    if slot_col in day_df.columns and "y_true" in day_df.columns:
                        yt = (day_df.sort_values(slot_col)
                              .drop_duplicates(subset=[slot_col])["y_true"])
                    else:
                        continue
                    # 只保留全部模型都在的行
                    if piv.empty or not piv.columns.isin(models).all():
                        continue
                    piv = piv.reindex(columns=models)
                    X_day = piv.to_numpy(float)
                    if len(X_day) != len(yt) or np.isnan(X_day).any() or np.isnan(yt).any():
                        continue
                    X_parts.append(X_day)
                    y_parts.append(yt.to_numpy(float))

                if len(X_parts) < max(5, len(models)):
                    # 冷启动：AdaHedge 或等权
                    w = self._fallback_weights(models, period_df, recent_days)
                    self._trace(key, None, w, "cold_start")
                    weights[key] = dict(w)
                    continue

                X = np.vstack(X_parts)
                y = np.concatenate(y_parts)

                # 标准化列（数值稳定）
                Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

                try:
                    from scipy.optimize import nnls as scipy_nnls
                    sol, _ = scipy_nnls(Xs, y)
                except Exception:
                    sol = np.ones(len(models)) / len(models)
                s = float(sol.sum())
                if s < 1e-9:
                    w = self._fallback_weights(models, period_df, recent_days)
                    self._trace(key, X, w, "nnls_zero")
                    weights[key] = dict(w)
                    continue

                w_raw = sol / s
                # 权重下界
                w_vec = np.maximum(w_raw, cfg.weight_floor)
                w_vec = w_vec / w_vec.sum()
                w = dict(zip(models, w_vec))
                self._trace(key, X, w, "nnls")
                weights[key] = dict(w)

        self.weights_ = weights
        return weights

    def _fallback_weights(self, models, period_df, recent_days) -> dict:
        """AdaHedge 在线更新（样本不足时）。"""
        cfg = self.config
        w = {m: 1.0 / len(models) for m in models}
        cum_sq_best = 0.0
        for day in recent_days:
            day_df = period_df[period_df["target_day"] == day]
            losses = {}
            for m in models:
                m_df = day_df[day_df["model_name"] == m].dropna(subset=["y_true", "y_pred"])
                if len(m_df) == 0:
                    continue
                losses[m] = compute_daily_loss(
                    m_df["y_true"].values, m_df["y_pred"].values, cfg.loss_type
                )
            avail = [m for m in models if m in losses and np.isfinite(losses[m])]
            if len(avail) < 2:
                continue
            med = float(np.median([losses[m] for m in avail]))
            if med < 1e-6:
                med = 1e-6
            best_loss = min(losses[m] for m in avail)
            cum_sq_best += best_loss ** 2
            eta = np.sqrt(2.0 * np.log(len(models)) / (cum_sq_best + 1e-8))
            for m in models:
                if m in losses and np.isfinite(losses[m]):
                    w[m] *= np.exp(-eta * (losses[m] / med))
                w[m] = max(w[m], cfg.weight_floor)
            tot = sum(w.values())
            if tot > 0:
                w = {m: v / tot for m, v in w.items()}
        return w

    def _trace(self, key, X, w, method):
        self.trace_.append({
            "task": key[0], "period": key[1], "method": method,
            "n_obs": 0 if X is None else X.shape[0],
            **{f"w_{m}": round(v, 6) for m, v in w.items()},
        })

    def get_weights_df(self) -> pd.DataFrame:
        rows = []
        for (task, period), wdict in self.weights_.items():
            for model, weight in wdict.items():
                rows.append({
                    "task": task, "period": period,
                    "model_name": model, "weight": round(weight, 6),
                })
        return pd.DataFrame(rows)

    def get_trace_df(self) -> pd.DataFrame:
        if not self.trace_:
            return pd.DataFrame()
        return pd.DataFrame(self.trace_)

    def get_candidate_metrics(self, training_table: pd.DataFrame) -> pd.DataFrame:
        """兼容接口：逐模型指标。"""
        rows = []
        for (task, model), grp in training_table.groupby(["task", "model_name"]):
            if len(grp) == 0:
                continue
            smape = smape_floor50(grp["y_true"].values, grp["y_pred"].values)
            mp = mae_percent(grp["y_true"].values, grp["y_pred"].values)
            rows.append({
                "task": task, "model_name": model, "n_samples": len(grp),
                "learner_loss": round(0.7 * smape + 0.3 * mp, 4) if not np.isnan(smape) and not np.isnan(mp) else None,
                "smape_floor50": round(smape, 4) if not np.isnan(smape) else None,
                "mae_percent": round(mp, 4) if not np.isnan(mp) else None,
            })
        return pd.DataFrame(rows)
