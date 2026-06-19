from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import make_cfg
from .deep_no_leak_experiment import SequenceWindowDataset, make_eval_loader
from .extrapolation_robustness import temperature_balanced_loader
from .models import DecomposedWindowDataset, collate_meta_to_frame
from .nmc_branchbands_experiment import build_feature_frames, estimate_r0_by_temperature, find_csv_files, write_start_audit
from .nmc_vcorr_it_condinv_staged_exact import (
    CondInvConfig,
    CondInvVariant,
    SnapshotEnsembleBase,
    Stage2CorrectedModel,
    _meta_list,
    conditional_profile_mmd,
    eval_by_temp,
    make_model,
    set_seed,
    to_variant,
    train_stage2_correction,
)
from .nmc_vcorr_it_goal_remote_screen import VcorrITGoalModel, group_keys, temp_weights
from .nmc_vcorr_it_lstm_singlehead_bytemp import FEATURE_COLS
from .nmc_vcorr_it_train_selector_audit import eval_by_temp_drive
from .nmc_vit_feature_lstm_experiment import add_vit_engineered_features, write_input_schema, write_leakage_audit
from .nmc_vit_feature_lstm_experiment import feature_columns as vit_feature_columns
from .runtime import configure_torch_runtime, device
from .training import make_scaled_frames_for_ablation


BASE_PREFIX = "nmc_goal_vcorr_it_train_dst25_selector"


@dataclass
class TrainDSTSelectorConfig:
    base_dir: Path = Path(".")
    raw_root: Path = Path("nmc_soc_ocvstart_relabelled_from_lc_ocv/data/NMC SAMSUNG INR 18650 2Ah")
    output_prefix: str = BASE_PREFIX
    seeds: tuple[int, ...] = (0, 1, 2)
    train_profiles: tuple[str, ...] = ("DST", "US06")
    valid_profiles: tuple[str, ...] = ("BJDST",)
    test_profiles: tuple[str, ...] = ("FUDS",)
    train_temperatures: tuple[float, ...] = ()
    valid_temperatures: tuple[float, ...] = ()
    test_temperatures: tuple[float, ...] = ()
    finetune_temperatures: tuple[float, ...] = ()
    finetune_epochs: int = 0
    finetune_lr_scale: float = 0.35
    stage2_train_temperatures: tuple[float, ...] = ()
    stage2_valid_temperatures: tuple[float, ...] = ()
    stage2_test_temperatures: tuple[float, ...] = ()
    window_len: int = 50
    stride: int = 3
    epochs: int = 10
    selector_min_epoch: int = 1
    selector_max_epoch: int = 0
    stage2_epochs: int = 60
    eval_every: int = 5
    stage1_eval_every: int = 1
    batch_size: int = 1024
    lr: float = 8e-4
    lr_stage2: float = 8e-4
    weight_decay: float = 2e-4
    hidden_size: int = 64
    layers: int = 5
    kernel_size: int = 5
    recurrent: str = "tcn"
    head_kind: str = "linear"
    temp_mode: str = "moe"
    dropout: float = 0.06
    loss_kind: str = "huber"
    huber_beta: float = 0.02
    rex_group: str = "temperature_drive"
    lambda_rex: float = 2.0
    lambda_condinv: float = 0.02
    weight_0: float = 4.0
    weight_25: float = 2.2
    weight_45: float = 1.0
    stage1_selector: str = "train25_dst"
    selector_regime_min_windows: int = 30
    fixed_stage1_epoch: int = 0
    model_kind: str = "single"
    fusion_h64_weight: float = 0.5
    focus45_weight: float = 12.0
    keep_lambda: float = 4.0
    corr_mode: str = "cold_hot"
    corr_limit: float = 1.2
    corr_zero_init: bool = True
    stage2_select_rule: str = "none"
    stage2_stage1_threshold: int = 7
    stage2_early_epoch: int = 30
    stage2_late_epoch: int = 35
    test_blind: bool = False
    train_sampler: str = "temperature_balanced"
    stage1_ensemble_epochs: str = ""
    feature_set: str = "vcorr_it"
    stage2_feature_set: str = ""
    sampler_seed_mode: str = "seed"
    test_blind_rng_burn: int = 0
    diagnostic_test_every: int = 0
    skip_stage2: bool = False
    lambda_current_consistency: float = 0.0
    current_noise_std: float = 0.0
    current_dropout_prob: float = 0.0
    lambda_profile_adv: float = 0.0
    profile_adv_grl: float = 1.0
    lambda_profile_supcon: float = 0.0
    supcon_temperature: float = 0.15
    anchor_residual_limit: float = 0.12
    lambda_anchor_loss: float = 0.2
    ssl_pretrain_epochs: int = 0
    ssl_lr: float = 8e-4
    ssl_recon_weight: float = 1.0
    ssl_next_vcorr_weight: float = 0.5
    ssl_slope_weight: float = 0.25
    valid_split_mode: str = "profile"
    valid_block_rows: int = 800
    valid_block_mod: int = 5
    valid_block_index: int = 4
    save_predictions: bool = False
    ema_perturbation_importance: bool = False
    sequence_training: bool = False
    num_workers: int = 4
    prefetch_factor: int = 4
    v_corr_tau_s: float = 120.0
    v_pol_mid_tau_s: float = 60.0
    v_pol_slow_tau_s: float = 600.0
    v_hys_tau_s: float = 1200.0


class H64H128FusionModel(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.06, fixed_h64_weight: float | None = None):
        super().__init__()
        self.fixed_h64_weight = fixed_h64_weight
        self.branch64 = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=64,
            recurrent="tcn",
            layers=5,
            head_kind="linear",
            temp_mode="moe",
            dropout=float(dropout),
            kernel_size=5,
            norm_kind="channel",
        )
        self.branch128 = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=128,
            recurrent="tcn",
            layers=6,
            head_kind="linear",
            temp_mode="moe",
            dropout=float(dropout),
            kernel_size=5,
            norm_kind="channel",
        )
        self.proj64 = nn.Linear(64, 128)
        gate_dim = input_dim * 4 + 3
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _window_stats(self, x: torch.Tensor) -> torch.Tensor:
        x_end = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_delta = x_end - x[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def _branch_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h64 = self.branch64.encode_sequence(x)
        h128 = self.branch128.encode_sequence(x)
        log64 = self._logits_from_hidden(self.branch64, h64, x)
        log128 = self._logits_from_hidden(self.branch128, h128, x)
        return h64, h128, log64, log128

    @staticmethod
    def _logits_from_hidden(model: VcorrITGoalModel, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        temp = x[..., model.temp_idx:model.temp_idx + 1]
        if model.temp_mode == "none":
            return model.base_head(h)
        if model.temp_mode == "bias":
            return model.base_head(h) + model.temp_bias(temp)
        if model.temp_mode == "moe":
            logits = torch.stack([head(h) for head in model.expert_heads], dim=-1).squeeze(-2)
            gate = torch.softmax(model.temp_gate(temp), dim=-1)
            return torch.sum(logits * gate, dim=-1, keepdim=True)
        if model.temp_mode == "hard_heads":
            logits = torch.stack([head(h) for head in model.expert_heads], dim=-1).squeeze(-2)
            idx = model._hard_head_indices(temp)
            return torch.gather(logits, dim=-1, index=idx.unsqueeze(-1))
        raise ValueError(f"Unknown temp_mode={model.temp_mode}")

    def _weights(self, x: torch.Tensor, log64: torch.Tensor, log128: torch.Tensor) -> torch.Tensor:
        if self.fixed_h64_weight is not None:
            w64 = torch.full((x.shape[0], 1), float(self.fixed_h64_weight), device=x.device, dtype=x.dtype)
            return torch.cat([w64, 1.0 - w64], dim=1)
        end64 = log64[:, -1, :]
        end128 = log128[:, -1, :]
        gate_input = torch.cat([self._window_stats(x), end64, end128, end64 - end128], dim=1)
        return torch.softmax(self.gate(gate_input), dim=1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h64, h128, log64, log128 = self._branch_outputs(x)
        weights = self._weights(x, log64, log128)
        h64_proj = self.proj64(h64)
        return weights[:, None, 0:1] * h64_proj + weights[:, None, 1:2] * h128

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        _h64, _h128, log64, log128 = self._branch_outputs(x)
        weights = self._weights(x, log64, log128)
        return weights[:, None, 0:1] * log64 + weights[:, None, 1:2] * log128

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class AnchorResidualTCNModel(nn.Module):
    """Voltage-anchor SOC estimator with a bounded dynamic residual.

    The anchor branch sees only causal voltage/temperature-derived channels at
    each time step. The TCN may use the full selected feature set, but it can
    only add a bounded SOC residual around the anchor prediction.
    """

    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        residual_limit: float,
        recurrent: str = "tcn",
        temp_mode: str = "moe",
    ):
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.dynamic = VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(hidden_size),
            recurrent=str(recurrent),
            layers=int(layers),
            head_kind="linear",
            temp_mode=str(temp_mode),
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            norm_kind="channel",
        )
        preferred = [
            "V_corr_raw",
            "T",
            "V_corr_raw_ema50",
            "V_corr_raw_dev_ema50",
            "V_corr_raw_ema200",
            "V_corr_raw_dev_ema200",
            "V_corr_raw_ema800",
            "V_corr_raw_dev_ema800",
            "dV_corr",
            "abs_dV_corr",
            "Vcorr_x_absI",
        ]
        idx = [feature_cols.index(col) for col in preferred if col in feature_cols]
        if "V_corr_raw" not in feature_cols or "T" not in feature_cols:
            raise RuntimeError("AnchorResidualTCNModel requires V_corr_raw and T features.")
        self.anchor_indices = idx
        anchor_dim = len(idx)
        self.anchor_head = nn.Sequential(
            nn.Linear(anchor_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic.encode_sequence(x)

    def anchor_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor_x = x.index_select(dim=2, index=torch.as_tensor(self.anchor_indices, device=x.device))
        return torch.sigmoid(self.anchor_head(anchor_x))

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor_sequence(x)
        h = self.encode_sequence(x)
        residual = float(self.residual_limit) * torch.tanh(self.residual_head(h))
        return (anchor + residual).clamp(0.0, 1.0)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        soc = self.forward_sequence(x).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(soc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


class MLPControlModel(nn.Module):
    """Non-recurrent control model for fixed G4 model-class comparisons."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float, mode: str):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in {"endpoint", "window_summary"}:
            raise ValueError(f"Unknown MLPControlModel mode={mode!r}")
        in_dim = int(input_dim) if self.mode == "endpoint" else int(input_dim) * 4
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.SiLU(),
        )
        self.head = nn.Linear(int(hidden_size), 1)

    def _summarize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "endpoint":
            return x[:, -1, :]
        x_end = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1, unbiased=False)
        x_delta = x_end - x[:, 0, :]
        return torch.cat([x_end, x_mean, x_std, x_delta], dim=1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self._summarize(x))
        return h[:, None, :].expand(-1, x.shape[1], -1)

    def logits_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x)
        logits = self.head(h[:, -1, :])
        return logits[:, None, :].expand(-1, x.shape[1], -1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_sequence(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1, :]


def _selected_feature_columns(feature_set: str) -> list[str]:
    cols = vit_feature_columns(str(feature_set))
    prefix = ["V_corr_raw", "I_raw", "T"]
    missing_prefix = [c for c in prefix if c not in cols]
    if missing_prefix:
        raise RuntimeError(f"feature_set={feature_set!r} is missing required leading columns: {missing_prefix}")
    return prefix + [c for c in cols if c not in set(prefix)]


def _make_stage1_model(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    input_dim: int | None = None,
    feature_cols: list[str] | None = None,
) -> nn.Module:
    input_dim = int(input_dim if input_dim is not None else len(FEATURE_COLS))
    if str(cfg.model_kind) == "fusion_h64_h128":
        return H64H128FusionModel(input_dim, dropout=float(cfg.dropout)).to(device)
    if str(cfg.model_kind) == "fusion_h64_h128_fixed":
        return H64H128FusionModel(
            input_dim,
            dropout=float(cfg.dropout),
            fixed_h64_weight=float(cfg.fusion_h64_weight),
        ).to(device)
    if str(cfg.model_kind) == "anchor_residual_tcn":
        return AnchorResidualTCNModel(
            input_dim=input_dim,
            feature_cols=list(feature_cols or _selected_feature_columns(str(cfg.feature_set))),
            hidden_size=int(cfg.hidden_size),
            layers=int(cfg.layers),
            kernel_size=int(cfg.kernel_size),
            dropout=float(cfg.dropout),
            residual_limit=float(cfg.anchor_residual_limit),
            recurrent=str(cfg.recurrent),
            temp_mode=str(cfg.temp_mode),
        ).to(device)
    if str(cfg.model_kind) == "endpoint_mlp":
        return MLPControlModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            dropout=float(cfg.dropout),
            mode="endpoint",
        ).to(device)
    if str(cfg.model_kind) == "window_summary_mlp":
        return MLPControlModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            dropout=float(cfg.dropout),
            mode="window_summary",
        ).to(device)
    if str(cfg.model_kind) == "single":
        model = make_model(variant, CondInvConfig(hidden_size=cfg.hidden_size, window_len=cfg.window_len))
        if input_dim == len(FEATURE_COLS):
            return model
        return VcorrITGoalModel(
            input_dim=input_dim,
            hidden_size=int(cfg.hidden_size),
            recurrent=variant.recurrent,
            layers=int(variant.layers),
            head_kind=variant.head_kind,
            temp_mode=variant.temp_mode,
            dropout=float(variant.dropout),
            kernel_size=int(variant.kernel_size),
            norm_kind="channel",
        ).to(device)
    raise ValueError(f"Unknown model_kind={cfg.model_kind!r}")


def _metric_value(metrics: pd.DataFrame, split: str, temp: float, drive: str) -> float:
    if metrics.empty or "split" not in metrics.columns:
        return float("inf")
    sub = metrics[
        metrics["split"].eq(split)
        & np.isclose(metrics["temperature_C"], float(temp))
        & metrics["drive_cycle"].astype(str).eq(str(drive))
    ]
    if not len(sub):
        return float("inf")
    return float(sub["MAE_pct"].iloc[0])


def _temp_metric_value(metrics: pd.DataFrame, split: str, temp: float) -> float:
    if metrics.empty or "split" not in metrics.columns:
        return float("inf")
    sub = metrics[metrics["split"].eq(split) & np.isclose(metrics["temperature_C"], float(temp))]
    if not len(sub):
        return float("inf")
    return float(sub["MAE_pct"].iloc[0])


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("inf")


def _finite_max(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.max(finite)) if finite else float("inf")


def _target_for_temp(temp: float) -> float:
    if np.isclose(float(temp), 0.0):
        return 1.0
    if np.isclose(float(temp), 25.0):
        return 0.7
    if np.isclose(float(temp), 45.0):
        return 0.3
    return 1.0


def _regime_selector_values(valid_regime_metrics: pd.DataFrame | None, min_windows: int) -> tuple[float, float, int, int]:
    if valid_regime_metrics is None or valid_regime_metrics.empty:
        return float("inf"), float("inf"), 0, 0
    df = valid_regime_metrics.copy()
    if "temperature_C" not in df.columns or "MAE_pct" not in df.columns:
        return float("inf"), float("inf"), 0, 0
    df["target_norm"] = [
        float(mae) / _target_for_temp(float(temp))
        for mae, temp in zip(df["MAE_pct"], df["temperature_C"])
    ]
    total = int(len(df))
    if "n_windows" in df.columns:
        df = df[pd.to_numeric(df["n_windows"], errors="coerce").fillna(0).ge(int(min_windows))]
    finite = df[np.isfinite(df["target_norm"])]
    if finite.empty:
        return float("inf"), float("inf"), 0, total
    return float(finite["target_norm"].mean()), float(finite["target_norm"].max()), int(len(finite)), total


def _selector_score(
    train_metrics: pd.DataFrame,
    rule: str,
    valid_metrics: pd.DataFrame | None = None,
    valid_regime_metrics: pd.DataFrame | None = None,
    regime_min_windows: int = 30,
) -> dict[str, float]:
    dst25 = _metric_value(train_metrics, "train", 25.0, "DST")
    us0625 = _metric_value(train_metrics, "train", 25.0, "US06")
    vals = [dst25, us0625]
    mean25 = _finite_mean(vals)
    worst25 = _finite_max(vals)
    gap25 = float(abs(dst25 - us0625)) if np.isfinite(dst25) and np.isfinite(us0625) else float("inf")
    valid_metrics = valid_metrics if valid_metrics is not None else pd.DataFrame()
    valid0 = _temp_metric_value(valid_metrics, "valid", 0.0)
    valid25 = _temp_metric_value(valid_metrics, "valid", 25.0)
    valid45 = _temp_metric_value(valid_metrics, "valid", 45.0)
    valid_vals = [valid0, valid25, valid45]
    valid_mean = _finite_mean(valid_vals)
    valid_worst = _finite_max(valid_vals)
    valid_finite = [v for v in valid_vals if np.isfinite(v)]
    valid_spread = float(valid_worst - min(valid_finite)) if valid_finite else float("inf")
    valid_hot_gap = float(max(0.0, valid45 - valid25)) if np.isfinite(valid45) and np.isfinite(valid25) else float("inf")
    valid_target_norm = [
        valid0 / 1.0 if np.isfinite(valid0) else float("inf"),
        valid25 / 0.7 if np.isfinite(valid25) else float("inf"),
        valid45 / 0.3 if np.isfinite(valid45) else float("inf"),
    ]
    valid_target_norm_mean = _finite_mean(valid_target_norm)
    valid_target_norm_worst = _finite_max(valid_target_norm)
    (
        valid_regime_target_norm_mean,
        valid_regime_target_norm_worst,
        valid_regime_slices_used,
        valid_regime_slices_total,
    ) = _regime_selector_values(valid_regime_metrics, int(regime_min_windows))
    if rule == "train25_dst":
        score = dst25
    elif rule == "train25_us06":
        score = us0625
    elif rule == "train25_mean_drive":
        score = mean25
    elif rule == "train25_worst_drive":
        score = worst25
    elif rule == "train25_worst_gap":
        score = worst25 + 0.5 * gap25
    elif rule == "valid_worst_temp":
        score = valid_worst
    elif rule == "val_mean_mae":
        score = valid_mean
    elif rule == "val_worst_mae":
        score = valid_worst
    elif rule == "val_mean_plus_worst":
        score = valid_mean + 0.5 * valid_worst
    elif rule == "val_target_worst":
        score = valid_target_norm_worst
    elif rule == "val_target_mean_plus_worst":
        score = valid_target_norm_mean + 0.5 * valid_target_norm_worst
    elif rule == "val_regime_target_worst":
        score = valid_regime_target_norm_worst
    elif rule == "val_regime_target_mean_plus_worst":
        score = valid_regime_target_norm_mean + 0.5 * valid_regime_target_norm_worst
    elif rule == "valid25_temp":
        score = valid25
    elif rule == "last_epoch":
        score = -float(valid_metrics["epoch"].iloc[0]) if len(valid_metrics) and "epoch" in valid_metrics else 0.0
    elif rule == "train25_dst_valid_worst":
        score = dst25 + 0.7 * valid_worst
    elif rule == "train25_worst_valid_worst":
        score = worst25 + 0.7 * valid_worst
    elif rule == "train25_dst_valid45_guard":
        score = dst25 + 0.7 * valid45 + 0.3 * valid_hot_gap
    elif rule == "train25_worst_valid_balance":
        score = worst25 + 0.5 * gap25 + 0.6 * valid_worst + 0.2 * valid_spread
    else:
        raise ValueError(f"Unknown stage1_selector={rule!r}")
    return {
        "selector_score": float(score),
        "selector_train25_dst": float(dst25),
        "selector_train25_us06": float(us0625),
        "selector_train25_mean_drive": float(mean25),
        "selector_train25_worst_drive": float(worst25),
        "selector_train25_gap": float(gap25),
        "selector_valid0": float(valid0),
        "selector_valid25": float(valid25),
        "selector_valid45": float(valid45),
        "selector_valid_mean": float(valid_mean),
        "selector_valid_worst": float(valid_worst),
        "selector_valid_spread": float(valid_spread),
        "selector_valid_hot_gap": float(valid_hot_gap),
        "selector_valid_target_norm_mean": float(valid_target_norm_mean),
        "selector_valid_target_norm_worst": float(valid_target_norm_worst),
        "selector_valid_regime_target_norm_mean": float(valid_regime_target_norm_mean),
        "selector_valid_regime_target_norm_worst": float(valid_regime_target_norm_worst),
        "selector_valid_regime_min_windows": int(regime_min_windows),
        "selector_valid_regime_slices_used": int(valid_regime_slices_used),
        "selector_valid_regime_slices_total": int(valid_regime_slices_total),
    }


def _standard_train_loader(ds, cfg: TrainDSTSelectorConfig, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": True,
        "generator": generator,
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _soc_bin4(soc: float) -> str:
    value = float(soc)
    if value < 0.2:
        return "soc0_0_20"
    if value < 0.5:
        return "soc1_20_50"
    if value < 0.8:
        return "soc2_50_80"
    return "soc3_80_100"


def _endpoint_balance_keys(ds, mode: str) -> list[tuple]:
    keys = []
    for fi, _start, end in ds.index:
        frame = ds.frames[fi]
        temp = round(float(frame["temperature"][end]), 3)
        drive = str(frame["drive_cycle"][end]).upper()
        if str(mode) == "temperature_profile_balanced":
            keys.append((temp, drive))
        elif str(mode) == "temperature_profile_soc_balanced":
            keys.append((temp, drive, _soc_bin4(float(frame["y_physical"][end]))))
        else:
            raise ValueError(f"Unknown profile balance mode={mode!r}")
    return keys


def _metric_bin(value: float, edges: tuple[float, float]) -> str:
    if not np.isfinite(value):
        return "nan"
    lo, hi = edges
    if value <= lo:
        return "low"
    if value <= hi:
        return "mid"
    return "high"


def _regime_edges(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    lo, hi = np.nanquantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 0.0)
    if hi <= lo:
        hi = lo + 1e-6
    return (float(lo), float(hi))


def _window_regime_rows(ds) -> pd.DataFrame:
    rows = []
    feature_to_idx = {name: idx for idx, name in enumerate(getattr(ds, "feature_cols", []))}

    def get_feature(frame, col: str, sl: slice) -> np.ndarray | None:
        idx = feature_to_idx.get(col)
        if idx is None:
            return None
        return np.asarray(frame["x"][sl, idx], dtype=np.float64)

    for row_id, (fi, start, end) in enumerate(ds.index):
        frame = ds.frames[fi]
        sl = slice(int(start), int(end) + 1)
        temp = round(float(frame["temperature"][end]), 3)
        drive = str(frame["drive_cycle"][end]).upper()
        i = get_feature(frame, "I_raw", sl)
        if i is None:
            raise RuntimeError("Regime sampler requires I_raw in the selected feature set.")
        abs_i = get_feature(frame, "absI", sl)
        if abs_i is None:
            abs_i = np.abs(i)
        di = get_feature(frame, "dI", sl)
        if di is None:
            di = np.diff(i, prepend=i[0] if len(i) else 0.0)
        v = get_feature(frame, "V_corr_raw", sl)
        if v is None:
            raise RuntimeError("Regime sampler requires V_corr_raw in the selected feature set.")
        rows.append(
            {
                "row_id": int(row_id),
                "temperature_C": float(temp),
                "drive_cycle": drive,
                "absI_mean": float(np.nanmean(np.abs(abs_i))),
                "I_std": float(np.nanstd(i)),
                "dI_energy": float(np.nanmean(np.square(di))),
                "V_corr_span": float(np.nanmax(v) - np.nanmin(v)) if len(v) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _regime_balance_keys(ds, mode: str) -> tuple[list[tuple], pd.DataFrame, pd.DataFrame]:
    rows = _window_regime_rows(ds)
    metric_cols = ["absI_mean", "I_std", "dI_energy", "V_corr_span"]
    edges = {col: _regime_edges(rows[col].tolist()) for col in metric_cols}
    for col in metric_cols:
        rows[f"{col}_bin"] = [_metric_bin(v, edges[col]) for v in rows[col]]
    key_cols = ["temperature_C", "drive_cycle", "absI_mean_bin", "I_std_bin", "dI_energy_bin", "V_corr_span_bin"]
    if mode == "temperature_regime_balanced":
        key_cols = ["temperature_C", "absI_mean_bin", "I_std_bin", "dI_energy_bin", "V_corr_span_bin"]
    elif mode != "temperature_profile_regime_balanced":
        raise ValueError(f"Unknown regime balance mode={mode!r}")
    keys = [tuple(row[col] for col in key_cols) for _, row in rows.iterrows()]
    edge_rows = [
        {"metric": col, "low_mid_edge": edges[col][0], "mid_high_edge": edges[col][1]}
        for col in metric_cols
    ]
    counts = rows.assign(regime_key=[repr(k) for k in keys]).groupby("regime_key", as_index=False).size()
    counts = counts.rename(columns={"size": "n_windows"}).sort_values(["n_windows", "regime_key"]).reset_index(drop=True)
    return keys, pd.DataFrame(edge_rows), counts


def _write_regime_sampler_audit(ds, cfg: TrainDSTSelectorConfig, out_dir: Path, seed: int) -> None:
    if str(cfg.train_sampler) not in {"temperature_regime_balanced", "temperature_profile_regime_balanced"}:
        return
    _keys, edges, counts = _regime_balance_keys(ds, str(cfg.train_sampler))
    prefix = f"{cfg.output_prefix}_seed{seed}_{cfg.train_sampler}"
    edges.to_csv(out_dir / f"{prefix}_regime_edges.csv", index=False)
    counts.to_csv(out_dir / f"{prefix}_regime_counts.csv", index=False)


def _profile_balanced_train_loader(ds, cfg: TrainDSTSelectorConfig, seed: int, mode: str) -> DataLoader:
    if str(mode) in {"temperature_regime_balanced", "temperature_profile_regime_balanced"}:
        keys, _edges, _counts = _regime_balance_keys(ds, str(mode))
    else:
        keys = _endpoint_balance_keys(ds, mode)
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = torch.as_tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "sampler": sampler,
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(ds, **kwargs)


def _make_train_loader(ds, base_cfg, cfg: TrainDSTSelectorConfig, seed: int):
    if str(cfg.train_sampler) == "standard":
        return _standard_train_loader(ds, cfg, int(seed))
    if str(cfg.train_sampler) == "temperature_balanced":
        return temperature_balanced_loader(ds, base_cfg, shuffle=True)
    if str(cfg.train_sampler) in {
        "temperature_profile_balanced",
        "temperature_profile_soc_balanced",
        "temperature_regime_balanced",
        "temperature_profile_regime_balanced",
    }:
        return _profile_balanced_train_loader(ds, cfg, int(seed), str(cfg.train_sampler))
    raise ValueError(f"Unknown train_sampler={cfg.train_sampler!r}")


def _meta_float_tensor(meta, key: str) -> torch.Tensor:
    vals = meta[key]
    if torch.is_tensor(vals):
        return vals.to(device=device, dtype=torch.float32)
    return torch.as_tensor(vals, device=device, dtype=torch.float32)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def _grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(x, float(scale))


def _profile_label_tensor(meta, profile_to_label: dict[str, int]) -> torch.Tensor:
    vals = meta["drive_cycle"]
    if torch.is_tensor(vals):
        raise TypeError("drive_cycle metadata is expected to be string-like, not a tensor.")
    labels = []
    for value in vals:
        key = str(value).upper()
        if key not in profile_to_label:
            raise RuntimeError(f"Unexpected drive_cycle={value!r}; known profiles={sorted(profile_to_label)}")
        labels.append(profile_to_label[key])
    return torch.as_tensor(labels, device=device, dtype=torch.long)


def _profile_supcon_loss(
    h: torch.Tensor,
    y_endpoint: torch.Tensor,
    meta,
    *,
    temperature: float,
) -> torch.Tensor:
    """Align same-temperature/SOC-bin samples across drive profiles.

    This keeps SOC information as the condition: positives must be in the
    same temperature and endpoint-SOC bin, but from a different drive profile.
    It uses training labels only and never sees validation/test SOC labels.
    """
    n = int(h.shape[0])
    if n < 2:
        return h.new_tensor(0.0)
    features = F.normalize(h, dim=1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    eye = torch.eye(n, device=h.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    temps_raw = meta["temperature"]
    if torch.is_tensor(temps_raw):
        temps = temps_raw.to(device=h.device, dtype=torch.float32)
    else:
        temps = torch.as_tensor(temps_raw, device=h.device, dtype=torch.float32)
    temps = torch.round(temps * 10.0) / 10.0
    y_flat = y_endpoint[:, 0].detach() if y_endpoint.ndim > 1 else y_endpoint.detach()
    soc_bins = torch.bucketize(y_flat, torch.as_tensor([0.2, 0.5, 0.8], device=h.device))
    drives = _meta_list(meta, "drive_cycle")
    drive_ids = {str(v).upper(): i for i, v in enumerate(sorted({str(x).upper() for x in drives}))}
    drive = torch.as_tensor([drive_ids[str(v).upper()] for v in drives], device=h.device, dtype=torch.long)

    same_context = temps[:, None].eq(temps[None, :]) & soc_bins[:, None].eq(soc_bins[None, :])
    cross_profile = drive[:, None].ne(drive[None, :])
    positives = same_context & cross_profile & (~eye)
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return h.new_tensor(0.0)
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    pos_sum = (exp_logits * positives.to(exp_logits.dtype)).sum(dim=1)
    denom = exp_logits.sum(dim=1).clamp_min(1e-12)
    loss = -torch.log((pos_sum / denom).clamp_min(1e-12))
    return loss[valid].mean()


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, cfg: TrainDSTSelectorConfig, reduction: str = "mean") -> torch.Tensor:
    loss_kind = str(cfg.loss_kind).lower()
    if loss_kind == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    if loss_kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=float(cfg.huber_beta), reduction=reduction)
    raise ValueError(f"Unknown loss_kind={cfg.loss_kind!r}")


def _burn_global_rng(count: int) -> None:
    for _ in range(max(0, int(count))):
        torch.empty((), dtype=torch.int64).random_()


def _augment_current_channel(
    x: torch.Tensor,
    *,
    noise_std: float,
    dropout_prob: float,
) -> torch.Tensor:
    if float(noise_std) <= 0.0 and float(dropout_prob) <= 0.0:
        return x
    out = x.clone()
    current = out[..., 1:2]
    if float(noise_std) > 0.0:
        scale = 1.0 + float(noise_std) * torch.randn(
            (x.shape[0], 1, 1),
            device=x.device,
            dtype=x.dtype,
        )
        current = current * scale.clamp(0.5, 1.5)
    if float(dropout_prob) > 0.0:
        keep = torch.rand((x.shape[0], 1, 1), device=x.device, dtype=x.dtype).ge(float(dropout_prob)).float()
        current = current * keep
    out[..., 1:2] = current
    return out


SSL_VOLTAGE_TARGET_CANDIDATES = [
    "V_corr_raw",
    "dV_corr",
    "abs_dV_corr",
    "d2V_corr",
    "V_corr_raw_ema50",
    "V_corr_raw_dev_ema50",
    "V_corr_raw_ema200",
    "V_corr_raw_dev_ema200",
    "V_corr_raw_ema800",
    "V_corr_raw_dev_ema800",
]


def _ssl_encoder_modules(model: nn.Module) -> tuple[nn.Module, list[nn.Parameter]]:
    encoder = getattr(model, "dynamic", model)
    params: list[nn.Parameter] = []
    for name in ("input_proj", "tcn_blocks", "rnn", "norm"):
        module = getattr(encoder, name, None)
        if module is not None:
            params.extend(module.parameters())
    if not params:
        raise RuntimeError("Voltage-shape SSL requires a model with input_proj and sequence encoder modules.")
    return encoder, params


def _voltage_shape_ssl_pretrain(
    model: nn.Module,
    loader,
    cfg: TrainDSTSelectorConfig,
    feature_cols: list[str],
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    if int(cfg.ssl_pretrain_epochs) <= 0:
        return pd.DataFrame()
    target_cols = [col for col in SSL_VOLTAGE_TARGET_CANDIDATES if col in feature_cols]
    if "V_corr_raw" not in feature_cols or not target_cols:
        raise RuntimeError("Voltage-shape SSL requires V_corr_raw and at least one voltage target column.")
    target_idx = [feature_cols.index(col) for col in target_cols]
    vcorr_idx = int(feature_cols.index("V_corr_raw"))
    encoder, encoder_params = _ssl_encoder_modules(model)
    recon_head = nn.Linear(int(cfg.hidden_size), len(target_idx)).to(device)
    next_head = nn.Linear(int(cfg.hidden_size), 1).to(device)
    slope_head = nn.Linear(int(cfg.hidden_size), 1).to(device)
    opt = torch.optim.AdamW(
        list(encoder_params) + list(recon_head.parameters()) + list(next_head.parameters()) + list(slope_head.parameters()),
        lr=float(cfg.ssl_lr),
        weight_decay=float(cfg.weight_decay),
    )
    rows = []
    for ep in range(1, int(cfg.ssl_pretrain_epochs) + 1):
        model.train()
        recon_head.train()
        next_head.train()
        slope_head.train()
        losses = []
        recon_losses = []
        next_losses = []
        slope_losses = []
        for x, _y, _meta in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            x_masked = x.clone()
            x_masked[:, :, target_idx] = 0.0
            h_masked = encoder.encode_sequence(x_masked)
            recon = recon_head(h_masked)
            recon_target = x[:, :, target_idx]
            recon_loss = F.smooth_l1_loss(recon, recon_target, beta=float(cfg.huber_beta))
            if x.shape[1] > 1:
                h = encoder.encode_sequence(x)
                next_pred = next_head(h[:, :-1, :])
                next_target = x[:, 1:, vcorr_idx:vcorr_idx + 1]
                next_loss = F.smooth_l1_loss(next_pred, next_target, beta=float(cfg.huber_beta))
                slope_pred = slope_head(h[:, :-1, :])
                slope_target = x[:, 1:, vcorr_idx:vcorr_idx + 1] - x[:, :-1, vcorr_idx:vcorr_idx + 1]
                slope_loss = F.smooth_l1_loss(slope_pred, slope_target, beta=float(cfg.huber_beta))
            else:
                next_loss = recon_loss.new_tensor(0.0)
                slope_loss = recon_loss.new_tensor(0.0)
            loss = (
                float(cfg.ssl_recon_weight) * recon_loss
                + float(cfg.ssl_next_vcorr_weight) * next_loss
                + float(cfg.ssl_slope_weight) * slope_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder_params) + list(recon_head.parameters()) + list(next_head.parameters()) + list(slope_head.parameters()), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            next_losses.append(float(next_loss.detach().cpu()))
            slope_losses.append(float(slope_loss.detach().cpu()))
        row = {
            "seed": int(seed),
            "epoch": int(ep),
            "ssl_loss": float(np.mean(losses)),
            "ssl_recon_loss": float(np.mean(recon_losses)),
            "ssl_next_vcorr_loss": float(np.mean(next_losses)),
            "ssl_slope_loss": float(np.mean(slope_losses)),
            "ssl_target_columns": ",".join(target_cols),
            "pretrain_uses_soc_labels": False,
        }
        rows.append(row)
        if ep == 1 or ep == int(cfg.ssl_pretrain_epochs) or ep % max(1, int(cfg.eval_every)) == 0:
            print(
                f"seed={seed} ssl_epoch={ep}/{cfg.ssl_pretrain_epochs} "
                f"loss={row['ssl_loss']:.6f} recon={row['ssl_recon_loss']:.6f} "
                f"next={row['ssl_next_vcorr_loss']:.6f} slope={row['ssl_slope_loss']:.6f}",
                flush=True,
            )
    history = pd.DataFrame(rows)
    history.to_csv(out_dir / f"{cfg.output_prefix}_seed{seed}_voltage_shape_ssl_history.csv", index=False)
    return history


def _stage2_selected_epoch_for_rule(cfg: TrainDSTSelectorConfig, selected_epoch: int) -> int:
    rule = str(cfg.stage2_select_rule)
    if rule == "stage1_epoch_30_35":
        if int(selected_epoch) <= int(cfg.stage2_stage1_threshold):
            return int(cfg.stage2_early_epoch)
        return int(cfg.stage2_late_epoch)
    if rule.startswith("fixed"):
        return int(rule.replace("fixed", ""))
    return int(cfg.stage2_epochs)


def _parse_stage1_ensemble_epochs(raw: str, selected_epoch: int) -> list[int]:
    text = str(raw or "").strip().lower()
    if not text or text in {"none", "selected"}:
        return [int(selected_epoch)]
    if text.startswith("range:"):
        parts = [int(x) for x in text.split(":", 2)[1:]]
        if len(parts) != 2:
            raise ValueError(f"Invalid stage1_ensemble_epochs={raw!r}")
        start, end = parts
        if end < start:
            raise ValueError(f"Invalid stage1_ensemble_epochs range {raw!r}")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_stage1_model_for_epoch(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    state_by_epoch: dict[int, dict[str, torch.Tensor]],
    epoch: int,
    input_dim: int,
) -> nn.Module:
    if int(epoch) not in state_by_epoch:
        raise RuntimeError(f"Requested stage1 ensemble epoch {epoch} is missing from state_by_epoch.")
    model = _make_stage1_model(cfg, variant, input_dim=int(input_dim))
    model.load_state_dict({k: v.to(device) for k, v in state_by_epoch[int(epoch)].items()})
    model.eval()
    return model


def _make_selected_stage1_base(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    state_by_epoch: dict[int, dict[str, torch.Tensor]],
    selected_epoch: int,
    input_dim: int,
) -> tuple[nn.Module, list[int]]:
    epochs = _parse_stage1_ensemble_epochs(str(cfg.stage1_ensemble_epochs), int(selected_epoch))
    models = [_load_stage1_model_for_epoch(cfg, variant, state_by_epoch, ep, input_dim=int(input_dim)) for ep in epochs]
    if len(models) == 1:
        setattr(models[0], "input_dim_for_stage2", int(input_dim))
        return models[0], epochs
    ensemble = SnapshotEnsembleBase(models).to(device)
    setattr(ensemble, "input_dim_for_stage2", int(input_dim))
    return ensemble, epochs


def _train_stage2_correction_test_blind(
    cfg: TrainDSTSelectorConfig,
    variant: CondInvVariant,
    base_model,
    train_loader,
    valid_loader,
    test_loader,
    seed: int,
    selected_epoch: int,
) -> pd.DataFrame:
    target_epoch = min(int(cfg.stage2_epochs), _stage2_selected_epoch_for_rule(cfg, int(selected_epoch)))
    stage2_rule = str(cfg.stage2_select_rule)
    stage2_val_rule = stage2_rule in {"val_mean_mae", "val_worst_mae", "val_mean_plus_worst"}
    corrected = Stage2CorrectedModel(
        base_model,
        int(cfg.hidden_size),
        float(variant.corr_limit),
        str(variant.corr_mode),
        freeze_base=True,
        input_dim=int(
            getattr(
                base_model,
                "correction_input_dim_for_stage2",
                getattr(base_model, "input_dim_for_stage2", 3),
            )
        ),
        base_input_dim=int(getattr(base_model, "input_dim_for_stage2", 3)),
        corr_zero_init=bool(cfg.corr_zero_init),
    ).to(device)
    opt = torch.optim.AdamW(
        [p for p in corrected.parameters() if p.requires_grad],
        lr=float(cfg.lr_stage2),
        weight_decay=float(cfg.weight_decay),
    )
    rows = []
    history = []
    valid_parts = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for ep in range(1, int(target_epoch) + 1):
        corrected.train()
        losses = []
        supervised_losses = []
        keep_losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y_endpoint = y[:, -1, :] if y.ndim == 3 else y
            with torch.no_grad():
                base_pred = corrected.base_prediction(x)
            pred = corrected(x)
            sample_loss = F.smooth_l1_loss(pred, y_endpoint, beta=float(cfg.huber_beta), reduction="none").mean(dim=1)
            temps = _meta_float_tensor(meta, "temperature")
            temp45 = torch.isclose(temps, torch.full_like(temps, 45.0), atol=0.75).float()
            if variant.corr_mode == "temp_bands":
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(temp25, temp45).detach()
            elif variant.corr_mode in {"cold_hot", "cold_hot_deep"}:
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                focus = torch.maximum(temp0, temp45).detach()
            elif variant.corr_mode == "cold_mid_hot":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(torch.maximum(temp0, temp25), temp45).detach()
            elif variant.corr_mode == "cold_hot_midlite":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                temp25 = torch.isclose(temps, torch.full_like(temps, 25.0), atol=0.75).float()
                focus = torch.maximum(temp0, temp45) + 0.25 * temp25
                focus = focus.clamp(max=1.0).detach()
            elif variant.corr_mode == "all_temp":
                focus = torch.ones_like(temp45).detach()
            elif variant.corr_mode == "cold_only":
                temp0 = torch.isclose(temps, torch.full_like(temps, 0.0), atol=0.75).float()
                focus = temp0.detach()
            elif variant.corr_mode == "hot_only":
                focus = temp45.detach()
            else:
                low_voltage_focus = torch.sigmoid(-(x[:, -1, 0] + 0.25) * 5.0)
                focus = (temp45 * low_voltage_focus).detach()
            weights = 1.0 + (float(cfg.focus45_weight) - 1.0) * focus
            supervised = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
            keep_weight = (1.0 - focus).clamp_min(0.0).unsqueeze(1)
            keep = ((pred - base_pred).square() * keep_weight).sum() / keep_weight.sum().clamp_min(1e-6)
            loss = supervised + float(cfg.keep_lambda) * keep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(corrected.correction.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            supervised_losses.append(float(supervised.detach().cpu()))
            keep_losses.append(float(keep.detach().cpu()))
        history.append(
            {
                "seed": int(seed),
                "variant": f"{variant.name}_stage2",
                "epoch": int(ep),
                "loss": float(np.mean(losses)),
                "supervised_loss": float(np.mean(supervised_losses)),
                "keep_loss": float(np.mean(keep_losses)),
                "test_blind": True,
                "stage2_train_until_epoch": int(target_epoch),
                "stage2_select_rule": stage2_rule,
            }
        )
        if ep == 1 or ep % int(cfg.eval_every) == 0 or ep == int(target_epoch):
            valid = eval_by_temp(corrected, valid_loader, "valid", f"{variant.name}_stage2", ep)
            valid["test_blind"] = True
            valid_parts.append(valid.copy())
            if stage2_val_rule:
                snapshots[int(ep)] = copy.deepcopy({k: v.detach().cpu() for k, v in corrected.state_dict().items()})
            if int(cfg.test_blind_rng_burn) > 0:
                _burn_global_rng(int(cfg.test_blind_rng_burn))
            rows.append(valid)
            piv = valid.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
            mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            print(
                f"{variant.name}_stage2 epoch={ep} loss={np.mean(losses):.5f} "
                f"supervised={np.mean(supervised_losses):.5f} keep={np.mean(keep_losses):.5f} "
                f"valid0={mae0:.3f}% valid25={mae25:.3f}% valid45={mae45:.3f}% test=hidden",
                flush=True,
            )
    selected_stage2_epoch = int(target_epoch)
    stage2_selector_score = float("nan")
    stage2_val_mean = float("nan")
    stage2_val_worst = float("nan")
    if stage2_val_rule:
        if not valid_parts:
            raise RuntimeError("Stage 2 validation selector has no validation rows.")
        valid_all = pd.concat(valid_parts, ignore_index=True)
        per_epoch = (
            valid_all.groupby("epoch", as_index=False)["MAE_pct"]
            .agg(val_mean_mae="mean", val_worst_mae="max")
        )
        if stage2_rule == "val_mean_plus_worst":
            per_epoch["selector_score"] = per_epoch["val_mean_mae"] + 0.5 * per_epoch["val_worst_mae"]
        else:
            per_epoch["selector_score"] = per_epoch[stage2_rule]
        best = per_epoch.sort_values(["selector_score", "epoch"]).iloc[0]
        selected_stage2_epoch = int(best["epoch"])
        stage2_selector_score = float(best["selector_score"])
        stage2_val_mean = float(best["val_mean_mae"])
        stage2_val_worst = float(best["val_worst_mae"])
        if selected_stage2_epoch not in snapshots:
            raise RuntimeError(f"Missing Stage 2 snapshot for selected epoch {selected_stage2_epoch}.")
        corrected.load_state_dict(snapshots[selected_stage2_epoch], strict=True)
    test = eval_by_temp(corrected, test_loader, "test", f"{variant.name}_stage2", int(selected_stage2_epoch))
    test["test_blind"] = True
    test["stage2_selected_epoch"] = int(selected_stage2_epoch)
    test["stage2_select_rule"] = stage2_rule
    test["stage2_selector_score"] = stage2_selector_score
    test["val_mean_mae"] = stage2_val_mean
    test["val_worst_mae"] = stage2_val_worst
    rows.append(test)
    if bool(cfg.save_predictions):
        pred_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_prefix = f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{variant.name}_stage2"
        _predict_rows(corrected, valid_loader, "valid", f"{variant.name}_stage2", int(selected_stage2_epoch)).to_csv(
            pred_dir / f"{pred_prefix}_valid_prediction_rows.csv.gz",
            index=False,
            compression="gzip",
        )
        _predict_rows(corrected, test_loader, "test", f"{variant.name}_stage2", int(selected_stage2_epoch)).to_csv(
            pred_dir / f"{pred_prefix}_test_prediction_rows.csv.gz",
            index=False,
            compression="gzip",
        )
    history_dir = cfg.base_dir / "nmc_goal_vcorr_it_conditional_invariant_results"
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        history_dir / f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{variant.name}_stage2_test_blind_history.csv",
        index=False,
    )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


REGIME_SELECTOR_METRICS = ["absI_mean", "I_std", "dI_energy", "V_corr_span"]


def _regime_edges_from_train_ds(ds) -> dict[str, tuple[float, float]]:
    rows = _window_regime_rows(ds)
    return {col: _regime_edges(rows[col].tolist()) for col in REGIME_SELECTOR_METRICS}


def _regime_stats_from_batch(x_cpu: np.ndarray, feature_cols: list[str], edges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    if "I_raw" not in feature_to_idx or "V_corr_raw" not in feature_to_idx:
        raise RuntimeError("Regime-sliced selector requires I_raw and V_corr_raw in the selected feature set.")
    i = x_cpu[:, :, feature_to_idx["I_raw"]].astype(np.float64)
    v = x_cpu[:, :, feature_to_idx["V_corr_raw"]].astype(np.float64)
    if "absI" in feature_to_idx:
        abs_i = np.abs(x_cpu[:, :, feature_to_idx["absI"]].astype(np.float64))
    else:
        abs_i = np.abs(i)
    if "dI" in feature_to_idx:
        d_i = x_cpu[:, :, feature_to_idx["dI"]].astype(np.float64)
    else:
        d_i = np.diff(i, axis=1, prepend=i[:, :1])
    rows = pd.DataFrame(
        {
            "absI_mean": np.nanmean(abs_i, axis=1),
            "I_std": np.nanstd(i, axis=1),
            "dI_energy": np.nanmean(np.square(d_i), axis=1),
            "V_corr_span": np.nanmax(v, axis=1) - np.nanmin(v, axis=1),
        }
    )
    for col in REGIME_SELECTOR_METRICS:
        rows[f"{col}_bin"] = [_metric_bin(float(value), edges[col]) for value in rows[col]]
    rows["regime_key"] = [
        "|".join(str(row[f"{col}_bin"]) for col in REGIME_SELECTOR_METRICS)
        for _, row in rows.iterrows()
    ]
    return rows


@torch.no_grad()
def eval_by_temp_regime(
    model: nn.Module,
    loader,
    split: str,
    variant_name: str,
    seed: int,
    epoch: int,
    feature_cols: list[str],
    train_regime_edges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, meta in loader:
        x_cpu = x.detach().cpu().numpy().astype(np.float32)
        regime = _regime_stats_from_batch(x_cpu, feature_cols, train_regime_edges)
        x_dev = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x_dev).detach().cpu().numpy()[:, 0]
        true = y[:, -1, 0].numpy() if y.ndim == 3 else y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        mdf = pd.concat([mdf.reset_index(drop=True), regime.reset_index(drop=True)], axis=1)
        mdf["y_pred"] = pred
        mdf["y_true"] = true
        mdf["error"] = mdf["y_pred"] - mdf["y_true"]
        for (temp, regime_key), g in mdf.groupby(["temperature", "regime_key"], dropna=False):
            err = g["error"].to_numpy(np.float64)
            row = {
                "variant": variant_name,
                "seed": int(seed),
                "epoch": int(epoch),
                "split": split,
                "temperature_C": float(temp),
                "regime_key": str(regime_key),
                "n_windows": int(len(g)),
                "sum_abs_error": float(np.sum(np.abs(err))),
                "sum_sq_error": float(np.sum(np.square(err))),
            }
            for col in REGIME_SELECTOR_METRICS:
                row[f"{col}_bin"] = str(g[f"{col}_bin"].iloc[0])
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        key_cols = [
            "variant",
            "seed",
            "epoch",
            "split",
            "temperature_C",
            "regime_key",
            *[f"{col}_bin" for col in REGIME_SELECTOR_METRICS],
        ]
        out = out.groupby(key_cols, as_index=False).agg(
            n_windows=("n_windows", "sum"),
            sum_abs_error=("sum_abs_error", "sum"),
            sum_sq_error=("sum_sq_error", "sum"),
        )
        denom = out["n_windows"].clip(lower=1).astype(float)
        out["MAE_pct"] = out["sum_abs_error"] / denom * 100.0
        out["RMSE_pct"] = np.sqrt(out["sum_sq_error"] / denom) * 100.0
        out = out.drop(columns=["sum_abs_error", "sum_sq_error"])
    return out


@torch.no_grad()
def _predict_rows(model: nn.Module, loader, split: str, variant_name: str, epoch: int) -> pd.DataFrame:
    model.eval()
    rows = []
    offset = 0
    for x, y, meta in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        pred = model(x).detach().cpu().numpy()[:, 0]
        true = y.numpy()[:, 0]
        mdf = collate_meta_to_frame(meta)
        n = len(mdf)
        mdf["row_id"] = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        mdf["split"] = split
        mdf["variant"] = variant_name
        mdf["epoch"] = int(epoch)
        mdf["target_label"] = "physical"
        mdf["y_true"] = true
        mdf["y_pred"] = pred
        mdf["error"] = mdf["y_pred"] - mdf["y_true"]
        mdf["abs_error"] = np.abs(mdf["error"])
        rows.append(mdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_eval_tensors(loader) -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    xs = []
    ys = []
    metas = []
    for x, y, meta in loader:
        xs.append(x.detach().cpu().float())
        ys.append(y.detach().cpu().numpy()[:, 0])
        metas.append(collate_meta_to_frame(meta))
    if not xs:
        return torch.empty(0), np.empty(0), pd.DataFrame()
    return torch.cat(xs, dim=0), np.concatenate(ys, axis=0), pd.concat(metas, ignore_index=True)


@torch.no_grad()
def _predict_from_tensor(model: nn.Module, x_cpu: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    n = int(x_cpu.shape[0])
    for start in range(0, n, int(batch_size)):
        xb = x_cpu[start : start + int(batch_size)].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        preds.append(model(xb).detach().cpu().numpy()[:, 0])
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)


def _indices_for_feature_group(feature_cols: list[str], group: str) -> list[int]:
    voltage = {
        "V_corr_raw_ema50",
        "V_corr_raw_dev_ema50",
        "V_corr_raw_ema200",
        "V_corr_raw_dev_ema200",
        "V_corr_raw_ema800",
        "V_corr_raw_dev_ema800",
    }
    voltage_dev = {c for c in voltage if "_dev_" in c}
    current = {
        "I_raw_ema50",
        "I_raw_dev_ema50",
        "I_raw_ema200",
        "I_raw_dev_ema200",
    }
    current_dev = {c for c in current if "_dev_" in c}
    abs_current = {
        "absI_ema50",
        "absI_dev_ema50",
        "absI_ema200",
        "absI_dev_ema200",
    }
    abs_current_dev = {c for c in abs_current if "_dev_" in c}
    groups = {
        "voltage_ema": voltage,
        "voltage_ema_dev": voltage_dev,
        "current_ema": current,
        "current_ema_dev": current_dev,
        "abs_current_ema": abs_current,
        "abs_current_ema_dev": abs_current_dev,
        "all_ema": voltage | current | abs_current,
    }
    wanted = groups[group]
    return [idx for idx, col in enumerate(feature_cols) if col in wanted]


def _shuffle_within_temp_profile(x: torch.Tensor, meta: pd.DataFrame, indices: list[int], seed: int) -> torch.Tensor:
    out = x.clone()
    if not indices or meta.empty:
        return out
    rng = np.random.default_rng(int(seed))
    key_cols = [c for c in ["temperature", "drive_cycle", "file_name"] if c in meta.columns]
    if not key_cols:
        perm = rng.permutation(int(out.shape[0]))
        out[:, :, indices] = out[perm][:, :, indices]
        return out
    for _, idx in meta.groupby(key_cols, sort=False).groups.items():
        arr_idx = np.asarray(list(idx), dtype=np.int64)
        if len(arr_idx) <= 1:
            continue
        perm = rng.permutation(arr_idx)
        arr_t = torch.as_tensor(arr_idx, dtype=torch.long)
        perm_t = torch.as_tensor(perm, dtype=torch.long)
        for feat_idx in indices:
            out[arr_t, :, feat_idx] = out[perm_t, :, feat_idx]
    return out


def _replace_ema_with_endpoint_memoryless(x: torch.Tensor, feature_cols: list[str]) -> torch.Tensor:
    out = x.clone()
    raw_index = {col: idx for idx, col in enumerate(feature_cols)}
    for idx, col in enumerate(feature_cols):
        if "_ema" not in col:
            continue
        if col.startswith("V_corr_raw_"):
            base = raw_index.get("V_corr_raw")
            if base is not None:
                out[:, :, idx] = out[:, -1:, base]
        elif col.startswith("I_raw_"):
            base = raw_index.get("I_raw")
            if base is not None:
                out[:, :, idx] = out[:, -1:, base]
        elif col.startswith("absI_"):
            base = raw_index.get("I_raw")
            if base is not None:
                out[:, :, idx] = out[:, -1:, base].abs()
    return out


def _ema_perturbation_importance(
    model: nn.Module,
    loader,
    feature_cols: list[str],
    variant_name: str,
    epoch: int,
    seed: int,
    batch_size: int,
) -> pd.DataFrame:
    x, y_true, meta = _collect_eval_tensors(loader)
    if x.numel() == 0:
        return pd.DataFrame()
    perturbations: list[tuple[str, torch.Tensor, str]] = []
    perturbations.append(("P0_no_perturbation", x, "original selected G4 input"))
    x1 = x.clone()
    x1[:, :, _indices_for_feature_group(feature_cols, "voltage_ema_dev")] = 0.0
    perturbations.append(("P1_zero_voltage_ema_deviation", x1, "zero V_corr_raw_dev_ema* channels"))
    x2 = x.clone()
    x2[:, :, _indices_for_feature_group(feature_cols, "current_ema_dev")] = 0.0
    perturbations.append(("P2_zero_current_ema_deviation", x2, "zero I_raw_dev_ema* channels"))
    x3 = x.clone()
    x3[:, :, _indices_for_feature_group(feature_cols, "abs_current_ema_dev")] = 0.0
    perturbations.append(("P3_zero_abs_current_ema_deviation", x3, "zero absI_dev_ema* channels"))
    perturbations.append(
        (
            "P4_shuffle_voltage_ema",
            _shuffle_within_temp_profile(x, meta, _indices_for_feature_group(feature_cols, "voltage_ema"), int(seed) + 4001),
            "shuffle V_corr EMA channels within same temperature/profile/file groups",
        )
    )
    perturbations.append(
        (
            "P5_shuffle_current_ema",
            _shuffle_within_temp_profile(
                x,
                meta,
                _indices_for_feature_group(feature_cols, "current_ema")
                + _indices_for_feature_group(feature_cols, "abs_current_ema"),
                int(seed) + 5001,
            ),
            "shuffle current and abs-current EMA channels within same temperature/profile/file groups",
        )
    )
    perturbations.append(
        (
            "P6_endpoint_raw_memoryless_ema",
            _replace_ema_with_endpoint_memoryless(x, feature_cols),
            "replace EMA channels with endpoint raw proxy values to destroy within-window EMA memory",
        )
    )
    rows = []
    baseline_mae_by_temp: dict[float, float] = {}
    for perturbation, x_pert, detail in perturbations:
        y_pred = _predict_from_tensor(model, x_pert, int(batch_size))
        err = y_pred - y_true
        m = meta.copy()
        m["error"] = err
        m["abs_error"] = np.abs(err)
        for temp, g in m.groupby("temperature", sort=True):
            mae = float(g["abs_error"].mean() * 100.0)
            if perturbation == "P0_no_perturbation":
                baseline_mae_by_temp[float(temp)] = mae
            rows.append(
                {
                    "seed": int(seed),
                    "variant": variant_name,
                    "epoch": int(epoch),
                    "perturbation": perturbation,
                    "temperature_C": float(temp),
                    "n_windows": int(len(g)),
                    "MAE_pct": mae,
                    "RMSE_pct": float(np.sqrt(np.mean(np.square(g["error"].to_numpy(np.float64)))) * 100.0),
                    "delta_MAE_vs_P0_pct": mae - float(baseline_mae_by_temp.get(float(temp), mae)),
                    "detail": detail,
                }
            )
        mae_all = float(m["abs_error"].mean() * 100.0)
        base_all = baseline_mae_by_temp.get(float("inf"), mae_all)
        if perturbation == "P0_no_perturbation":
            baseline_mae_by_temp[float("inf")] = mae_all
            base_all = mae_all
        rows.append(
            {
                "seed": int(seed),
                "variant": variant_name,
                "epoch": int(epoch),
                "perturbation": perturbation,
                "temperature_C": "ALL",
                "n_windows": int(len(m)),
                "MAE_pct": mae_all,
                "RMSE_pct": float(np.sqrt(np.mean(np.square(m["error"].to_numpy(np.float64)))) * 100.0),
                "delta_MAE_vs_P0_pct": mae_all - float(base_all),
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def _train_select_and_correct(cfg: TrainDSTSelectorConfig, frames, out_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    feature_cols = _selected_feature_columns(str(cfg.feature_set))
    stage2_feature_set = str(cfg.stage2_feature_set or cfg.feature_set)
    stage2_feature_cols = _selected_feature_columns(stage2_feature_set)
    scaled, _ = make_scaled_frames_for_ablation(frames, feature_cols)
    if stage2_feature_cols == feature_cols:
        scaled_stage2 = scaled
    else:
        scaled_stage2, _ = make_scaled_frames_for_ablation(frames, stage2_feature_cols)
    scaled_stage1 = _filter_scaled_frames_by_temperatures(
        scaled,
        train_temperatures=tuple(cfg.train_temperatures),
        valid_temperatures=tuple(cfg.valid_temperatures),
        test_temperatures=tuple(cfg.test_temperatures),
        name="stage1",
    )
    stage2_train_temperatures = tuple(cfg.stage2_train_temperatures or cfg.train_temperatures)
    stage2_valid_temperatures = tuple(cfg.stage2_valid_temperatures or cfg.valid_temperatures)
    stage2_test_temperatures = tuple(cfg.stage2_test_temperatures or cfg.test_temperatures)
    scaled_stage2_filtered = _filter_scaled_frames_by_temperatures(
        scaled_stage2,
        train_temperatures=stage2_train_temperatures,
        valid_temperatures=stage2_valid_temperatures,
        test_temperatures=stage2_test_temperatures,
        name="stage2",
    )
    base_cfg = make_cfg()
    base_cfg.output_dir = out_dir
    base_cfg.batch_size = int(cfg.batch_size)
    base_cfg.dataloader_num_workers = int(cfg.num_workers)
    base_cfg.dataloader_prefetch_factor = int(cfg.prefetch_factor)
    base_cfg.dataloader_pin_memory = True
    base_cfg.dataloader_persistent_workers = int(cfg.num_workers) > 0
    if str(cfg.sampler_seed_mode) == "seed":
        base_cfg.sampler_seed = int(seed)
    elif str(cfg.sampler_seed_mode) == "none":
        base_cfg.sampler_seed = None
    else:
        raise ValueError(f"Unknown sampler_seed_mode={cfg.sampler_seed_mode!r}")

    train_dataset_cls = SequenceWindowDataset if bool(cfg.sequence_training) else DecomposedWindowDataset
    train_ds = train_dataset_cls(scaled_stage1["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    train_eval_ds = DecomposedWindowDataset(scaled_stage1["train"], feature_cols, cfg.window_len, cfg.stride, target_label="physical")
    valid_ds = DecomposedWindowDataset(scaled_stage1["valid"], feature_cols, cfg.window_len, 1, target_label="physical")
    test_ds = DecomposedWindowDataset(scaled_stage1["test"], feature_cols, cfg.window_len, 1, target_label="physical")
    train_regime_edges = _regime_edges_from_train_ds(train_eval_ds)
    _write_regime_sampler_audit(train_ds, cfg, out_dir, int(seed))
    train_loader = _make_train_loader(train_ds, base_cfg, cfg, int(seed))
    train_eval_loader = make_eval_loader(train_eval_ds, cfg)
    valid_loader = make_eval_loader(valid_ds, cfg)
    test_loader = make_eval_loader(test_ds, cfg)
    stage2_train_ds = train_dataset_cls(
        scaled_stage2_filtered["train"],
        stage2_feature_cols,
        cfg.window_len,
        cfg.stride,
        target_label="physical",
    )
    stage2_valid_ds = DecomposedWindowDataset(
        scaled_stage2_filtered["valid"],
        stage2_feature_cols,
        cfg.window_len,
        1,
        target_label="physical",
    )
    stage2_test_ds = DecomposedWindowDataset(
        scaled_stage2_filtered["test"],
        stage2_feature_cols,
        cfg.window_len,
        1,
        target_label="physical",
    )
    stage2_train_loader = _make_train_loader(stage2_train_ds, base_cfg, cfg, int(seed))
    stage2_valid_loader = make_eval_loader(stage2_valid_ds, cfg)
    stage2_test_loader = make_eval_loader(stage2_test_ds, cfg)
    finetune_train_loader = None
    finetune_train_eval_loader = train_eval_loader
    if int(cfg.finetune_epochs) > 0:
        if not tuple(cfg.finetune_temperatures):
            raise RuntimeError("--finetune-epochs requires --finetune-temperatures.")
        scaled_finetune_train = _filter_frame_list_by_temperatures(
            scaled["train"],
            tuple(cfg.finetune_temperatures),
            split_name="finetune:train",
        )
        finetune_train_ds = train_dataset_cls(
            scaled_finetune_train,
            feature_cols,
            cfg.window_len,
            cfg.stride,
            target_label="physical",
        )
        finetune_train_eval_ds = DecomposedWindowDataset(
            scaled_finetune_train,
            feature_cols,
            cfg.window_len,
            cfg.stride,
            target_label="physical",
        )
        finetune_train_loader = _make_train_loader(finetune_train_ds, base_cfg, cfg, int(seed) + 100003)
        finetune_train_eval_loader = make_eval_loader(finetune_train_eval_ds, cfg)

    variant = CondInvVariant(
        f"condinv_{cfg.recurrent}{cfg.layers}_{cfg.temp_mode}_{cfg.head_kind}_mmd0p02_trainDST25_selector_base",
        recurrent=str(cfg.recurrent),
        lambda_condinv=float(cfg.lambda_condinv),
        lambda_rex=float(cfg.lambda_rex),
        weight_0=float(cfg.weight_0),
        weight_25=float(cfg.weight_25),
        weight_45=float(cfg.weight_45),
        layers=int(cfg.layers),
        kernel_size=int(cfg.kernel_size),
        head_kind=str(cfg.head_kind),
        temp_mode=str(cfg.temp_mode),
        dropout=float(cfg.dropout),
        sequence_training=bool(cfg.sequence_training),
    )
    model = _make_stage1_model(cfg, variant, input_dim=len(feature_cols), feature_cols=feature_cols)
    _voltage_shape_ssl_pretrain(model, train_loader, cfg, feature_cols, out_dir, int(seed))
    profile_adv_head = None
    profile_to_label = {str(profile).upper(): idx for idx, profile in enumerate(cfg.train_profiles)}
    if float(cfg.lambda_profile_adv) > 0.0:
        profile_adv_head = nn.Sequential(
            nn.Linear(int(cfg.hidden_size), int(cfg.hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_size), max(2, len(profile_to_label))),
        ).to(device)
    opt_params = list(model.parameters())
    if profile_adv_head is not None:
        opt_params.extend(profile_adv_head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    legacy_variant = to_variant(variant)
    state_by_epoch = {}
    metric_rows = []
    regime_metric_rows = []
    history = []
    selector_rows = []
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        mmds = []
        adv_losses = []
        adv_accs = []
        supcon_losses = []
        consistency_losses = []
        for x, y, meta in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            h = model.encode_sequence(x)
            h_last = h[:, -1, :]
            if bool(variant.sequence_training):
                pred = model.forward_sequence(x)
                sample_loss = _regression_loss(pred, y, cfg, reduction="none").mean(dim=(1, 2))
                y_for_mmd = y[:, -1, :]
            else:
                pred = model(x)
                sample_loss = _regression_loss(pred, y, cfg, reduction="none").mean(dim=1)
                y_for_mmd = y
            sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
            keys = group_keys(meta, cfg.rex_group)
            group_losses = []
            group_weights = []
            for key in sorted(set(keys)):
                idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                group_losses.append(sample_loss.index_select(0, idx).mean())
                group_weights.append(sw.index_select(0, idx).mean())
            stack = torch.stack(group_losses)
            wstack = torch.stack(group_weights)
            mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
            rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
            mmd_loss = conditional_profile_mmd(h_last, y_for_mmd, meta)
            loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
            if float(cfg.lambda_anchor_loss) > 0.0 and hasattr(model, "anchor_sequence"):
                anchor_seq = model.anchor_sequence(x)
                if bool(variant.sequence_training):
                    anchor_loss = _regression_loss(anchor_seq, y, cfg)
                else:
                    anchor_loss = _regression_loss(anchor_seq[:, -1, :], y, cfg)
                loss = loss + float(cfg.lambda_anchor_loss) * anchor_loss
            if float(cfg.lambda_profile_supcon) > 0.0:
                supcon_loss = _profile_supcon_loss(
                    h_last,
                    y_for_mmd,
                    meta,
                    temperature=float(cfg.supcon_temperature),
                )
                loss = loss + float(cfg.lambda_profile_supcon) * supcon_loss
                supcon_losses.append(float(supcon_loss.detach().cpu()))
            if profile_adv_head is not None:
                profile_labels = _profile_label_tensor(meta, profile_to_label)
                adv_logits = profile_adv_head(_grad_reverse(h_last, float(cfg.profile_adv_grl)))
                adv_loss = F.cross_entropy(adv_logits, profile_labels)
                loss = loss + float(cfg.lambda_profile_adv) * adv_loss
                adv_losses.append(float(adv_loss.detach().cpu()))
                adv_accs.append(float((adv_logits.argmax(dim=1) == profile_labels).float().mean().detach().cpu()))
            if float(cfg.lambda_current_consistency) > 0.0:
                x_aug = _augment_current_channel(
                    x,
                    noise_std=float(cfg.current_noise_std),
                    dropout_prob=float(cfg.current_dropout_prob),
                )
                pred_aug = model.forward_sequence(x_aug) if bool(variant.sequence_training) else model(x_aug)
                consistency = _regression_loss(pred_aug, pred.detach(), cfg)
                loss = loss + float(cfg.lambda_current_consistency) * consistency
                consistency_losses.append(float(consistency.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mmds.append(float(mmd_loss.detach().cpu()))
        state_by_epoch[int(ep)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "seed": int(seed),
                "epoch": int(ep),
                "variant": variant.name,
                "loss": float(np.mean(losses)),
                "condinv_loss": float(np.mean(mmds)),
                "profile_supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
                "profile_adv_loss": float(np.mean(adv_losses)) if adv_losses else 0.0,
                "profile_adv_acc": float(np.mean(adv_accs)) if adv_accs else np.nan,
                "current_consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
            }
        )
        do_stage1_eval = (
            int(ep) == 1
            or int(ep) % max(1, int(cfg.stage1_eval_every)) == 0
            or int(ep) == int(cfg.epochs)
        )
        if do_stage1_eval:
            train_metrics = eval_by_temp_drive(model, train_eval_loader, "train", variant.name, seed, ep)
            valid_metrics = eval_by_temp(model, valid_loader, "valid", variant.name, ep)
            test_metrics = pd.DataFrame()
            diagnostic_test = (
                bool(cfg.test_blind)
                and int(cfg.diagnostic_test_every) > 0
                and (int(ep) % int(cfg.diagnostic_test_every) == 0 or int(ep) == int(cfg.epochs))
            )
            if (not bool(cfg.test_blind)) or diagnostic_test:
                test_metrics = eval_by_temp(model, test_loader, "test", variant.name, ep)
                if diagnostic_test:
                    test_metrics["diagnostic_test_peek"] = True
            elif int(cfg.test_blind_rng_burn) > 0:
                _burn_global_rng(int(cfg.test_blind_rng_burn))
            for df in (valid_metrics, test_metrics):
                df["seed"] = int(seed)
            metric_rows.extend([df for df in (train_metrics, valid_metrics, test_metrics) if len(df)])
            valid_regime_metrics = pd.DataFrame()
            if str(cfg.stage1_selector).startswith("val_regime"):
                valid_regime_metrics = eval_by_temp_regime(
                    model,
                    valid_loader,
                    "valid",
                    variant.name,
                    int(seed),
                    int(ep),
                    feature_cols,
                    train_regime_edges,
                )
                if len(valid_regime_metrics):
                    regime_metric_rows.append(valid_regime_metrics)
            selector_payload = _selector_score(
                train_metrics,
                str(cfg.stage1_selector),
                valid_metrics,
                valid_regime_metrics,
                int(cfg.selector_regime_min_windows),
            )
            selector_rows.append(
                {
                    "seed": int(seed),
                    "epoch": int(ep),
                    "selector": str(cfg.stage1_selector),
                    **selector_payload,
                }
            )
            report_metrics = valid_metrics if bool(cfg.test_blind) else test_metrics
            if len(report_metrics) and {"variant", "epoch", "split", "temperature_C", "MAE_pct"}.issubset(report_metrics.columns):
                piv = report_metrics.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
                mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
            else:
                mae0 = mae25 = mae45 = float("nan")
            label = "valid" if bool(cfg.test_blind) else "test"
            print(
                f"seed={seed} epoch={ep} loss={np.mean(losses):.5f} {cfg.stage1_selector}={selector_rows[-1]['selector_score']:.3f}% "
                f"{label}0={mae0:.3f}% {label}25={mae25:.3f}% {label}45={mae45:.3f}%"
                + (" test=hidden" if bool(cfg.test_blind) else ""),
                flush=True,
            )
        else:
            if int(ep) % 10 == 0:
                print(f"seed={seed} epoch={ep} loss={np.mean(losses):.5f} eval=skipped", flush=True)

    if finetune_train_loader is not None:
        opt = torch.optim.AdamW(
            opt_params,
            lr=float(cfg.lr) * float(cfg.finetune_lr_scale),
            weight_decay=float(cfg.weight_decay),
        )
        total_epochs = int(cfg.epochs) + int(cfg.finetune_epochs)
        for ft_ep in range(1, int(cfg.finetune_epochs) + 1):
            ep = int(cfg.epochs) + int(ft_ep)
            model.train()
            losses = []
            mmds = []
            adv_losses = []
            adv_accs = []
            supcon_losses = []
            consistency_losses = []
            for x, y, meta in finetune_train_loader:
                x = x.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                y = y.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                h = model.encode_sequence(x)
                h_last = h[:, -1, :]
                if bool(variant.sequence_training):
                    pred = model.forward_sequence(x)
                    sample_loss = _regression_loss(pred, y, cfg, reduction="none").mean(dim=(1, 2))
                    y_for_mmd = y[:, -1, :]
                else:
                    pred = model(x)
                    sample_loss = _regression_loss(pred, y, cfg, reduction="none").mean(dim=1)
                    y_for_mmd = y
                sw = temp_weights(meta, legacy_variant, int(sample_loss.numel()))
                keys = group_keys(meta, cfg.rex_group)
                group_losses = []
                group_weights = []
                for key in sorted(set(keys)):
                    idx = torch.as_tensor([i for i, k in enumerate(keys) if k == key], device=device, dtype=torch.long)
                    group_losses.append(sample_loss.index_select(0, idx).mean())
                    group_weights.append(sw.index_select(0, idx).mean())
                stack = torch.stack(group_losses)
                wstack = torch.stack(group_weights)
                mean_loss = (stack * wstack).sum() / wstack.sum().clamp_min(1e-6)
                rex_var = stack.var(unbiased=False) if len(stack) > 1 else stack.new_tensor(0.0)
                mmd_loss = conditional_profile_mmd(h_last, y_for_mmd, meta)
                loss = mean_loss + float(variant.lambda_rex) * rex_var + float(variant.lambda_condinv) * mmd_loss
                if float(cfg.lambda_anchor_loss) > 0.0 and hasattr(model, "anchor_sequence"):
                    anchor_seq = model.anchor_sequence(x)
                    if bool(variant.sequence_training):
                        anchor_loss = _regression_loss(anchor_seq, y, cfg)
                    else:
                        anchor_loss = _regression_loss(anchor_seq[:, -1, :], y, cfg)
                    loss = loss + float(cfg.lambda_anchor_loss) * anchor_loss
                if float(cfg.lambda_profile_supcon) > 0.0:
                    supcon_loss = _profile_supcon_loss(
                        h_last,
                        y_for_mmd,
                        meta,
                        temperature=float(cfg.supcon_temperature),
                    )
                    loss = loss + float(cfg.lambda_profile_supcon) * supcon_loss
                    supcon_losses.append(float(supcon_loss.detach().cpu()))
                if profile_adv_head is not None:
                    profile_labels = _profile_label_tensor(meta, profile_to_label)
                    adv_logits = profile_adv_head(_grad_reverse(h_last, float(cfg.profile_adv_grl)))
                    adv_loss = F.cross_entropy(adv_logits, profile_labels)
                    loss = loss + float(cfg.lambda_profile_adv) * adv_loss
                    adv_losses.append(float(adv_loss.detach().cpu()))
                    adv_accs.append(float((adv_logits.argmax(dim=1) == profile_labels).float().mean().detach().cpu()))
                if float(cfg.lambda_current_consistency) > 0.0:
                    x_aug = _augment_current_channel(
                        x,
                        noise_std=float(cfg.current_noise_std),
                        dropout_prob=float(cfg.current_dropout_prob),
                    )
                    pred_aug = model.forward_sequence(x_aug) if bool(variant.sequence_training) else model(x_aug)
                    consistency = _regression_loss(pred_aug, pred.detach(), cfg)
                    loss = loss + float(cfg.lambda_current_consistency) * consistency
                    consistency_losses.append(float(consistency.detach().cpu()))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                mmds.append(float(mmd_loss.detach().cpu()))
            state_by_epoch[int(ep)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(
                {
                    "seed": int(seed),
                    "epoch": int(ep),
                    "variant": variant.name,
                    "phase": "finetune",
                    "loss": float(np.mean(losses)),
                    "condinv_loss": float(np.mean(mmds)),
                    "profile_supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
                    "profile_adv_loss": float(np.mean(adv_losses)) if adv_losses else 0.0,
                    "profile_adv_acc": float(np.mean(adv_accs)) if adv_accs else np.nan,
                    "current_consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
                }
            )
            do_stage1_eval = (
                int(ep) == int(cfg.epochs) + 1
                or int(ep) % max(1, int(cfg.stage1_eval_every)) == 0
                or int(ep) == total_epochs
            )
            if do_stage1_eval:
                train_metrics = eval_by_temp_drive(model, finetune_train_eval_loader, "train", variant.name, seed, ep)
                valid_metrics = eval_by_temp(model, valid_loader, "valid", variant.name, ep)
                test_metrics = pd.DataFrame()
                diagnostic_test = (
                    bool(cfg.test_blind)
                    and int(cfg.diagnostic_test_every) > 0
                    and (int(ep) % int(cfg.diagnostic_test_every) == 0 or int(ep) == total_epochs)
                )
                if (not bool(cfg.test_blind)) or diagnostic_test:
                    test_metrics = eval_by_temp(model, test_loader, "test", variant.name, ep)
                    if diagnostic_test:
                        test_metrics["diagnostic_test_peek"] = True
                elif int(cfg.test_blind_rng_burn) > 0:
                    _burn_global_rng(int(cfg.test_blind_rng_burn))
                for df in (valid_metrics, test_metrics):
                    df["seed"] = int(seed)
                metric_rows.extend([df for df in (train_metrics, valid_metrics, test_metrics) if len(df)])
                valid_regime_metrics = pd.DataFrame()
                if str(cfg.stage1_selector).startswith("val_regime"):
                    valid_regime_metrics = eval_by_temp_regime(
                        model,
                        valid_loader,
                        "valid",
                        variant.name,
                        int(seed),
                        int(ep),
                        feature_cols,
                        train_regime_edges,
                    )
                    if len(valid_regime_metrics):
                        valid_regime_metrics["stage1_phase"] = "finetune"
                        regime_metric_rows.append(valid_regime_metrics)
                selector_payload = _selector_score(
                    train_metrics,
                    str(cfg.stage1_selector),
                    valid_metrics,
                    valid_regime_metrics,
                    int(cfg.selector_regime_min_windows),
                )
                selector_rows.append(
                    {
                        "seed": int(seed),
                        "epoch": int(ep),
                        "selector": str(cfg.stage1_selector),
                        "stage1_phase": "finetune",
                        **selector_payload,
                    }
                )
                report_metrics = valid_metrics if bool(cfg.test_blind) else test_metrics
                if len(report_metrics) and {"variant", "epoch", "split", "temperature_C", "MAE_pct"}.issubset(report_metrics.columns):
                    piv = report_metrics.pivot_table(index=["variant", "epoch", "split"], columns="temperature_C", values="MAE_pct").reset_index()
                    mae0 = float(piv.get(0.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                    mae25 = float(piv.get(25.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                    mae45 = float(piv.get(45.0, pd.Series([np.nan])).iloc[0]) if len(piv) else float("nan")
                else:
                    mae0 = mae25 = mae45 = float("nan")
                label = "valid" if bool(cfg.test_blind) else "test"
                print(
                    f"seed={seed} finetune_epoch={ft_ep}/{cfg.finetune_epochs} epoch={ep} "
                    f"loss={np.mean(losses):.5f} {cfg.stage1_selector}={selector_rows[-1]['selector_score']:.3f}% "
                    f"{label}0={mae0:.3f}% {label}25={mae25:.3f}% {label}45={mae45:.3f}%"
                    + (" test=hidden" if bool(cfg.test_blind) else ""),
                    flush=True,
                )
            else:
                if int(ft_ep) % 10 == 0:
                    print(
                        f"seed={seed} finetune_epoch={ft_ep}/{cfg.finetune_epochs} epoch={ep} "
                        f"loss={np.mean(losses):.5f} eval=skipped",
                        flush=True,
                    )

    selector_df = pd.DataFrame(selector_rows)
    selector_df["selector_min_epoch"] = int(cfg.selector_min_epoch)
    selector_df["selector_max_epoch"] = int(cfg.selector_max_epoch)
    if regime_metric_rows:
        pd.concat(regime_metric_rows, ignore_index=True).to_csv(
            out_dir / f"{cfg.output_prefix}_seed{seed}_valid_regime_metrics.csv",
            index=False,
        )
    selector_pool = selector_df.copy()
    if int(cfg.selector_min_epoch) > 1:
        selector_pool = selector_pool[selector_pool["epoch"].ge(int(cfg.selector_min_epoch))].copy()
    if int(cfg.selector_max_epoch) > 0:
        selector_pool = selector_pool[selector_pool["epoch"].le(int(cfg.selector_max_epoch))].copy()
    if not len(selector_pool):
        raise RuntimeError(
            f"No selector candidates remain for selector_min_epoch={cfg.selector_min_epoch}, "
            f"selector_max_epoch={cfg.selector_max_epoch}"
        )
    if int(cfg.fixed_stage1_epoch) > 0:
        selected_epoch = int(cfg.fixed_stage1_epoch)
        if selected_epoch not in state_by_epoch:
            raise RuntimeError(
                f"fixed_stage1_epoch={selected_epoch} is missing from saved states. "
                f"Available epoch range: 1..{max(state_by_epoch)}"
            )
        selected_score = float(
            selector_pool.loc[selector_pool["epoch"].eq(selected_epoch), "selector_score"].iloc[0]
        ) if selected_epoch in set(selector_pool["epoch"].astype(int)) else float("nan")
    else:
        selected_epoch = int(selector_pool.sort_values(["selector_score", "epoch"]).iloc[0]["epoch"])
        selected_score = float(selector_pool.loc[selector_pool["epoch"].eq(selected_epoch), "selector_score"].iloc[0])
    selected_model, ensemble_epochs = _make_selected_stage1_base(
        cfg,
        variant,
        state_by_epoch,
        selected_epoch,
        input_dim=len(feature_cols),
    )
    setattr(selected_model, "correction_input_dim_for_stage2", int(len(stage2_feature_cols)))
    selected_name = f"condinv_trainDST25_selected_seed{seed}_ep{selected_epoch}_base"
    base_valid = eval_by_temp(selected_model, valid_loader, "valid", selected_name, selected_epoch)
    base_test = (
        pd.DataFrame()
        if bool(cfg.test_blind) and int(cfg.diagnostic_test_every) <= 0
        else eval_by_temp(selected_model, test_loader, "test", selected_name, selected_epoch)
    )
    if len(base_test) and bool(cfg.test_blind):
        base_test["diagnostic_test_peek"] = True
    for df in (base_valid, base_test):
        df["seed"] = int(seed)
        df["selector"] = str(cfg.stage1_selector)
        df["selector_min_epoch"] = int(cfg.selector_min_epoch)
        df["selector_max_epoch"] = int(cfg.selector_max_epoch)
        df["selected_epoch"] = selected_epoch
        df["selector_score"] = selected_score
        df["stage1_ensemble_epochs"] = ",".join(str(ep) for ep in ensemble_epochs)

    if bool(cfg.ema_perturbation_importance):
        pert = _ema_perturbation_importance(
            selected_model,
            test_loader,
            feature_cols,
            selected_name,
            int(selected_epoch),
            int(seed),
            int(cfg.batch_size),
        )
        if len(pert):
            pert.to_csv(
                out_dir / f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_ema_perturbation_importance.csv",
                index=False,
            )

    if bool(cfg.skip_stage2):
        if bool(cfg.save_predictions):
            pred_prefix = f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}_{selected_name}"
            valid_pred = _predict_rows(selected_model, valid_loader, "valid", selected_name, int(selected_epoch))
            valid_pred.to_csv(
                out_dir / f"{pred_prefix}_valid_prediction_rows.csv.gz",
                index=False,
                compression="gzip",
            )
            test_pred = _predict_rows(selected_model, test_loader, "test", selected_name, int(selected_epoch))
            if bool(cfg.test_blind):
                test_pred["diagnostic_test_peek"] = True
            test_pred.to_csv(
                out_dir / f"{pred_prefix}_test_prediction_rows.csv.gz",
                index=False,
                compression="gzip",
            )
        print(
            f"seed={seed} selected_epoch={selected_epoch} selector_max_epoch={cfg.selector_max_epoch} "
            f"selector_score={selected_score:.3f}% stage1_ensemble_epochs={ensemble_epochs} skip_stage2=True",
            flush=True,
        )
        out_frames = [df for df in (metric_rows + [base_valid, base_test]) if len(df)]
        return pd.concat(out_frames, ignore_index=True), pd.DataFrame(history), selector_df

    stage2_variant = CondInvVariant(
        f"condinv_trainDST25_selected_seed{seed}_ep{selected_epoch}_stage2_0_45_corr",
        lambda_condinv=0.02,
        enable_stage2=True,
        corr_limit=float(cfg.corr_limit),
        corr_mode=str(cfg.corr_mode),
    )
    cond_cfg = CondInvConfig(
        base_dir=cfg.base_dir,
        output_prefix=f"{cfg.output_prefix}_seed{seed}_sel{selected_epoch}",
        hidden_size=cfg.hidden_size,
        window_len=cfg.window_len,
        stage2_epochs=cfg.stage2_epochs,
        lr_stage2=cfg.lr_stage2,
        weight_decay=cfg.weight_decay,
        huber_beta=cfg.huber_beta,
        focus45_weight=cfg.focus45_weight,
        keep_lambda=cfg.keep_lambda,
        eval_every=cfg.eval_every,
    )
    if bool(cfg.test_blind):
        stage2_rows = _train_stage2_correction_test_blind(
            cfg,
            stage2_variant,
            selected_model,
            stage2_train_loader,
            stage2_valid_loader,
            stage2_test_loader,
            int(seed),
            int(selected_epoch),
        )
    else:
        stage2_rows = train_stage2_correction(cond_cfg, stage2_variant, selected_model, stage2_train_loader, stage2_valid_loader, stage2_test_loader)
    if len(stage2_rows):
        stage2_rows["seed"] = int(seed)
        stage2_rows["selector"] = str(cfg.stage1_selector)
        stage2_rows["selector_min_epoch"] = int(cfg.selector_min_epoch)
        stage2_rows["selector_max_epoch"] = int(cfg.selector_max_epoch)
        stage2_rows["selected_epoch"] = selected_epoch
        stage2_rows["selector_score"] = selected_score
        stage2_rows["stage1_ensemble_epochs"] = ",".join(str(ep) for ep in ensemble_epochs)
    print(
        f"seed={seed} selected_epoch={selected_epoch} selector_max_epoch={cfg.selector_max_epoch} "
        f"selector_score={selected_score:.3f}% stage1_ensemble_epochs={ensemble_epochs}",
        flush=True,
    )
    out_frames = [df for df in (metric_rows + [base_valid, base_test, stage2_rows]) if len(df)]
    return pd.concat(out_frames, ignore_index=True), pd.DataFrame(history), selector_df


def run(cfg: TrainDSTSelectorConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.base_dir = Path(cfg.base_dir).resolve()
    if not Path(cfg.raw_root).is_absolute():
        cfg.raw_root = cfg.base_dir / cfg.raw_root
    out_dir = cfg.base_dir / "nmc_goal_vcorr_it_train_dst_selector_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    files = find_csv_files(cfg.raw_root)
    raw_source_columns = list(pd.read_csv(files[0], nrows=1).columns)
    write_start_audit(files, out_dir / f"{cfg.output_prefix}_file_start_audit.csv")
    r0_df = estimate_r0_by_temperature(files, cfg.train_profiles)
    r0_df.to_csv(out_dir / f"{cfg.output_prefix}_decomposition_params.csv", index=False)
    frames = add_vit_engineered_features(build_feature_frames(cfg, files, r0_df))
    frames = _split_train_profile_blocks_for_validation(
        frames,
        cfg,
        out_dir / f"{cfg.output_prefix}_internal_valid_split_audit.csv",
    )
    feature_cols = _selected_feature_columns(str(cfg.feature_set))
    stage2_feature_set = str(cfg.stage2_feature_set or cfg.feature_set)
    stage2_feature_cols = _selected_feature_columns(stage2_feature_set)
    available = set().union(*(set(frame.columns) for split_frames in frames.values() for frame in split_frames))
    missing = [col for col in feature_cols if col not in available]
    if missing:
        raise RuntimeError(f"Selected feature_set={cfg.feature_set!r} has missing columns: {missing}")
    stage2_missing = [col for col in stage2_feature_cols if col not in available]
    if stage2_missing:
        raise RuntimeError(f"Selected stage2_feature_set={stage2_feature_set!r} has missing columns: {stage2_missing}")
    write_input_schema(feature_cols, out_dir / f"{cfg.output_prefix}_input_schema.csv")
    write_leakage_audit(feature_cols, raw_source_columns, out_dir / f"{cfg.output_prefix}_leakage_audit.csv")
    if stage2_feature_cols != feature_cols:
        write_input_schema(stage2_feature_cols, out_dir / f"{cfg.output_prefix}_stage2_input_schema.csv")
        write_leakage_audit(stage2_feature_cols, raw_source_columns, out_dir / f"{cfg.output_prefix}_stage2_leakage_audit.csv")
    all_metrics = []
    all_history = []
    all_selector = []
    for seed in cfg.seeds:
        metrics, history, selector = _train_select_and_correct(cfg, frames, out_dir, int(seed))
        all_metrics.append(metrics)
        all_history.append(history)
        all_selector.append(selector)
    metrics = pd.concat(all_metrics, ignore_index=True)
    history = pd.concat(all_history, ignore_index=True)
    selector = pd.concat(all_selector, ignore_index=True)
    metrics.to_csv(out_dir / f"{cfg.output_prefix}_by_temperature.csv", index=False)
    history.to_csv(out_dir / f"{cfg.output_prefix}_history.csv", index=False)
    selector.to_csv(out_dir / f"{cfg.output_prefix}_selector_trace.csv", index=False)
    test = metrics[metrics["split"].eq("test")].copy()
    temp_piv = test.pivot_table(index=["seed", "variant", "epoch"], columns="temperature_C", values="MAE_pct", aggfunc="mean").reset_index()
    for c in [0.0, 25.0, 45.0]:
        if c not in temp_piv.columns:
            temp_piv[c] = np.nan
    temp_piv["max_target"] = temp_piv[[0.0, 25.0, 45.0]].max(axis=1)
    temp_piv["target_met"] = (temp_piv[0.0] < 1.0) & (temp_piv[25.0] < 0.7) & (temp_piv[45.0] < 0.3)
    temp_piv.sort_values(["seed", "target_met", "max_target"], ascending=[True, False, True]).to_csv(
        out_dir / f"{cfg.output_prefix}_test_summary.csv",
        index=False,
    )
    if str(cfg.stage2_select_rule) != "none":
        selected = _stage2_rule_selection(cfg, metrics)
        selected.to_csv(out_dir / f"{cfg.output_prefix}_stage2_rule_selected_by_temperature.csv", index=False)
        selected_test = selected[selected["split"].eq("test")].copy()
        selected_piv = selected_test.pivot_table(
            index=["seed", "variant", "selected_epoch", "stage2_selected_epoch", "stage2_select_rule"],
            columns="temperature_C",
            values="MAE_pct",
            aggfunc="mean",
        ).reset_index()
        for c in [0.0, 25.0, 45.0]:
            if c not in selected_piv.columns:
                selected_piv[c] = np.nan
        selected_piv["max_target"] = selected_piv[[0.0, 25.0, 45.0]].max(axis=1)
        selected_piv["target_met"] = (selected_piv[0.0] < 1.0) & (selected_piv[25.0] < 0.7) & (selected_piv[45.0] < 0.3)
        selected_piv.to_csv(out_dir / f"{cfg.output_prefix}_stage2_rule_selected_test_summary.csv", index=False)
    (out_dir / f"{cfg.output_prefix}_metadata.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "feature_columns": feature_cols,
                "input_feature_dim": len(feature_cols),
                "stage2_feature_set_effective": stage2_feature_set,
                "stage2_feature_columns": stage2_feature_cols,
                "stage2_input_feature_dim": len(stage2_feature_cols),
                "voltage_shape_ssl_target_candidates": SSL_VOLTAGE_TARGET_CANDIDATES,
                "voltage_shape_ssl_uses_soc_labels": False,
                "voltage_shape_ssl_uses_future_current_input": False,
                "selector": f"min {cfg.stage1_selector} within selector_max_epoch",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("Train-DST25 selector test summary:")
    print(temp_piv.sort_values(["seed", "target_met", "max_target"], ascending=[True, False, True]).head(60).to_string(index=False), flush=True)
    if str(cfg.stage2_select_rule) != "none":
        print("Stage2 rule-selected test summary:")
        print(selected_piv.to_string(index=False), flush=True)
    return metrics, history, selector


def _stage2_rule_selection(cfg: TrainDSTSelectorConfig, metrics: pd.DataFrame) -> pd.DataFrame:
    stage2 = metrics[metrics["variant"].astype(str).str.contains("_stage2", na=False)].copy()
    if not len(stage2):
        return pd.DataFrame()
    if str(cfg.stage2_select_rule) == "stage1_epoch_30_35":
        selected_epoch = pd.to_numeric(stage2["selected_epoch"], errors="coerce")
        stage2["stage2_selected_epoch"] = np.where(
            selected_epoch.le(int(cfg.stage2_stage1_threshold)),
            int(cfg.stage2_early_epoch),
            int(cfg.stage2_late_epoch),
        )
    elif str(cfg.stage2_select_rule).startswith("fixed"):
        fixed_raw = str(cfg.stage2_select_rule).replace("fixed", "")
        stage2["stage2_selected_epoch"] = int(fixed_raw)
    elif str(cfg.stage2_select_rule) in {"val_mean_mae", "val_worst_mae", "val_mean_plus_worst"}:
        key_cols = ["seed", "variant", "selected_epoch"]
        stage2 = stage2.drop(
            columns=[
                "stage2_selected_epoch",
                "stage2_selector_score",
                "val_mean_mae",
                "val_worst_mae",
            ],
            errors="ignore",
        )
        valid = stage2[stage2["split"].eq("valid")].copy()
        if valid.empty:
            raise RuntimeError("stage2_select_rule requires validation rows, but none were found.")
        per_epoch = (
            valid.groupby(key_cols + ["epoch"], as_index=False)["MAE_pct"]
            .agg(val_mean_mae="mean", val_worst_mae="max")
        )
        if str(cfg.stage2_select_rule) == "val_mean_plus_worst":
            per_epoch["stage2_selector_score"] = per_epoch["val_mean_mae"] + 0.5 * per_epoch["val_worst_mae"]
        else:
            per_epoch["stage2_selector_score"] = per_epoch[str(cfg.stage2_select_rule)]
        chosen = (
            per_epoch.sort_values(key_cols + ["stage2_selector_score", "epoch"])
            .groupby(key_cols, as_index=False)
            .first()
            .rename(columns={"epoch": "stage2_selected_epoch"})
        )
        stage2 = stage2.merge(
            chosen[key_cols + ["stage2_selected_epoch", "stage2_selector_score", "val_mean_mae", "val_worst_mae"]],
            on=key_cols,
            how="left",
            validate="many_to_one",
        )
    else:
        raise ValueError(f"Unknown stage2_select_rule={cfg.stage2_select_rule!r}")
    selected = stage2[stage2["epoch"].eq(stage2["stage2_selected_epoch"])].copy()
    selected["stage2_select_rule"] = str(cfg.stage2_select_rule)
    selected["stage2_stage1_threshold"] = int(cfg.stage2_stage1_threshold)
    selected["stage2_early_epoch"] = int(cfg.stage2_early_epoch)
    selected["stage2_late_epoch"] = int(cfg.stage2_late_epoch)
    return selected


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _parse_profiles(raw: str) -> tuple[str, ...]:
    profiles = tuple(str(x).strip().upper() for x in str(raw).split(",") if str(x).strip())
    if not profiles:
        raise ValueError("At least one profile is required.")
    return profiles


def _parse_temperatures(raw: str) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return ()
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _frame_temperature(frame: pd.DataFrame) -> float:
    if "temperature" not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame["temperature"], errors="coerce").iloc[0])


def _filter_frame_list_by_temperatures(
    frames: list[pd.DataFrame],
    temperatures: tuple[float, ...],
    *,
    split_name: str,
) -> list[pd.DataFrame]:
    if not temperatures:
        return list(frames)
    wanted = np.asarray([float(t) for t in temperatures], dtype=np.float64)
    out = [
        frame
        for frame in frames
        if np.isfinite(_frame_temperature(frame)) and np.isclose(_frame_temperature(frame), wanted, atol=0.75).any()
    ]
    if not out:
        raise RuntimeError(f"No frames left in split={split_name!r} after temperature filter {tuple(temperatures)}")
    return out


def _filter_scaled_frames_by_temperatures(
    scaled: dict[str, list[pd.DataFrame]],
    *,
    train_temperatures: tuple[float, ...],
    valid_temperatures: tuple[float, ...],
    test_temperatures: tuple[float, ...],
    name: str,
) -> dict[str, list[pd.DataFrame]]:
    filtered = {
        "train": _filter_frame_list_by_temperatures(
            scaled["train"],
            train_temperatures,
            split_name=f"{name}:train",
        ),
        "valid": _filter_frame_list_by_temperatures(
            scaled["valid"],
            valid_temperatures,
            split_name=f"{name}:valid",
        ),
        "test": _filter_frame_list_by_temperatures(
            scaled["test"],
            test_temperatures,
            split_name=f"{name}:test",
        ),
    }
    return filtered


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.astype(bool).tolist()):
        if flag and start is None:
            start = idx
        elif (not flag) and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(len(mask))))
    return runs


def _slice_segment(frame: pd.DataFrame, start: int, end: int, split_tag: str, segment_idx: int) -> pd.DataFrame:
    out = frame.iloc[int(start) : int(end)].copy().reset_index(drop=True)
    base_tid = str(frame["trajectory_id"].iloc[0]) if "trajectory_id" in frame.columns and len(frame) else "trajectory"
    out["source_trajectory_id"] = base_tid
    out["trajectory_id"] = f"{base_tid}__{split_tag}{segment_idx:03d}"
    out["split_segment_start"] = int(start)
    out["split_segment_end"] = int(end)
    out["split_segment_role"] = str(split_tag)
    return out


def _split_train_profile_blocks_for_validation(
    frames: dict[str, list[pd.DataFrame]],
    cfg: TrainDSTSelectorConfig,
    out_path: Path,
) -> dict[str, list[pd.DataFrame]]:
    mode = str(cfg.valid_split_mode)
    if mode == "profile":
        return frames
    if mode != "train_profile_blocks":
        raise ValueError(f"Unknown valid_split_mode={mode!r}")
    if int(cfg.valid_block_mod) < 2:
        raise ValueError("--valid-block-mod must be at least 2 for train_profile_blocks.")
    block_rows = int(cfg.valid_block_rows)
    if block_rows <= int(cfg.window_len):
        raise ValueError("--valid-block-rows must be larger than --window-len.")
    valid_mod = int(cfg.valid_block_mod)
    valid_index = int(cfg.valid_block_index) % valid_mod
    min_rows = int(cfg.window_len) + 1
    train_segments: list[pd.DataFrame] = []
    valid_segments: list[pd.DataFrame] = []
    audit_rows = []

    for frame_idx, frame in enumerate(frames["train"]):
        n = int(len(frame))
        if n < min_rows * valid_mod:
            raise RuntimeError(
                f"Training frame {frame_idx} is too short for block validation: n={n}, "
                f"window_len={cfg.window_len}, valid_block_mod={valid_mod}"
            )
        block_id = np.arange(n, dtype=np.int64) // block_rows
        valid_mask = (block_id % valid_mod) == valid_index
        source_tid = str(frame["trajectory_id"].iloc[0])
        temp = float(frame["temperature"].iloc[0]) if "temperature" in frame.columns else float("nan")
        drive = str(frame["drive_cycle"].iloc[0]) if "drive_cycle" in frame.columns else "unknown"
        for role, mask, target in [
            ("trainblk", ~valid_mask, train_segments),
            ("validblk", valid_mask, valid_segments),
        ]:
            for start, end in _contiguous_true_runs(mask):
                if int(end) - int(start) < min_rows:
                    continue
                segment = _slice_segment(frame, int(start), int(end), role, len(target))
                target.append(segment)
                audit_rows.append(
                    {
                        "source_trajectory_id": source_tid,
                        "segment_trajectory_id": str(segment["trajectory_id"].iloc[0]),
                        "split": "train" if role == "trainblk" else "valid",
                        "temperature_C": temp,
                        "drive_cycle": drive,
                        "start_row": int(start),
                        "end_row_exclusive": int(end),
                        "n_rows": int(end) - int(start),
                        "valid_block_rows": block_rows,
                        "valid_block_mod": valid_mod,
                        "valid_block_index": valid_index,
                    }
                )

    if not train_segments or not valid_segments:
        raise RuntimeError("Internal train-profile block validation produced an empty train or valid split.")
    train_keys = {
        (str(seg["source_trajectory_id"].iloc[0]), int(seg["end_index"].iloc[i]))
        for seg in train_segments
        for i in range(len(seg))
    }
    valid_keys = {
        (str(seg["source_trajectory_id"].iloc[0]), int(seg["end_index"].iloc[i]))
        for seg in valid_segments
        for i in range(len(seg))
    }
    overlap = train_keys & valid_keys
    if overlap:
        raise RuntimeError(f"Internal validation split leakage: {len(overlap)} overlapping source rows.")
    pd.DataFrame(audit_rows).to_csv(out_path, index=False)
    return {"train": train_segments, "valid": valid_segments, "test": list(frames["test"])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run train-only DST25 checkpoint selector with cold/hot correction.")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--raw-root", default=TrainDSTSelectorConfig.raw_root)
    p.add_argument("--output-prefix", default=BASE_PREFIX)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-profiles", default=",".join(TrainDSTSelectorConfig.train_profiles))
    p.add_argument("--valid-profiles", default=",".join(TrainDSTSelectorConfig.valid_profiles))
    p.add_argument("--test-profiles", default=",".join(TrainDSTSelectorConfig.test_profiles))
    p.add_argument(
        "--train-temperatures",
        default="all",
        help="Comma-separated Stage 1 train temperatures. Use 'all' for the default all-temperature train split.",
    )
    p.add_argument(
        "--valid-temperatures",
        default="all",
        help="Comma-separated Stage 1 validation temperatures. Use 'all' for default.",
    )
    p.add_argument(
        "--test-temperatures",
        default="all",
        help="Comma-separated Stage 1 test temperatures. Use 'all' for default.",
    )
    p.add_argument(
        "--finetune-temperatures",
        default="all",
        help="Comma-separated Stage 1 fine-tune train temperatures. Empty/all disables fine-tune unless --finetune-epochs is positive.",
    )
    p.add_argument("--finetune-epochs", type=int, default=0)
    p.add_argument("--finetune-lr-scale", type=float, default=0.35)
    p.add_argument(
        "--stage2-train-temperatures",
        default="all",
        help="Comma-separated Stage 2 correction train temperatures. Empty/all means follow Stage 1 train temperatures.",
    )
    p.add_argument(
        "--stage2-valid-temperatures",
        default="all",
        help="Comma-separated Stage 2 validation temperatures. Empty/all means follow Stage 1 validation temperatures.",
    )
    p.add_argument(
        "--stage2-test-temperatures",
        default="all",
        help="Comma-separated Stage 2 test temperatures. Empty/all means follow Stage 1 test temperatures.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--selector-min-epoch", type=int, default=1)
    p.add_argument("--selector-max-epoch", type=int, default=0)
    p.add_argument("--stage2-epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--stage1-eval-every", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--recurrent", choices=["tcn", "lstm", "gru"], default="tcn")
    p.add_argument("--head-kind", choices=["linear", "mlp"], default="linear")
    p.add_argument("--temp-mode", choices=["none", "bias", "moe", "hard_heads"], default="moe")
    p.add_argument("--dropout", type=float, default=0.06)
    p.add_argument("--loss-kind", choices=["huber", "mse"], default="huber")
    p.add_argument(
        "--model-kind",
        choices=[
            "single",
            "fusion_h64_h128",
            "fusion_h64_h128_fixed",
            "anchor_residual_tcn",
            "endpoint_mlp",
            "window_summary_mlp",
        ],
        default="single",
    )
    p.add_argument("--fusion-h64-weight", type=float, default=0.5)
    p.add_argument("--anchor-residual-limit", type=float, default=0.12)
    p.add_argument("--lambda-anchor-loss", type=float, default=0.2)
    p.add_argument("--ssl-pretrain-epochs", type=int, default=0)
    p.add_argument("--ssl-lr", type=float, default=8e-4)
    p.add_argument("--ssl-recon-weight", type=float, default=1.0)
    p.add_argument("--ssl-next-vcorr-weight", type=float, default=0.5)
    p.add_argument("--ssl-slope-weight", type=float, default=0.25)
    p.add_argument("--corr-mode", default="cold_hot")
    p.add_argument("--corr-limit", type=float, default=1.2)
    p.add_argument("--no-corr-zero-init", action="store_false", dest="corr_zero_init")
    p.set_defaults(corr_zero_init=True)
    p.add_argument("--lr-stage2", type=float, default=8e-4)
    p.add_argument("--focus45-weight", type=float, default=12.0)
    p.add_argument("--keep-lambda", type=float, default=4.0)
    p.add_argument("--lambda-rex", type=float, default=2.0)
    p.add_argument("--lambda-condinv", type=float, default=0.02)
    p.add_argument("--weight-0", type=float, default=4.0)
    p.add_argument("--weight-25", type=float, default=2.2)
    p.add_argument("--weight-45", type=float, default=1.0)
    p.add_argument(
        "--stage1-selector",
        choices=[
            "train25_dst",
            "train25_us06",
            "train25_mean_drive",
            "train25_worst_drive",
            "train25_worst_gap",
            "valid_worst_temp",
            "val_mean_mae",
            "val_worst_mae",
            "val_mean_plus_worst",
            "val_target_worst",
            "val_target_mean_plus_worst",
            "val_regime_target_worst",
            "val_regime_target_mean_plus_worst",
            "valid25_temp",
            "last_epoch",
            "train25_dst_valid_worst",
            "train25_worst_valid_worst",
            "train25_dst_valid45_guard",
            "train25_worst_valid_balance",
        ],
        default="train25_dst",
    )
    p.add_argument(
        "--fixed-stage1-epoch",
        type=int,
        default=0,
        help="Override metric-based Stage 1 selection with a predeclared epoch. Useful for inner-LOO-derived fixed horizons.",
    )
    p.add_argument(
        "--selector-regime-min-windows",
        type=int,
        default=30,
        help="Minimum validation windows required for a regime slice to contribute to val_regime_* selectors.",
    )
    p.add_argument("--stage2-select-rule", default="none")
    p.add_argument("--stage2-stage1-threshold", type=int, default=7)
    p.add_argument("--stage2-early-epoch", type=int, default=30)
    p.add_argument("--stage2-late-epoch", type=int, default=35)
    p.add_argument("--test-blind", action="store_true")
    p.add_argument(
        "--train-sampler",
        choices=[
            "temperature_balanced",
            "standard",
            "temperature_profile_balanced",
            "temperature_profile_soc_balanced",
            "temperature_regime_balanced",
            "temperature_profile_regime_balanced",
        ],
        default="temperature_balanced",
    )
    p.add_argument(
        "--feature-set",
        choices=[
            "vcorr_it",
            "vcorr_it_excitation_ema",
            "paper_g0_raw",
            "paper_g1_derivatives",
            "paper_g4_all_ema",
            "paper_t6_voltage_ema_all",
            "paper_t7_current_abs_ema_all",
            "paper_voltage_ema50_only",
            "paper_voltage_ema200_only",
            "paper_voltage_ema800_only",
            "paper_current_abs_ema50_only",
            "paper_current_abs_ema200_only",
            "paper_g6_full23",
            "paper_g7_no_current_ema",
            "paper_g8_no_voltage_ema",
            "vit_basic",
            "vit_decomp",
            "vit_ema_compact",
            "vit_response_ema",
            "vit_full_ema",
        ],
        default="vcorr_it",
    )
    p.add_argument(
        "--stage2-feature-set",
        choices=[
            "",
            "vcorr_it",
            "vcorr_it_excitation_ema",
            "paper_g0_raw",
            "paper_g1_derivatives",
            "paper_g4_all_ema",
            "paper_t6_voltage_ema_all",
            "paper_t7_current_abs_ema_all",
            "paper_voltage_ema50_only",
            "paper_voltage_ema200_only",
            "paper_voltage_ema800_only",
            "paper_current_abs_ema50_only",
            "paper_current_abs_ema200_only",
            "paper_g6_full23",
            "paper_g7_no_current_ema",
            "paper_g8_no_voltage_ema",
            "vit_basic",
            "vit_decomp",
            "vit_ema_compact",
            "vit_response_ema",
            "vit_full_ema",
        ],
        default="",
        help="Optional correction-only feature set. Empty means reuse --feature-set.",
    )
    p.add_argument(
        "--stage1-ensemble-epochs",
        default="",
        help="Optional comma list or range:start:end of stage1 checkpoint epochs to average before stage2.",
    )
    p.add_argument(
        "--sampler-seed-mode",
        choices=["seed", "none"],
        default="seed",
        help="Use an independent per-seed sampler generator, or reproduce the old global-RNG sampler path.",
    )
    p.add_argument(
        "--test-blind-rng-burn",
        type=int,
        default=0,
        help="When test-blind, advance global RNG after validation to mimic old non-blind test-loader RNG side effects without reading test data.",
    )
    p.add_argument(
        "--diagnostic-test-every",
        type=int,
        default=0,
        help="When test-blind, still export diagnostic test metrics every N Stage 1 epochs. Use only for screening, not paper selection.",
    )
    p.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Train/evaluate Stage 1 only and skip the bounded correction stage.",
    )
    p.add_argument("--lambda-current-consistency", type=float, default=0.0)
    p.add_argument("--current-noise-std", type=float, default=0.0)
    p.add_argument("--current-dropout-prob", type=float, default=0.0)
    p.add_argument("--lambda-profile-adv", type=float, default=0.0)
    p.add_argument("--profile-adv-grl", type=float, default=1.0)
    p.add_argument("--lambda-profile-supcon", type=float, default=0.0)
    p.add_argument("--supcon-temperature", type=float, default=0.15)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument(
        "--ema-perturbation-importance",
        action="store_true",
        help="After selecting the fixed Stage 1 model, export inference-only EMA perturbation metrics.",
    )
    p.add_argument("--valid-split-mode", default="profile", choices=["profile", "train_profile_blocks"])
    p.add_argument("--valid-block-rows", type=int, default=800)
    p.add_argument("--valid-block-mod", type=int, default=5)
    p.add_argument("--valid-block-index", type=int, default=4)
    p.add_argument("--sequence-training", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainDSTSelectorConfig(
        base_dir=Path(args.base_dir),
        raw_root=Path(args.raw_root),
        output_prefix=str(args.output_prefix),
        seeds=_parse_seeds(args.seeds),
        train_profiles=_parse_profiles(args.train_profiles),
        valid_profiles=_parse_profiles(args.valid_profiles),
        test_profiles=_parse_profiles(args.test_profiles),
        train_temperatures=_parse_temperatures(args.train_temperatures),
        valid_temperatures=_parse_temperatures(args.valid_temperatures),
        test_temperatures=_parse_temperatures(args.test_temperatures),
        finetune_temperatures=_parse_temperatures(args.finetune_temperatures),
        finetune_epochs=int(args.finetune_epochs),
        finetune_lr_scale=float(args.finetune_lr_scale),
        stage2_train_temperatures=_parse_temperatures(args.stage2_train_temperatures),
        stage2_valid_temperatures=_parse_temperatures(args.stage2_valid_temperatures),
        stage2_test_temperatures=_parse_temperatures(args.stage2_test_temperatures),
        epochs=int(args.epochs),
        selector_min_epoch=int(args.selector_min_epoch),
        selector_max_epoch=int(args.selector_max_epoch),
        stage2_epochs=int(args.stage2_epochs),
        eval_every=int(args.eval_every),
        stage1_eval_every=int(args.stage1_eval_every),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        hidden_size=int(args.hidden_size),
        layers=int(args.layers),
        kernel_size=int(args.kernel_size),
        recurrent=str(args.recurrent),
        head_kind=str(args.head_kind),
        temp_mode=str(args.temp_mode),
        dropout=float(args.dropout),
        loss_kind=str(args.loss_kind),
        model_kind=str(args.model_kind),
        fusion_h64_weight=float(args.fusion_h64_weight),
        corr_mode=str(args.corr_mode),
        corr_limit=float(args.corr_limit),
        corr_zero_init=bool(args.corr_zero_init),
        lr_stage2=float(args.lr_stage2),
        focus45_weight=float(args.focus45_weight),
        keep_lambda=float(args.keep_lambda),
        lambda_rex=float(args.lambda_rex),
        lambda_condinv=float(args.lambda_condinv),
        weight_0=float(args.weight_0),
        weight_25=float(args.weight_25),
        weight_45=float(args.weight_45),
        stage1_selector=str(args.stage1_selector),
        selector_regime_min_windows=int(args.selector_regime_min_windows),
        fixed_stage1_epoch=int(args.fixed_stage1_epoch),
        stage2_select_rule=str(args.stage2_select_rule),
        stage2_stage1_threshold=int(args.stage2_stage1_threshold),
        stage2_early_epoch=int(args.stage2_early_epoch),
        stage2_late_epoch=int(args.stage2_late_epoch),
        test_blind=bool(args.test_blind),
        train_sampler=str(args.train_sampler),
        stage1_ensemble_epochs=str(args.stage1_ensemble_epochs),
        feature_set=str(args.feature_set),
        stage2_feature_set=str(args.stage2_feature_set),
        sampler_seed_mode=str(args.sampler_seed_mode),
        test_blind_rng_burn=int(args.test_blind_rng_burn),
        diagnostic_test_every=int(args.diagnostic_test_every),
        skip_stage2=bool(args.skip_stage2),
        lambda_current_consistency=float(args.lambda_current_consistency),
        current_noise_std=float(args.current_noise_std),
        current_dropout_prob=float(args.current_dropout_prob),
        lambda_profile_adv=float(args.lambda_profile_adv),
        profile_adv_grl=float(args.profile_adv_grl),
        lambda_profile_supcon=float(args.lambda_profile_supcon),
        supcon_temperature=float(args.supcon_temperature),
        anchor_residual_limit=float(args.anchor_residual_limit),
        lambda_anchor_loss=float(args.lambda_anchor_loss),
        ssl_pretrain_epochs=int(args.ssl_pretrain_epochs),
        ssl_lr=float(args.ssl_lr),
        ssl_recon_weight=float(args.ssl_recon_weight),
        ssl_next_vcorr_weight=float(args.ssl_next_vcorr_weight),
        ssl_slope_weight=float(args.ssl_slope_weight),
        valid_split_mode=str(args.valid_split_mode),
        valid_block_rows=int(args.valid_block_rows),
        valid_block_mod=int(args.valid_block_mod),
        valid_block_index=int(args.valid_block_index),
        save_predictions=bool(args.save_predictions),
        ema_perturbation_importance=bool(args.ema_perturbation_importance),
        sequence_training=bool(args.sequence_training),
    )
    run(cfg)


if __name__ == "__main__":
    main()
