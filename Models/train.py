from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from feature_sets import FEATURE_SETS, add_derived_features


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "Data"))
from prepare_calce_nmc import add_features, estimate_r0  # noqa: E402


def _norm_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_col(columns: list[object], aliases: list[str]) -> str | None:
    normalized = {_norm_col(c): str(c) for c in columns}
    for alias in aliases:
        hit = normalized.get(_norm_col(alias))
        if hit is not None:
            return hit
    return None


def _infer_temp_profile(path: Path, frame: pd.DataFrame) -> tuple[float, str]:
    temp_col = _find_col(list(frame.columns), ["temperature_C", "TempLabel", "T"])
    profile_col = _find_col(list(frame.columns), ["profile", "Profile", "drive_cycle"])
    if temp_col is not None:
        raw_temp = str(frame[temp_col].dropna().iloc[0]) if frame[temp_col].notna().any() else ""
        match = re.search(r"-?\d+(?:\.\d+)?", raw_temp)
        temp = float(match.group(0)) if match else np.nan
    else:
        match = re.search(r"(-?\d+(?:\.\d+)?)C", path.name, flags=re.IGNORECASE)
        temp = float(match.group(1)) if match else np.nan
    if profile_col is not None and frame[profile_col].notna().any():
        profile = str(frame[profile_col].dropna().iloc[0]).upper()
    else:
        profile = "UNKNOWN"
        for candidate in ("BJDST", "DST", "US06", "FUDS"):
            if candidate in path.name.upper():
                profile = candidate
                break
    return temp, profile


def _coerce_processed_frame(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if {"time_s", "V_raw", "I_raw", "SOC_percent", "profile", "temperature_C", "source_file"}.issubset(frame.columns):
        out = frame.copy()
    else:
        cols = list(frame.columns)
        time_col = _find_col(cols, ["time_s", "t_global(s)", "Test_Time(s)", "Step_Time(s)", "Time"])
        voltage_col = _find_col(cols, ["V_raw", "Voltage(V)", "Voltage", "V"])
        current_col = _find_col(cols, ["I_raw", "Current(A)", "Current", "I"])
        soc_pct_col = _find_col(cols, ["SOC_percent", "SOC_CC(%)", "SOC(%)"])
        soc_frac_col = _find_col(cols, ["SOC_CC", "SOC_fraction", "SOC"])
        missing = [name for name, col in {"time": time_col, "voltage": voltage_col, "current": current_col}.items() if col is None]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        temp, profile = _infer_temp_profile(path, frame)
        if soc_pct_col is not None:
            soc_percent = pd.to_numeric(frame[soc_pct_col], errors="coerce")
        elif soc_frac_col is not None:
            soc = pd.to_numeric(frame[soc_frac_col], errors="coerce")
            soc_percent = soc * 100.0 if soc.max(skipna=True) <= 1.5 else soc
        else:
            raise ValueError(f"{path} missing SOC label column")
        out = pd.DataFrame(
            {
                "time_s": pd.to_numeric(frame[time_col], errors="coerce"),
                "V_raw": pd.to_numeric(frame[voltage_col], errors="coerce"),
                "I_raw": pd.to_numeric(frame[current_col], errors="coerce"),
                "T": temp,
                "SOC_percent": soc_percent,
                "profile": profile,
                "temperature_C": temp,
                "source_file": path.name,
            }
        )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "V_raw", "I_raw", "SOC_percent"]).reset_index(drop=True)
    out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])
    if "V_corr_raw" not in out.columns:
        est = estimate_r0(out)
        out = add_features(out, est.r0_ohm)
    return add_derived_features(out)


def load_processed(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.rglob("NMC_*C_*.csv")):
        frame = pd.read_csv(path)
        frames.append(_coerce_processed_frame(path, frame))
    if not frames:
        raise FileNotFoundError(f"No processed CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def build_windows(frame: pd.DataFrame, features: list[str], window: int, stride: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, meta = [], [], []
    for _, g in frame.groupby(["temperature_C", "profile", "source_file"], sort=False):
        g = g.reset_index(drop=True)
        arr = g[features].to_numpy(np.float32)
        y = g["SOC_percent"].to_numpy(np.float32)
        for end in range(window - 1, len(g), stride):
            xs.append(arr[end - window + 1 : end + 1])
            ys.append(y[end])
            meta.append(
                {
                    "temperature_C": float(g["temperature_C"].iloc[end]),
                    "profile": str(g["profile"].iloc[end]),
                    "source_file": str(g["source_file"].iloc[end]),
                    "endpoint_index": int(end),
                    "time_s": float(g["time_s"].iloc[end]),
                }
            )
    if not xs:
        raise ValueError("No windows were generated. Check window length, stride, and processed data length.")
    return np.stack(xs), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta)


def split_train_test(frame: pd.DataFrame, test_profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame[frame["profile"].str.upper() == test_profile.upper()].copy()
    train = frame[frame["profile"].str.upper() != test_profile.upper()].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split is empty. Check profile names.")
    return train, test


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean.ravel(), std.ravel()


def default_run_id(args: argparse.Namespace) -> str:
    return f"{args.model}_{args.feature_set}_holdout-{args.test_profile}_seed{args.seed}"


def metric_summary(prediction_frame: pd.DataFrame) -> dict[str, float]:
    err = prediction_frame["y_pred_percent"].to_numpy(float) - prediction_frame["y_true_percent"].to_numpy(float)
    return {
        "MAE_percent": float(np.mean(np.abs(err))),
        "RMSE_percent": float(np.sqrt(np.mean(err**2))),
        "MaxAE_percent": float(np.max(np.abs(err))),
        "n_windows": int(len(prediction_frame)),
    }


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from model_zoo import make_model
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                "PyTorch is required for model training. Install the build matching your CPU/GPU environment from "
                "https://pytorch.org/get-started/locally/."
            ) from exc
        raise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    run_id = args.run_id or default_run_id(args)
    out_dir = Path(args.out_dir).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_processed(data_dir)
    features = FEATURE_SETS[args.feature_set]
    train_frame, test_frame = split_train_test(frame, args.test_profile)
    train_x, train_y, train_meta = build_windows(train_frame, features, args.window, args.stride)
    test_x, test_y, test_meta = build_windows(test_frame, features, args.window, args.test_stride)
    train_x, test_x, scaler_mean, scaler_std = standardize(train_x, test_x)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y / 100.0)), batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = make_model(args.model, input_dim=len(features), hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=args.huber_delta)

    history = []
    final_pred = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.report_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(test_x, device=device)).detach().cpu().numpy() * 100.0
            final_pred = pred
            err = pred - test_y
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "test_MAE_percent": float(np.mean(np.abs(err))),
                "test_RMSE_percent": float(np.sqrt(np.mean(err**2))),
                "test_MaxAE_percent": float(np.max(np.abs(err))),
            }
            history.append(row)
            print(
                f"epoch={epoch:04d} train_loss={row['train_loss']:.6f} "
                f"test_MAE={row['test_MAE_percent']:.4f} test_RMSE={row['test_RMSE_percent']:.4f}"
            )

    if final_pred is None:
        model.eval()
        with torch.no_grad():
            final_pred = model(torch.tensor(test_x, device=device)).detach().cpu().numpy() * 100.0

    pred_frame = test_meta.copy()
    pred_frame["y_true_percent"] = test_y
    pred_frame["y_pred_percent"] = final_pred
    pred_frame["abs_error_percent"] = np.abs(pred_frame["y_pred_percent"] - pred_frame["y_true_percent"])
    pred_frame.to_csv(out_dir / "test_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(out_dir / "metrics_history.csv", index=False)
    pd.DataFrame([metric_summary(pred_frame)]).to_csv(out_dir / "summary_metrics.csv", index=False)
    by_temp = (
        pred_frame.groupby("temperature_C", as_index=False)
        .agg(
            MAE_percent=("abs_error_percent", "mean"),
            RMSE_percent=("abs_error_percent", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            MaxAE_percent=("abs_error_percent", "max"),
            n_windows=("abs_error_percent", "size"),
        )
        .sort_values("temperature_C")
    )
    by_temp.to_csv(out_dir / "by_temperature.csv", index=False)
    pd.DataFrame({"feature_index": range(len(features)), "feature": features}).to_csv(out_dir / "input_schema.csv", index=False)
    pd.DataFrame({"feature": features, "train_mean": scaler_mean, "train_std": scaler_std}).to_csv(out_dir / "scaler_stats.csv", index=False)
    config = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir.as_posix(),
        "out_dir": out_dir.as_posix(),
        "model": args.model,
        "feature_set": args.feature_set,
        "features": features,
        "test_profile": args.test_profile,
        "window": args.window,
        "train_stride": args.stride,
        "test_stride": args.test_stride,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "huber_delta": args.huber_delta,
        "seed": args.seed,
        "device": str(device),
        "train_windows": int(len(train_meta)),
        "test_windows": int(len(test_meta)),
        "checkpoint_saved": bool(args.save_checkpoint),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_dir / "model.pt")

    print()
    print(f"Wrote run outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CEMA-TCN / baseline model on processed NMC data.")
    parser.add_argument("--data-dir", default=(ROOT / "Data" / "processed").as_posix())
    parser.add_argument("--out-dir", default=(ROOT / "Results" / "model_runs").as_posix())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", choices=["cema_tcn", "tcn", "lstm", "gru", "transformer", "mlp"], default="cema_tcn")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="G4")
    parser.add_argument("--test-profile", default="FUDS")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save model.pt under the local run directory. Checkpoints are ignored by Git.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
