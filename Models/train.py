from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_sets import FEATURE_SETS, add_derived_features
from model_zoo import make_model


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


def build_windows(frame: pd.DataFrame, features: list[str], window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, g in frame.groupby(["temperature_C", "profile", "source_file"], sort=False):
        g = g.reset_index(drop=True)
        arr = g[features].to_numpy(np.float32)
        y = g["SOC_percent"].to_numpy(np.float32)
        for end in range(window - 1, len(g), stride):
            xs.append(arr[end - window + 1 : end + 1])
            ys.append(y[end])
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def split_train_test(frame: pd.DataFrame, test_profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame[frame["profile"].str.upper() == test_profile.upper()].copy()
    train = frame[frame["profile"].str.upper() != test_profile.upper()].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split is empty. Check profile names.")
    return train, test


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame = load_processed(Path(args.data_dir))
    features = FEATURE_SETS[args.feature_set]
    train_frame, test_frame = split_train_test(frame, args.test_profile)
    train_x, train_y = build_windows(train_frame, features, args.window, args.stride)
    test_x, test_y = build_windows(test_frame, features, args.window, args.stride)
    train_x, test_x = standardize(train_x, test_x)

    train_loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y / 100.0)), batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = make_model(args.model, input_dim=len(features), hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=args.huber_delta)

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
            err = pred - test_y
            print(f"epoch={epoch:04d} train_loss={np.mean(losses):.6f} test_MAE={np.mean(np.abs(err)):.4f} test_RMSE={np.sqrt(np.mean(err**2)):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CEMA-TCN / baseline model on processed NMC data.")
    parser.add_argument("--data-dir", default=(ROOT / "Data" / "processed").as_posix())
    parser.add_argument("--model", choices=["cema_tcn", "tcn", "lstm", "gru", "transformer", "mlp"], default="cema_tcn")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="G4")
    parser.add_argument("--test-profile", default="FUDS")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=3)
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
