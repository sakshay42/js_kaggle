from pathlib import Path
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
MODEL_DIR = PROJECT_ROOT / "models" / "neural" / "regularized_mlp_sweep"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALID_PATH = DATA_DIR / "valid.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
REPORT_PATH = LOG_DIR / "train_regularized_mlp_sweep_report.json"
HISTORY_PATH = MODEL_DIR / "history.csv"
SUMMARY_PATH = MODEL_DIR / "summary.csv"
PRED_PATH = MODEL_DIR / "predictions.parquet"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
EMBED_COLS = ["symbol_id", "time_id"]
SEED = 42
BATCH_SIZE = 8192
EPOCHS = 8
PATIENCE = 2
SHRINK_GRID = [0.25, 0.50, 0.75, 1.00]


CONFIGS = [
    {
        "name": "tiny_mse",
        "hidden": [64, 32],
        "dropout": 0.20,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "loss": "mse",
        "use_embeddings": True,
    },
    {
        "name": "small_mse",
        "hidden": [128, 64],
        "dropout": 0.25,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "loss": "mse",
        "use_embeddings": True,
    },
    {
        "name": "small_huber",
        "hidden": [128, 64],
        "dropout": 0.30,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "loss": "huber",
        "huber_delta": 0.50,
        "use_embeddings": True,
    },
    {
        "name": "small_no_embed",
        "hidden": [128, 64],
        "dropout": 0.25,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "loss": "mse",
        "use_embeddings": False,
    },
    {
        "name": "linear_strong_decay",
        "hidden": [],
        "dropout": 0.00,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "loss": "huber",
        "huber_delta": 0.50,
        "use_embeddings": True,
    },
]


class TabularDataset(Dataset):
    def __init__(self, df, numeric_cols, cat_maps, scaler, use_embeddings):
        self.y = df[TARGET_COL].to_numpy(np.float32)
        self.w = df[WEIGHT_COL].to_numpy(np.float32)
        X = df[numeric_cols].to_numpy(np.float32)
        X = (X - scaler["mean"]) / scaler["std"]
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if use_embeddings:
            cats = []
            for col in EMBED_COLS:
                mapping = cat_maps[col]
                cats.append(np.array([mapping.get(int(v), 0) for v in df[col].to_numpy()], dtype=np.int64))
            self.cats = np.stack(cats, axis=1)
        else:
            self.cats = np.zeros((len(df), 0), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.cats[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.w[idx]),
        )


class RegularizedMLP(nn.Module):
    def __init__(self, n_num, cat_cardinalities, hidden, dropout, use_embeddings):
        super().__init__()
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, min(16, max(4, cardinality // 2))) for cardinality in cat_cardinalities]
            )
            emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        else:
            self.embeddings = nn.ModuleList()
            emb_dim = 0

        layers = []
        in_dim = n_num + emb_dim
        for out_dim in hidden:
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        if self.use_embeddings:
            emb = [layer(x_cat[:, i]) for i, layer in enumerate(self.embeddings)]
            x = torch.cat([x_num, *emb], dim=1)
        else:
            x = x_num
        return self.net(x).squeeze(1)


def weighted_loss(pred, y, w, config):
    if config["loss"] == "huber":
        delta = config.get("huber_delta", 1.0)
        abs_err = torch.abs(pred - y)
        loss = torch.where(abs_err <= delta, 0.5 * abs_err.pow(2), delta * (abs_err - 0.5 * delta))
    else:
        loss = (pred - y).pow(2)
    return torch.sum(w * loss) / torch.clamp(torch.sum(w), min=1e-8)


def weighted_r2(y_true, y_pred, weight):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denom = np.sum(weight * np.square(y_true))
    return np.nan if denom == 0 else 1.0 - np.sum(weight * np.square(y_true - y_pred)) / denom


def load_df(path, name):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"{name}_shape={df.shape}", flush=True)
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in [DATE_COL, WEIGHT_COL, TARGET_COL, "_split"]]


def fit_scaler(df, numeric_cols):
    X = df[numeric_cols].to_numpy(np.float32)
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def make_cat_maps(train_df):
    maps = {}
    sizes = []
    for col in EMBED_COLS:
        values = sorted(int(v) for v in train_df[col].dropna().unique())
        maps[col] = {v: i + 1 for i, v in enumerate(values)}
        sizes.append(len(values) + 1)
    return maps, sizes


def make_loader(df, numeric_cols, cat_maps, scaler, config, batch_size, shuffle):
    ds = TabularDataset(df, numeric_cols, cat_maps, scaler, config["use_embeddings"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys, ws = [], [], []
    for x_num, x_cat, y, w in loader:
        pred = model(x_num.to(device), x_cat.to(device)).detach().cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
        ws.append(w.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)


def train_epoch(model, loader, optimizer, device, config):
    model.train()
    total = 0.0
    for x_num, x_cat, y, w in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_loss(model(x_num, x_cat), y, w, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)


def choose_shrink(y, pred, w):
    rows = []
    for shrink in SHRINK_GRID:
        score = weighted_r2(y, shrink * pred, w)
        rows.append({"shrink": shrink, "valid_weighted_r2": float(score)})
    return max(rows, key=lambda x: x["valid_weighted_r2"]), rows


def train_config(config, train_df, valid_df, test_df, numeric_cols, cat_maps, cat_sizes, scaler, device):
    print(f"training_config={config}", flush=True)
    train_loader = make_loader(train_df, numeric_cols, cat_maps, scaler, config, BATCH_SIZE, True)
    valid_loader = make_loader(valid_df, numeric_cols, cat_maps, scaler, config, BATCH_SIZE * 2, False)
    test_loader = make_loader(test_df, numeric_cols, cat_maps, scaler, config, BATCH_SIZE * 2, False)

    model = RegularizedMLP(
        n_num=len(numeric_cols),
        cat_cardinalities=cat_sizes,
        hidden=config["hidden"],
        dropout=config["dropout"],
        use_embeddings=config["use_embeddings"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    history = []
    best_score = -np.inf
    best_state = None
    best_epoch = None
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        valid_pred, valid_y, valid_w = predict(model, valid_loader, device)
        no_shrink_score = weighted_r2(valid_y, valid_pred, valid_w)
        best_shrink, shrink_rows = choose_shrink(valid_y, valid_pred, valid_w)
        row = {
            "model": config["name"],
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_weighted_r2": float(no_shrink_score),
            "best_shrink": float(best_shrink["shrink"]),
            "best_shrunk_valid_weighted_r2": float(best_shrink["valid_weighted_r2"]),
        }
        history.append(row)
        print(row, flush=True)
        if best_shrink["valid_weighted_r2"] > best_score:
            best_score = best_shrink["valid_weighted_r2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    valid_pred, valid_y, valid_w = predict(model, valid_loader, device)
    test_pred, test_y, test_w = predict(model, test_loader, device)
    best_shrink, _ = choose_shrink(valid_y, valid_pred, valid_w)
    shrink = best_shrink["shrink"]
    valid_score = weighted_r2(valid_y, shrink * valid_pred, valid_w)
    test_score = weighted_r2(test_y, shrink * test_pred, test_w)

    model_path = MODEL_DIR / f"{config['name']}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config,
            "numeric_cols": numeric_cols,
            "embed_cols": EMBED_COLS,
            "cat_maps": cat_maps,
            "cat_sizes": cat_sizes,
            "scaler": scaler,
            "best_epoch": best_epoch,
            "best_shrink": shrink,
            "best_valid_weighted_r2": float(valid_score),
        },
        model_path,
    )

    preds = pd.concat(
        [
            valid_df[[DATE_COL, "time_id", "symbol_id", TARGET_COL, WEIGHT_COL]].assign(
                split="valid", model=config["name"], prediction=shrink * valid_pred
            ),
            test_df[[DATE_COL, "time_id", "symbol_id", TARGET_COL, WEIGHT_COL]].assign(
                split="test", model=config["name"], prediction=shrink * test_pred
            ),
        ],
        ignore_index=True,
    )
    summary = {
        "model": config["name"],
        "valid_weighted_r2": float(valid_score),
        "test_weighted_r2": float(test_score),
        "best_epoch": best_epoch,
        "best_shrink": float(shrink),
        "model_path": str(model_path),
        **{k: v for k, v in config.items() if k != "hidden"},
        "hidden": str(config["hidden"]),
    }
    return summary, history, preds


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    start = time.time()

    train_df = load_df(TRAIN_PATH, "train")
    valid_df = load_df(VALID_PATH, "valid")
    test_df = load_df(TEST_PATH, "test")
    features = feature_cols(train_df)
    numeric_cols = [c for c in features if c not in EMBED_COLS]
    scaler = fit_scaler(train_df, numeric_cols)
    cat_maps, cat_sizes = make_cat_maps(train_df)

    summaries = []
    histories = []
    pred_frames = []
    for config in CONFIGS:
        summary, history, preds = train_config(
            config, train_df, valid_df, test_df, numeric_cols, cat_maps, cat_sizes, scaler, device
        )
        summaries.append(summary)
        histories.extend(history)
        pred_frames.append(preds)

    summary_df = pd.DataFrame(summaries).sort_values("valid_weighted_r2", ascending=False)
    history_df = pd.DataFrame(histories)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    history_df.to_csv(HISTORY_PATH, index=False)
    pred_df.to_parquet(PRED_PATH, index=False)

    report = {
        "data_dir": str(DATA_DIR),
        "summary_path": str(SUMMARY_PATH),
        "history_path": str(HISTORY_PATH),
        "prediction_path": str(PRED_PATH),
        "feature_count": len(features),
        "numeric_feature_count": len(numeric_cols),
        "best": summary_df.iloc[0].to_dict(),
        "results": summary_df.to_dict(orient="records"),
        "elapsed_seconds": float(time.time() - start),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print("DONE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
