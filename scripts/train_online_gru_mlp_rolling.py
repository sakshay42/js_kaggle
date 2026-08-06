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
MODEL_DIR = PROJECT_ROOT / "models" / "neural" / "online_rolling_gru_mlp"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALID_PATH = DATA_DIR / "valid.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
REPORT_PATH = LOG_DIR / "train_online_gru_mlp_rolling_report.json"
BEST_MODEL_PATH = MODEL_DIR / "best_online_gru_mlp.pt"
PRED_PATH = MODEL_DIR / "predictions.parquet"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
EMBED_COLS = ["symbol_id", "time_id"]
SEED = 42
BATCH_SIZE = 4096
EPOCHS = 5
LR = 8e-4
WEIGHT_DECAY = 1e-4
ONLINE_LR = 1e-5
ONLINE_EPOCHS_PER_DATE = 1
LOOKBACK = 16


class SequenceDataset(Dataset):
    def __init__(self, df, numeric_cols, cat_maps, scaler):
        df = df.sort_values(["symbol_id", DATE_COL, "time_id"]).reset_index(drop=True)
        self.ids = df[[DATE_COL, "time_id", "symbol_id"]].copy()
        self.y = df[TARGET_COL].to_numpy(np.float32)
        self.w = df[WEIGHT_COL].to_numpy(np.float32)

        X = df[numeric_cols].to_numpy(np.float32)
        X = (X - scaler["mean"]) / scaler["std"]
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        cats = []
        for col in EMBED_COLS:
            mapping = cat_maps[col]
            cats.append(np.array([mapping.get(int(v), 0) for v in df[col].to_numpy()], dtype=np.int64))
        self.cats = np.stack(cats, axis=1)

        symbol = df["symbol_id"].to_numpy()
        self.starts = np.zeros(len(df), dtype=np.int64)
        last_start = 0
        for i in range(len(df)):
            if i > 0 and symbol[i] != symbol[i - 1]:
                last_start = i
            self.starts[i] = last_start

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        start = max(self.starts[idx], idx - LOOKBACK)
        hist = self.X[start:idx]
        seq = np.zeros((LOOKBACK, self.X.shape[1]), dtype=np.float32)
        if len(hist) > 0:
            seq[-len(hist) :] = hist
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(seq),
            torch.from_numpy(self.cats[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.w[idx]),
        )


class GRUMLP(nn.Module):
    def __init__(self, n_num, cat_cardinalities):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, min(32, max(4, cardinality // 2))) for cardinality in cat_cardinalities]
        )
        emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        self.gru = nn.GRU(input_size=n_num, hidden_size=64, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(n_num + 64 + emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
        )

    def forward(self, x_num, x_seq, x_cat):
        emb = [layer(x_cat[:, i]) for i, layer in enumerate(self.embeddings)]
        _, h = self.gru(x_seq)
        x = torch.cat([x_num, h[-1], *emb], dim=1)
        return self.head(x).squeeze(1)


def weighted_mse(pred, y, w):
    return torch.sum(w * (pred - y).pow(2)) / torch.clamp(torch.sum(w), min=1e-8)


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


def make_loader(df, numeric_cols, cat_maps, scaler, batch_size, shuffle):
    ds = SequenceDataset(df, numeric_cols, cat_maps, scaler)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    preds, ys, ws = [], [], []
    for x_num, x_seq, x_cat, y, w in loader:
        pred = model(x_num.to(device), x_seq.to(device), x_cat.to(device)).detach().cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
        ws.append(w.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x_num, x_seq, x_cat, y, w in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_seq = x_seq.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_mse(model(x_num, x_seq, x_cat), y, w)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)


def score_static(model, df, numeric_cols, cat_maps, scaler, device):
    _, loader = make_loader(df, numeric_cols, cat_maps, scaler, BATCH_SIZE * 2, False)
    pred, y, w = predict_model(model, loader, device)
    return weighted_r2(y, pred, w), pred


def online_predict_then_update(model, df, numeric_cols, cat_maps, scaler, device, split_name):
    rows = []
    date_scores = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=ONLINE_LR, weight_decay=WEIGHT_DECAY)

    for date_id in sorted(df[DATE_COL].unique()):
        day = df[df[DATE_COL] == date_id].copy()
        day_ds, pred_loader = make_loader(day, numeric_cols, cat_maps, scaler, BATCH_SIZE * 2, False)
        pred, y, w = predict_model(model, pred_loader, device)
        score = weighted_r2(y, pred, w)
        date_scores.append({"split": split_name, "date_id": int(date_id), "rows": int(len(day)), "weighted_r2": float(score)})
        rows.append(
            day_ds.ids.assign(
                split=split_name,
                responder_6=y,
                weight=w,
                online_gru_mlp_pred=pred,
            )
        )

        _, update_loader = make_loader(day, numeric_cols, cat_maps, scaler, BATCH_SIZE, True)
        for _ in range(ONLINE_EPOCHS_PER_DATE):
            train_epoch(model, update_loader, optimizer, device)
        print(date_scores[-1], flush=True)

    pred_df = pd.concat(rows, ignore_index=True)
    overall = weighted_r2(pred_df[TARGET_COL], pred_df["online_gru_mlp_pred"], pred_df[WEIGHT_COL])
    return pred_df, date_scores, overall


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

    _, train_loader = make_loader(train_df, numeric_cols, cat_maps, scaler, BATCH_SIZE, True)
    model = GRUMLP(len(numeric_cols), cat_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        static_valid_r2, _ = score_static(model, valid_df, numeric_cols, cat_maps, scaler, device)
        row = {"epoch": epoch, "train_loss": loss, "static_valid_weighted_r2": float(static_valid_r2)}
        history.append(row)
        print(row, flush=True)

    base_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "model_state_dict": base_state,
            "numeric_cols": numeric_cols,
            "embed_cols": EMBED_COLS,
            "cat_maps": cat_maps,
            "cat_sizes": cat_sizes,
            "scaler": scaler,
            "lookback": LOOKBACK,
        },
        BEST_MODEL_PATH,
    )

    model.load_state_dict(base_state)
    valid_pred_df, valid_date_scores, valid_online_r2 = online_predict_then_update(
        model, valid_df, numeric_cols, cat_maps, scaler, device, "valid"
    )
    test_pred_df, test_date_scores, test_online_r2 = online_predict_then_update(
        model, test_df, numeric_cols, cat_maps, scaler, device, "test"
    )

    pred_df = pd.concat([valid_pred_df, test_pred_df], ignore_index=True)
    pred_df.to_parquet(PRED_PATH, index=False)

    report = {
        "data_dir": str(DATA_DIR),
        "model_path": str(BEST_MODEL_PATH),
        "prediction_path": str(PRED_PATH),
        "lookback": LOOKBACK,
        "feature_count": len(features),
        "numeric_feature_count": len(numeric_cols),
        "online_lr": ONLINE_LR,
        "online_epochs_per_date": ONLINE_EPOCHS_PER_DATE,
        "valid_online_weighted_r2": float(valid_online_r2),
        "test_online_weighted_r2": float(test_online_r2),
        "history": history,
        "date_scores": valid_date_scores + test_date_scores,
        "elapsed_seconds": float(time.time() - start),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print("DONE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
