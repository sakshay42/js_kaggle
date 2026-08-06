from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "top20_rolling_p8"
MODEL_DIR = PROJECT_ROOT / "models" / "neural" / "rolling_gru_mlp"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALID_PATH = DATA_DIR / "valid.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
REPORT_PATH = LOG_DIR / "train_gru_mlp_rolling_report.json"
BEST_MODEL_PATH = MODEL_DIR / "best_gru_mlp.pt"
PRED_PATH = MODEL_DIR / "predictions.parquet"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
DATE_COL = "date_id"
EMBED_COLS = ["symbol_id", "time_id"]
SEED = 42
BATCH_SIZE = 4096
EPOCHS = 6
LR = 8e-4
WEIGHT_DECAY = 1e-4
LOOKBACK = 16


class SequenceDataset(Dataset):
    def __init__(self, df, numeric_cols, cat_maps, scaler=None, fit_scaler=False):
        df = df.sort_values(["symbol_id", DATE_COL, "time_id"]).reset_index(drop=True)
        self.ids = df[[DATE_COL, "time_id", "symbol_id"]].copy()
        self.y = df[TARGET_COL].to_numpy(np.float32)
        self.w = df[WEIGHT_COL].to_numpy(np.float32)

        X = df[numeric_cols].to_numpy(np.float32)
        if fit_scaler:
            mean = np.nanmean(X, axis=0).astype(np.float32)
            std = np.nanstd(X, axis=0).astype(np.float32)
            std[std < 1e-6] = 1.0
            scaler = {"mean": mean, "std": std}
        X = (X - scaler["mean"]) / scaler["std"]
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.scaler = scaler

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
            nn.Linear(n_num + emb_dim + 64, 256),
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


def make_cat_maps(train_df):
    maps = {}
    sizes = []
    for col in EMBED_COLS:
        values = sorted(int(v) for v in train_df[col].dropna().unique())
        maps[col] = {v: i + 1 for i, v in enumerate(values)}
        sizes.append(len(values) + 1)
    return maps, sizes


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys, ws = [], [], []
    for x_num, x_seq, x_cat, y, w in loader:
        pred = model(x_num.to(device), x_seq.to(device), x_cat.to(device)).detach().cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
        ws.append(w.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_df = load_df(TRAIN_PATH, "train")
    valid_df = load_df(VALID_PATH, "valid")
    test_df = load_df(TEST_PATH, "test")

    features = feature_cols(train_df)
    numeric_cols = [c for c in features if c not in EMBED_COLS]
    cat_maps, cat_sizes = make_cat_maps(train_df)

    train_ds = SequenceDataset(train_df, numeric_cols, cat_maps, fit_scaler=True)
    valid_ds = SequenceDataset(valid_df, numeric_cols, cat_maps, scaler=train_ds.scaler)
    test_ds = SequenceDataset(test_df, numeric_cols, cat_maps, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = GRUMLP(len(numeric_cols), cat_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_score = -np.inf
    best_epoch = None
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x_num, x_seq, x_cat, y, w in train_loader:
            x_num = x_num.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(x_num, x_seq, x_cat), y, w)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())

        valid_pred, valid_y, valid_w = predict(model, valid_loader, device)
        valid_r2 = weighted_r2(valid_y, valid_pred, valid_w)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_loader), "valid_weighted_r2": float(valid_r2)}
        history.append(row)
        print(row, flush=True)
        if valid_r2 > best_score:
            best_score = valid_r2
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "numeric_cols": numeric_cols,
                    "embed_cols": EMBED_COLS,
                    "cat_maps": cat_maps,
                    "cat_sizes": cat_sizes,
                    "scaler": train_ds.scaler,
                    "lookback": LOOKBACK,
                    "best_epoch": best_epoch,
                    "best_valid_weighted_r2": float(best_score),
                },
                BEST_MODEL_PATH,
            )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_pred, valid_y, valid_w = predict(model, valid_loader, device)
    test_pred, test_y, test_w = predict(model, test_loader, device)
    valid_score = weighted_r2(valid_y, valid_pred, valid_w)
    test_score = weighted_r2(test_y, test_pred, test_w)

    pred_df = pd.concat(
        [
            valid_ds.ids.assign(
                split="valid", responder_6=valid_y, weight=valid_w, gru_mlp_pred=valid_pred
            ),
            test_ds.ids.assign(
                split="test", responder_6=test_y, weight=test_w, gru_mlp_pred=test_pred
            ),
        ],
        ignore_index=True,
    )
    pred_df.to_parquet(PRED_PATH, index=False)

    report = {
        "data_dir": str(DATA_DIR),
        "model_path": str(BEST_MODEL_PATH),
        "prediction_path": str(PRED_PATH),
        "lookback": LOOKBACK,
        "feature_count": len(features),
        "numeric_feature_count": len(numeric_cols),
        "embed_cols": EMBED_COLS,
        "best_epoch": best_epoch,
        "valid_weighted_r2": float(valid_score),
        "test_weighted_r2": float(test_score),
        "history": history,
        "elapsed_seconds": float(time.time() - start),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print("DONE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
