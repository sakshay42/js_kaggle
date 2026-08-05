import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "logs" / "gpu_check_report.json"


def run_command(cmd):
    if shutil.which(cmd[0]) is None:
        return {"ok": False, "error": f"{cmd[0]} not found"}
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def test_lightgbm_gpu():
    result = {"available": False}
    try:
        import lightgbm as lgb

        rng = np.random.default_rng(42)
        X = rng.normal(size=(2000, 20)).astype(np.float32)
        y = (X[:, 0] * 0.1 + rng.normal(size=2000) * 0.01).astype(np.float32)

        train_set = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "metric": "l2",
            "device_type": "gpu",
            "verbosity": -1,
            "num_leaves": 16,
            "learning_rate": 0.1,
            "seed": 42,
        }
        model = lgb.train(params, train_set, num_boost_round=5)
        preds = model.predict(X[:5])
        result.update(
            {
                "available": True,
                "version": lgb.__version__,
                "pred_sample": [float(x) for x in preds],
            }
        )
    except Exception as exc:
        result.update({"error": repr(exc)})
    return result


def test_xgboost_gpu():
    result = {"available": False}
    try:
        import xgboost as xgb

        rng = np.random.default_rng(42)
        X = rng.normal(size=(2000, 20)).astype(np.float32)
        y = (X[:, 0] * 0.1 + rng.normal(size=2000) * 0.01).astype(np.float32)

        model = xgb.XGBRegressor(
            n_estimators=5,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X[:5])
        result.update(
            {
                "available": True,
                "version": xgb.__version__,
                "pred_sample": [float(x) for x in preds],
            }
        )
    except Exception as exc:
        result.update({"error": repr(exc)})
    return result


def test_pytorch_gpu():
    result = {"available": False}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        result["version"] = torch.__version__
        result["cuda_available"] = bool(cuda_available)
        result["cuda_version"] = torch.version.cuda

        if not cuda_available:
            result["error"] = "torch.cuda.is_available() is False"
            return result

        torch.manual_seed(42)
        device = torch.device("cuda")
        X = torch.randn((2000, 20), dtype=torch.float32, device=device)
        true_w = torch.randn((20, 1), dtype=torch.float32, device=device)
        y = X @ true_w + 0.01 * torch.randn((2000, 1), dtype=torch.float32, device=device)

        model = torch.nn.Linear(20, 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()

        result.update(
            {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "final_loss": float(loss.detach().cpu()),
            }
        )
    except Exception as exc:
        result.update({"error": repr(exc)})
    return result


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "nvidia_smi": run_command(["nvidia-smi"]),
        "lightgbm_gpu": test_lightgbm_gpu(),
        "xgboost_gpu": test_xgboost_gpu(),
        "pytorch_gpu": test_pytorch_gpu(),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("DONE")
    print(f"report={REPORT_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
