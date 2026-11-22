"""
Yer osti suvlari CNN+LSTM modelini CPU thread'lari soni bo'yicha o'lchash.

Nima o'lchanadi:
  - 1, 2, 4, 8 ... thread'da bitta epoxa (yoki bir nechta) o'qitish vaqti
  - Speedup S = T1/Tp,  Samaradorlik E = S/p,  Amdahl bo'yicha f = (1-1/S)/(1-1/p)
  - Har bir konfiguratsiyada TEST to'plamida RMSE va MAE (metrda) va R2

Ishga tushirish (loyiha papkasida, groundwater_data.csv yonida):
    pip install torch pandas numpy scikit-learn
    python benchmark_parallel.py --threads 1 2 4 8 --epochs 3 --repeats 3

Tez sinab ko'rish uchun (kichik qism):
    python benchmark_parallel.py --threads 1 2 --epochs 1 --repeats 1 --sensors 20

Natija: benchmark_results.csv va benchmark_summary.txt  (ikkalasini ham yuboring).
"""
import argparse, csv, os, platform, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN = 30
FEATURES = ["Water_Level", "Temperature", "Precipitation"]


class GroundwaterModel(nn.Module):          # loyihadagi model bilan bir xil
    def __init__(self, n_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2))
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_windows(arr, seq_len):
    """arr: [T, F] -> X [T-seq_len, seq_len, F], y [T-seq_len] (keyingi kun suv sathi)."""
    w = np.lib.stride_tricks.sliding_window_view(arr, seq_len, axis=0)  # [T-seq+1, F, seq]
    w = np.transpose(w, (0, 2, 1))
    return w[:-1].copy(), arr[seq_len:, 0].copy()


def load_data(path, n_sensors):
    df = pd.read_csv(path, parse_dates=["Date"])
    ids = sorted(df["Sensor_ID"].unique())
    if n_sensors:
        ids = ids[:n_sensors]
    parts = {"train": [], "val": [], "test": []}
    # Xronologik bo'linish HAR BIR sensor ichida: 70% / 15% / 15%.
    # Sensorlar chegarasida oyna hosil bo'lmaydi (asl kodda shu xato bor edi).
    raw_train = []
    series = {}
    for sid in ids:
        a = df[df.Sensor_ID == sid].sort_values("Date")[FEATURES].values.astype(np.float32)
        series[sid] = a
        raw_train.append(a[: int(len(a) * 0.70)])
    mn = np.concatenate(raw_train).min(0); mx = np.concatenate(raw_train).max(0)   # faqat train bo'yicha
    for sid, a in series.items():
        n = len(a); i1, i2 = int(n * 0.70), int(n * 0.85)
        s = (a - mn) / (mx - mn + 1e-9)
        for name, seg in (("train", s[:i1]), ("val", s[i1 - SEQ_LEN:i2]), ("test", s[i2 - SEQ_LEN:])):
            X, y = make_windows(seg, SEQ_LEN)
            parts[name].append((X, y))
    out = {}
    for k, v in parts.items():
        out[k] = (torch.tensor(np.concatenate([p[0] for p in v])),
                  torch.tensor(np.concatenate([p[1] for p in v])))
    return out, float(mx[0] - mn[0]), len(ids)


@torch.no_grad()
def evaluate(model, X, y, rng_m, bs=4096):
    model.eval(); preds = []
    for i in range(0, len(X), bs):
        preds.append(model(X[i:i + bs]).squeeze(1))
    p = torch.cat(preds)
    err = (p - y) * rng_m                      # metrga qaytarish
    rmse = torch.sqrt((err ** 2).mean()).item()
    mae = err.abs().mean().item()
    ss_res = ((p - y) ** 2).sum().item(); ss_tot = ((y - y.mean()) ** 2).sum().item()
    return rmse, mae, 1 - ss_res / ss_tot


def run_once(data, rng_m, threads, epochs, batch, seed):
    torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = data["train"]
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = GroundwaterModel(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    # isinish (warm-up): vaqtga kirmaydi
    model.train()
    xb, yb = next(iter(loader)); opt.zero_grad(); crit(model(xb).squeeze(1), yb).backward()
    torch.manual_seed(seed)
    model = GroundwaterModel(len(FEATURES)); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb).squeeze(1), yb)
            loss.backward(); opt.step()
    t = time.perf_counter() - t0
    rmse, mae, r2 = evaluate(model, *data["test"], rng_m)
    vr, vm, _ = evaluate(model, *data["val"], rng_m)
    return t, rmse, mae, r2, vr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="groundwater_data.csv")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--sensors", type=int, default=0, help="0 = hammasi (500)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    maxp = os.cpu_count() or 1
    threads = [p for p in a.threads if p <= maxp]
    skipped = [p for p in a.threads if p > maxp]
    print(f"CPU: {platform.processor() or platform.machine()}, mantiqiy yadrolar: {maxp}, torch {torch.__version__}")
    if skipped: print("Kompyuterda yetarli yadro yo'q, o'tkazib yuborildi:", skipped)

    data, rng_m, ns = load_data(a.data, a.sensors)
    print(f"Sensorlar: {ns}; train/val/test oynalar: "
          f"{len(data['train'][0])}/{len(data['val'][0])}/{len(data['test'][0])}")

    # Oldingi natijalarni o'qish (to'xtab qolsa, davom ettirish uchun)
    rows, done = [], set()
    fn = "benchmark_results.csv"
    if os.path.exists(fn):
        old = pd.read_csv(fn)
        for rec in old.to_dict("records"):
            rows.append(rec); done.add((int(rec["threads"]), int(rec["repeat"])))
        print(f"Oldingi {len(rows)} ta o'lchov topildi, davom ettiriladi.")
    for r in range(a.repeats):          # navbatma-navbat: har aylanishda barcha thread sonlari
        for p in threads:
            if (p, r + 1) in done:
                continue
            t, rmse, mae, r2, vr = run_once(data, rng_m, p, a.epochs, a.batch, a.seed)
            rows.append(dict(threads=p, repeat=r + 1, time_s=t, test_rmse_m=rmse,
                             test_mae_m=mae, test_r2=r2, val_rmse_m=vr))
            print(f"threads={p} repeat={r+1}: {t:.2f} s, RMSE={rmse:.4f} m, MAE={mae:.4f} m, R2={r2:.4f}")
            pd.DataFrame(rows).to_csv(fn, index=False)     # har o'lchovdan keyin saqlash

    df = pd.DataFrame(rows)
    g = df.groupby("threads").agg(time_mean=("time_s", "mean"), time_std=("time_s", "std"),
                                  rmse=("test_rmse_m", "mean"), mae=("test_mae_m", "mean"),
                                  r2=("test_r2", "mean")).reset_index()
    t1 = g.loc[g.threads == g.threads.min(), "time_mean"].iloc[0]
    g["speedup"] = t1 / g.time_mean
    g["efficiency_%"] = 100 * g.speedup / g.threads
    g["amdahl_f"] = [(1 - 1 / s) / (1 - 1 / p) if p > 1 else float("nan") for s, p in zip(g.speedup, g.threads)]
    txt = g.round(4).to_string(index=False)
    meta = dict(cpu=platform.processor() or platform.machine(), logical_cores=maxp, torch=torch.__version__,
                epochs=a.epochs, repeats=a.repeats, batch=a.batch, sensors=ns, seq_len=SEQ_LEN)
    with open("benchmark_summary.txt", "w") as f:
        f.write(json.dumps(meta, indent=1) + "\n\n" + txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
