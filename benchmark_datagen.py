"""
Ma'lumot generatsiyasini (yangi_datasett.py dagi algoritm) 1, 2, 4 jarayonda o'lchash.

Har bir sensor mustaqil, shuning uchun sensorlar jarayonlar o'rtasida bo'linadi (multiprocessing).
Qo'shimcha qator: "numpy vektorli" variant — parallellashdan oldin kodni to'g'ri yozish
qanchalik muhimligini ko'rsatadi.

Ishga tushirish (loyiha papkasida):
    python benchmark_datagen.py --procs 1 2 4 --repeats 3

Natija: datagen_results.csv va datagen_summary.txt  (ikkalasini ham yuboring).
"""
import argparse, csv, hashlib, os, platform, time
import numpy as np
from multiprocessing import Pool


def gen_sensor_loop(args):
    """Asl kodga mos: har bir kun uchun Python tsiklida qator qo'shish."""
    sid, days, years = args
    rng = np.random.default_rng(sid)                       # har sensor uchun aniq seed
    base = rng.uniform(10, 50)
    seasonal = 2 * np.sin(np.linspace(0, 2 * np.pi * years, days))
    water = base + seasonal + rng.normal(0, 0.5, days)
    temp = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi * years, days)) + rng.normal(0, 1, days)
    prec = rng.exponential(scale=2, size=days)
    rows = []
    for i in range(days):
        rows.append((i, sid, water[i], temp[i], prec[i]))
    return np.array(rows, dtype=np.float64)


def gen_sensor_vec(args):
    sid, days, years = args
    rng = np.random.default_rng(sid)
    base = rng.uniform(10, 50)
    seasonal = 2 * np.sin(np.linspace(0, 2 * np.pi * years, days))
    water = base + seasonal + rng.normal(0, 0.5, days)
    temp = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi * years, days)) + rng.normal(0, 1, days)
    prec = rng.exponential(scale=2, size=days)
    return np.column_stack([np.arange(days), np.full(days, sid), water, temp, prec])


def run(fn, procs, sensors, days, years):
    jobs = [(s, days, years) for s in range(sensors)]
    t0 = time.perf_counter()
    if procs == 1:
        parts = [fn(j) for j in jobs]
    else:
        with Pool(procs) as pool:
            parts = pool.map(fn, jobs, chunksize=max(1, sensors // (procs * 4)))
    arr = np.concatenate(parts)
    t = time.perf_counter() - t0
    return t, hashlib.md5(arr.tobytes()).hexdigest(), arr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--sensors", type=int, default=5000)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    days = a.years * 365
    maxp = os.cpu_count() or 1
    procs = [p for p in a.procs if p <= maxp]
    print(f"CPU: {platform.processor() or platform.machine()}, mantiqiy yadrolar: {maxp}")
    print(f"Sensorlar: {a.sensors}, kunlar: {days}, qatorlar: {a.sensors * days}")

    rows = []
    configs = [("tsikl", gen_sensor_loop, p) for p in procs] + [("numpy_vektor", gen_sensor_vec, 1)]
    for r in range(a.repeats):
        for name, fn, p in configs:
            t, h, n = run(fn, p, a.sensors, days, a.years)
            rows.append(dict(variant=name, procs=p, repeat=r + 1, time_s=t, rows=n, checksum=h))
            print(f"{name:13s} procs={p} repeat={r+1}: {t:.2f} s  (md5 {h[:8]})")
    with open("datagen_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    import statistics as st
    def mean(v, p): return st.mean(x["time_s"] for x in rows if x["variant"] == v and x["procs"] == p)
    t1 = mean("tsikl", 1)
    lines = ["variant        procs  time_mean_s  speedup  efficiency_%"]
    for p in procs:
        tp = mean("tsikl", p)
        lines.append(f"tsikl          {p:5d}  {tp:11.2f}  {t1/tp:7.2f}  {100*t1/tp/p:12.1f}")
    tv = mean("numpy_vektor", 1)
    lines.append(f"numpy_vektor   {1:5d}  {tv:11.2f}  {t1/tv:7.2f}  {'—':>12}")
    same = len({x["checksum"] for x in rows}) == 1
    lines.append(f"\nBarcha variantlarda natija bir xilmi (md5): {same}")
    txt = "\n".join(lines)
    open("datagen_summary.txt", "w").write(f"CPU yadrolari: {maxp}\n{txt}\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
