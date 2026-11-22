# Parallelization Efficiency of CNN-LSTM Training and Data Preparation for Groundwater Level Forecasting on a Multi-Core CPU

Ushbu repozitoriyda quyidagi maqolaning o'lchov skriptlari va xom natijalari joylashgan:

> Rajabov F.F., Abdupattoyeva Z. Yer osti suvlari sathini prognozlash uchun CNN-LSTM modelini o'qitish va
> ma'lumotlarni tayyorlashni ko'p yadroli CPU'da parallellash samaradorligi.

This repository contains the benchmark scripts and raw measurements for the paper above.

## Tarkibi / Contents

| Fayl | Vazifasi |
|---|---|
| `benchmark_parallel.py` | CNN-LSTM o'qitish vaqtini PyTorch thread soni (1, 2, 4) bo'yicha o'lchash; S, E, Amdahl f, RMSE/MAE/R² |
| `benchmark_datagen.py` | Sun'iy ma'lumotlar generatsiyasini jarayonlar soni (1, 2, 4) bo'yicha o'lchash va numpy vektorli variant bilan solishtirish |
| `results/` | Xom o'lchov natijalari (`benchmark_results.csv`, `benchmark_summary.txt`, `datagen_results.csv`, `datagen_summary.txt`) |
| `requirements.txt` | Kerakli kutubxonalar |

## Muhit / Environment

Maqoladagi o'lchovlar: Intel Core i5-7200U (2 fizik, 4 mantiqiy yadro), ≈6 GB RAM, Windows 10 (10.0.19045),
Python 3.15, PyTorch 2.15.0 (CPU).

## Ishga tushirish / How to run

```bash
pip install -r requirements.txt

# 1) Ma'lumotlar generatsiyasi benchmarki (5000 sensor, 9 125 000 qator)
python benchmark_datagen.py --procs 1 2 4 --repeats 3

# 2) O'qitish benchmarki (groundwater_data.csv shu papkada bo'lishi kerak)
python benchmark_parallel.py --threads 1 2 4 --epochs 2 --repeats 3 --sensors 100
```

`benchmark_parallel.py` sun'iy `groundwater_data.csv` faylini o'qiydi (500 sensor, 1825 kun, ustunlar:
`Date, Sensor_ID, Water_Level, Temperature, Precipitation`).

## Ogohlantirish / Note

Barcha tajribalar sun'iy ma'lumotlarda o'tkazilgan; aniqlik ko'rsatkichlari real sensor ma'lumotlariga tatbiq etilmaydi.
All experiments use synthetic data; accuracy figures do not transfer to real sensor data.

## Iqtibos / Citation

Maqoladan foydalansangiz, iltimos, unga iqtibos keltiring (qarang: `CITATION.cff`).
