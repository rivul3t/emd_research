import os, logging, warnings

pat = os.path.dirname(os.path.abspath(__file__))
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emd
from PyEMD import EMD

DATA = os.path.join(pat, "data")
IMG = os.path.join(pat, "images")
PERIOD = 12
WIN = 180 

def load(name):
    return pd.read_csv(os.path.join(DATA, name)).iloc[:, 0].values.astype(float)


def quinn(y):
    cfg = emd.sift.get_config("sift")
    cfg["extrema_opts/method"] = "rilling"
    cfg["imf_opts/sd_thresh"] = 0.05

    return emd.sift.sift(y, **cfg)


def pyemd(y):
    return EMD().emd(y).T


def annual_corr(mode):
    if np.std(mode) < 1e-12:
        return 0.0

    t = np.arange(len(mode))
    c = np.corrcoef(mode, np.cos(2 * np.pi * t / PERIOD))[0, 1]
    s = np.corrcoef(mode, np.sin(2 * np.pi * t / PERIOD))[0, 1]

    return float(np.hypot(c, s))


LIBS = {"emd": quinn, "PyEMD": pyemd}
DATASETS = {"сырой": "temperature_tomsk.csv", "сглаж.": "temperature_tomsk_smoothed.csv"}
FIG = {("сырой", "emd"): ("fig_var_raw_quinn.png", "Сырой ряд, библиотека emd"),
       ("сырой", "PyEMD"): ("fig_var_raw_pyemd.png", "Сырой ряд, библиотека PyEMD"),
       ("сглаж.", "emd"): ("fig_var_smooth_quinn.png", "Сглаженный ряд, библиотека emd"),
       ("сглаж.", "PyEMD"): ("fig_var_smooth_pyemd.png", "Сглаженный ряд, библиотека PyEMD")}


def plot_modes(y, imf, fname, title):
    n = imf.shape[1]
    t = np.arange(len(y))

    fig, ax = plt.subplots(n + 1, 1, figsize=(9, 1.25 * (n + 1)), sharex=True)
    ax[0].plot(t, y, "k", lw=0.6)
    ax[0].set_ylabel("сигнал", fontsize=9)

    for k in range(n):
        lbl = f"IMF{k+1}" if k < n - 1 else "остаток"
        ax[k + 1].plot(t, imf[:, k], lw=0.7, color="#2ca02c" if k == n - 1 else "#1f77b4")
        ax[k + 1].set_ylabel(lbl, fontsize=9)

    ax[-1].set_xlabel("месяц")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(IMG, fname), dpi=120)
    plt.close(fig)


def compare():
    print("Сравнение вариантов разложения\n")
    print("  набор    | библиотека | число мод | годовойцикл |corr| | реконструкция")
    print("  " + "-" * 62)

    chosen = None
    for dname, fname in DATASETS.items():
        y = load(fname)
        for lname, fn in LIBS.items():

            imf = fn(y)
            n = imf.shape[1]
            recon = float(np.max(np.abs(imf.sum(1) - y)))
            corr = [annual_corr(imf[:, k]) for k in range(min(2, n))]
            which = int(np.argmax(corr)) + 1
            best = max(corr)

            print(f"  {dname:8s} | {lname:10s} |  {n-1}+ост   |  {best:.2f} (IMF{which})"
                  f"  | {recon:.0e}")
            f, title = FIG[(dname, lname)]
            plot_modes(y, imf, f, title)
            if dname == "сглаж." and lname == "emd":
                chosen = (y, imf)
    return chosen


def real_scale(y, imf):
    n = imf.shape[1]
    t = np.arange(len(y))

    fig, ax = plt.subplots(n + 1, 1, figsize=(11, 1.4 * (n + 1)), sharex=True)
    ax[0].plot(t, y, "k", lw=0.7)
    ax[0].set_ylabel(f"сигнал\nразмах {np.ptp(y):.0f}°C", fontsize=9)

    for k in range(n):
        lbl = f"IMF{k+1}" if k < n - 1 else "остаток"
        ax[k + 1].plot(t, imf[:, k], lw=0.9, color="#2ca02c" if k == n - 1 else "#1f77b4")
        ax[k + 1].axhline(0, color="0.6", lw=0.5)
        ax[k + 1].set_ylabel(f"{lbl}\nразмах {np.ptp(imf[:, k]):.0f}°C", fontsize=9)

    ax[-1].set_xlabel("месяц")
    fig.suptitle("Моды выбранного разложения в реальном масштабе", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(IMG, "fig_real_scale.png"), dpi=110)
    plt.close(fig)


def annual(y, imf):
    corr = annual_corr(imf[:, 0])
    n = min(WIN, len(y))
    t = np.arange(n)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(t, y[:n], "k", lw=1.0)
    ax[0].set_ylabel("сигнал, °C", fontsize=9)
    ax[0].set_title(f"Первые {n // PERIOD} лет", fontsize=11)
    ax[1].plot(t, imf[:n, 0], color="#1f77b4", lw=1.2)
    ax[1].set_ylabel("IMF1, °C", fontsize=9)

    for a in ax:
        for m in range(0, n + 1, PERIOD):
            a.axvline(m, color="0.8", lw=0.6, ls="--", zorder=0)
        a.axhline(0, color="0.6", lw=0.5)

    ax[1].set_xlabel("месяц")
    ax[1].set_xlim(0, n - 1)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig_annual.png"), dpi=110)
    plt.close(fig)
    
    print(f"\nвыбранный вариант, корреляция IMF1 с годовым циклом {corr:.2f}")


def main():
    os.makedirs(IMG, exist_ok=True)
    print(f"emd {emd.__version__}\n")
    y, imf = compare()
    real_scale(y, imf)
    annual(y, imf)


if __name__ == "__main__":
    main()
