import os, logging, warnings

_P = os.path.dirname(os.path.abspath(__file__))
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emd

CSV = os.path.join(_P, "data", "temperature_tomsk_smoothed.csv")
IMG = os.path.join(_P, "images")
CUT = 36


def quinn(y):
    cfg = emd.sift.get_config("sift")
    cfg["extrema_opts/method"] = "rilling"
    cfg["imf_opts/sd_thresh"] = 0.05
    return emd.sift.sift(y, **cfg)


def main():
    os.makedirs(IMG, exist_ok=True)
    y = pd.read_csv(CSV).iloc[:, 0].values.astype(float)
    N = len(y)

    full = quinn(y)
    trunc = quinn(y[:N - CUT])
    m = min(full.shape[1], trunc.shape[1])
    L = N - CUT                     # длина общего участка
    lbls = [f"IMF{k+1}" if k < m - 1 else "остаток" for k in range(m)]

    print(f"Полный ряд против ряда без последних {CUT} месяцев\n")
    print(f"  мода    | RMSE по общему участку | RMSE у края ({CUT} мес)")
    print("  " + "-" * 50)

    for k in range(m):
        diff = full[:L, k] - trunc[:L, k]
        rms_all = np.sqrt(np.mean(diff ** 2))
        rms_edge = np.sqrt(np.mean(diff[-CUT:] ** 2))
        print(f"  {lbls[k]:7s} |        {rms_all:5.2f}          |     {rms_edge:5.2f}")

    show = min(L, 180)
    t = np.arange(L - show, L)

    fig, ax = plt.subplots(m, 1, figsize=(11, 1.7 * m), sharex=True)
    for k in range(m):
        ax[k].plot(t, full[L - show:L, k], "k", lw=1.2, label="полный ряд")
        ax[k].plot(t, trunc[L - show:L, k], "#d62728", lw=1.2, ls="--",
                   label=f"без {CUT} мес")
        ax[k].axvline(L - 1, color="0.6", lw=0.8)
        ax[k].set_ylabel(lbls[k], fontsize=9)
        
    ax[0].legend(fontsize=8, loc="upper left")
    ax[-1].set_xlabel("месяц")
    fig.suptitle(f"Моды полного ряда и ряда без последних {CUT} месяцев", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(IMG, "fig_edge_truncation.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
