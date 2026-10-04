"""Step 6: charts.   ~/tools/venv/bin/python charts.py -> charts/00_cover.png, 01_coverage.png, 02_dispersion.png, 03_stores.png, 04_size.png, 05_inflation.png  (every number comes from results/)"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
CM = pd.read_csv(R / "coverage_month.csv"); CM = CM[CM["month"] < CM["month"].max()].reset_index(drop=True); CC = pd.read_csv(R / "coverage_country.csv"); DS = pd.read_csv(R / "dispersion_summary.csv"); DN = pd.read_csv(R / "dispersion_by_stores.csv", dtype={"stores": str})
DC = pd.read_csv(R / "dispersion_country.csv"); LA = pd.read_csv(R / "ladder_a.csv").set_index("key")["value"]; CH = pd.read_csv(R / "chains.csv"); LB = pd.read_csv(R / "ladder_b.csv").set_index("model")
PU = pd.read_csv(R / "produce_units.csv"); IN = pd.read_csv(R / "inflation.csv"); CO = pd.read_csv(R / "concentration.csv").set_index("key")["value"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def pcts(ax, bars, vals, fmt="{:.0f}%", dx=0.0, horizontal=False, fs=9):
    for b, v in zip(bars, vals):
        if horizontal:
            ax.text(b.get_width() + (0.4 if v >= 0 else -0.4), b.get_y() + b.get_height() / 2, fmt.format(v) if abs(v) >= 0.5 else "0%", va="center", ha="left" if v >= 0 else "right", fontsize=fs, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dx, fmt.format(v), ha="center", va="bottom", fontsize=fs, color=INK)


def stores_bars(ax, fs=10):
    x = np.arange(len(DN)); v = 100 * DN["median_spread"]; b = ax.bar(x, v, color=BLUE, width=0.6)
    pcts(ax, b, v, fs=fs + 0.5); ax.set_xticks(x); ax.set_xticklabels([f"{s} shops\n({c:,} cases)" for s, c in zip(DN["stores"], DN["cells"])], fontsize=fs - 1)
    ax.set_ylabel("dearest shop over cheapest, same barcode (%)"); ax.set_ylim(0, 38); ax.grid(axis="x", visible=False)


def chain_bars(ax, fs=9.5):
    v = 100 * CH["effect_pct"]; y = np.arange(len(CH)); cols = [GREEN if t < 0 else ORANGE for t in v]
    b = ax.barh(y, v, color=cols, height=0.62); pcts(ax, b, v, fmt="{:+.0f}%", horizontal=True, fs=fs)
    ax.set_yticks(y); ax.set_yticklabels([f"{c} ({s})" for c, s in zip(CH["chain"], CH["stores"])], fontsize=fs); ax.axvline(0, color=DIM, linewidth=0.9)
    ax.set_xlim(-18, 36); ax.set_xlabel("vs. the same barcodes elsewhere (%), (n) = shops"); ax.grid(axis="y", visible=False)


def cover():
    fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.15]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    stores_bars(a, fs=11); a.set_title("Same barcode, different shops: the gap", fontsize=14); chain_bars(b, fs=11); b.set_title("Who is cheaper, same barcodes, French chains", fontsize=14)
    s0 = DS.iloc[0]
    fig.suptitle(f"One barcode, {100 * s0['median_spread']:.0f}% apart across French shops (median); the shop's price level explains {100 * LA['share_removed_by_store']:.0f}% of the variance", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)


def coverage():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); ax = axes[0]
    ax.bar(np.arange(len(CM)), CM["prices"] / 1000, color=BLUE, width=0.8); ax.set_xticks(np.arange(0, len(CM), 3)); ax.set_xticklabels([m[2:] for m in CM["month"][::3]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("prices entered per month (thousands)"); ax.grid(axis="x", visible=False); ax.set_title("Prices by month of the price date", fontsize=12)
    i = CM["prices"].idxmax(); ax.text(i - 0.4, CM["prices"].max() / 1000, f"{CM['month'][i]}: {CM['prices'].max() / 1000:.0f}k", ha="right", va="top", fontsize=9, color=INK)
    ax = axes[1]; d = CC.head(8).iloc[::-1]; b = ax.barh(np.arange(len(d)), 100 * d["share"], color=BLUE, height=0.62); pcts(ax, b, 100 * d["share"], fmt="{:.0f}%", horizontal=True)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["country"]); ax.set_xlabel("share of cleaned prices (%)"); ax.grid(axis="y", visible=False); ax.set_xlim(0, 78)
    ax.set_title(f"By country; top 1% of contributors enter {100 * float(CO['top1pct_contributors_share']):.0f}%", fontsize=12); save(fig, "01_coverage.png")


def dispersion():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); ax = axes[0]
    names = ["no promotions,\nno far outliers", "far outliers\nkept", "promotions\nkept", "price tags\nonly", "receipts\nonly"]; x = np.arange(5)
    for xi, (_, r) in zip(x, DS.iterrows()):
        ax.plot([xi, xi], [100 * r["p25_spread"], 100 * r["p75_spread"]], color=GRAY, linewidth=7, solid_capstyle="butt"); ax.scatter([xi], [100 * r["median_spread"]], color=BLUE, s=60, zorder=3); ax.scatter([xi], [100 * r["p90_spread"]], color=ORANGE, marker="v", s=46, zorder=3)
        ax.text(xi + 0.12, 100 * r["median_spread"], f"{100 * r['median_spread']:.0f}%", fontsize=8.5, va="center", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0, 58); ax.set_xlim(-0.5, 4.6); ax.set_ylabel("dearest over cheapest shop (%)"); ax.grid(axis="x", visible=False)
    ax.scatter([], [], color=BLUE, s=40, label="median"); ax.scatter([], [], color=ORANGE, marker="v", s=36, label="90th percentile"); ax.plot([], [], color=GRAY, linewidth=6, label="middle half"); ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper right")
    ax.set_title("France, five ways to clean the same prices", fontsize=12)
    ax = axes[1]; d = DC.iloc[::-1]; y = np.arange(len(d)); ax.barh(y - 0.19, 100 * d["median_spread"], height=0.36, color=BLUE, label="median"); ax.barh(y + 0.19, 100 * d["p90_spread"], height=0.36, color=ORANGE, label="90th percentile")
    for yi, m, p in zip(y, d["median_spread"], d["p90_spread"]):
        ax.text(100 * m + 0.6, yi - 0.19, f"{100 * m:.0f}%", va="center", fontsize=8.5, color=INK); ax.text(100 * p + 0.6, yi + 0.19, f"{100 * p:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f"{c} ({n:,})" for c, n in zip(d["country"], d["cells"])], fontsize=9); ax.set_xlim(0, 60); ax.set_xlabel("dearest over cheapest shop (%), (n) = cases"); ax.grid(axis="y", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right")
    ax.set_title("By country, each in its own currency", fontsize=12); save(fig, "02_dispersion.png")


def stores():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); ax = axes[0]
    v = [100 * LA["median_spread_raw"], 100 * LA["median_spread_after_store"], 100 * LA["p90_spread_raw"], 100 * LA["p90_spread_after_store"]]; x = np.array([0, 1, 2.6, 3.6])
    b = ax.bar(x, v, color=[BLUE, GREEN, BLUE, GREEN], width=0.8); pcts(ax, b, v, fs=10); ax.set_xticks(x); ax.set_xticklabels(["raw", "shop level\nremoved", "raw", "shop level\nremoved"], fontsize=9)
    ax.text(0.5, -9.5, "median case", ha="center", fontsize=9, color=DIM); ax.text(3.1, -9.5, "90th percentile case", ha="center", fontsize=9, color=DIM)
    ax.set_ylabel("dearest over cheapest shop (%)"); ax.set_ylim(0, 52); ax.grid(axis="x", visible=False); ax.set_title(f"Shop price level: {100 * LA['share_removed_by_store']:.0f}% of the variance", fontsize=12)
    chain_bars(axes[1], fs=8.5); axes[1].set_title("Chain price levels (same barcodes)", fontsize=12); save(fig, "03_stores.png")


def size():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); ax = axes[0]
    r = [100 * LB.loc[m, "r2"] for m in ("price: category", "price: category + size", "price: category + size + store")]; x = np.arange(3)
    bb = ax.bar(x, r, color=[BLUE, GREEN, ORANGE], width=0.6); pcts(ax, bb, r, fmt="{:.1f}%", fs=10); ax.axhline(100, color=DIM, linewidth=0.9, linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels(["category", "+ pack size", "+ shop"], fontsize=10); ax.set_ylim(0, 112); ax.set_ylabel("share of the variance of log price (%)"); ax.grid(axis="x", visible=False)
    ax.text(2.45, 70, f"{100 - r[2]:.0f}% left over\n(brand, quality, pack type)", ha="right", fontsize=8.5, color=DIM)
    ax.set_title(f"Across barcodes: slope on log size {LB.loc['price: category + size', 'slope_log_size']:.2f}", fontsize=12)
    ax = axes[1]; d = PU.iloc[::-1].reset_index(drop=True); y = np.arange(len(d)); ax.barh(y + 0.19, d["median_per_kg"], height=0.36, color=BLUE, label='entered "per kilogram"'); ax.barh(y - 0.19, d["median_per_unit"], height=0.36, color=ORANGE, label='entered "per unit"')
    ax.set_yticks(y); ax.set_yticklabels([t.replace("en:", "") for t in d["category_tag"]], fontsize=9); ax.set_xlabel("median price entered (EUR)"); ax.set_xlim(0, 9); ax.grid(axis="y", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right")
    ax.set_title("Raw produce: kilo or piece?", fontsize=12); save(fig, "04_size.png")


def inflation():
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.9)); x = np.arange(len(IN)); m = 100 * IN["mean_change"]; lo = 100 * np.expm1(np.log1p(IN["mean_change"]) - 1.96 * IN["se_mean_log"]); hi = 100 * np.expm1(np.log1p(IN["mean_change"]) + 1.96 * IN["se_mean_log"])
    ax.bar(x - 0.2, m, width=0.36, color=BLUE, label="crowd: same barcode, same shop, a year apart (mean, 95% band)"); ax.errorbar(x - 0.2, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    ax.bar(x + 0.2, 100 * IN["hicp_food_yoy"], width=0.36, color=ORANGE, label="Eurostat HICP France, food"); ax.scatter(x + 0.2, 100 * IN["hicp_all_yoy"], color=GREEN, marker="D", s=34, zorder=3, label="HICP France, all items")
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels([f"{q}\n{n:,} pairs" for q, n in zip(IN["quarter"], IN["pairs"])], fontsize=8); ax.set_ylabel("price change on a year earlier (%)"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower left"); ax.set_title("A year of crowd prices against official inflation (no official figure after 2025 in the Eurostat file)", fontsize=11); save(fig, "05_inflation.png")


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    cover(); coverage(); dispersion(); stores(); size(); inflation()


if __name__ == "__main__":
    main()
