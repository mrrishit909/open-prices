"""Steps 2-4: audit the Open Prices dump, then measure what the crowd's shelf prices show.   ~/tools/venv/bin/python analyze.py -> results/*.csv

Data: data/prices.csv.gz (every price in Open Prices, ODbL, volunteers' photos of shelf labels and receipts), data/off_products.csv.gz (Open Food Facts pack-size text and categories for those barcodes),
data/hicp_fr.json (Eurostat HICP France). Published here: aggregates only (no rows, no contributor ids, no photos).

Audit: a cleaning funnel (audit.csv), and every data-quality trap counted (dq.csv). The cleaned set A: a barcode product row with a positive price, a currency, a date between 2018 and the pull date, a store with a country, the
country's own currency, and one row per (barcode, store, date, price, currency). A "far outlier" is a price at least 3 times or at most a third of its barcode's median in the same currency (barcodes with >= 5 prices).
The study sample S: France, euro, 2025-01-01 on, price tag or receipt proof, not a promotion, not a far outlier. (Size note: the same barcode is the same pack, so size is held fixed inside a barcode.)

Coverage: by country and month; concentration of contributors, stores and proof types.
Dispersion: one value per (barcode, store, quarter) = mean log price. A cell is a (barcode, quarter) sold in >= 2 stores; spread = max/min - 1 over the stores. Variants: far outliers kept, promotions kept, receipts only, price tags only.
Ladder A: within the largest connected set of cells, log price = cell effect + store effect + residual (least squares on dummies, sparse LSQR); the share of the within-cell variance that the store effect removes, and the store effects by chain.
Ladder B (across barcodes): log price on category, log pack size (g or ml parsed from the Open Food Facts text "250 g", "1 l", ...), store; and the same in price per 100 g/ml.
Proof effect: receipts, loyalty-card exports (GDPR_REQUEST) and shop imports against price tags, inside (barcode, quarter) and store, store-clustered errors.
Inflation: the same barcode in the same store about a year apart (300-430 days, closest to 365), bucketed by the quarter of the later price, against Eurostat HICP France food and all-items over the same quarter.
Cross-country: the same barcode in France and another euro country (no currency conversion needed), 2025 on.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
PULL = json.loads((R / "pull.json").read_text())["pulled"]
FIRST_OK, STUDY_FROM = "2018-01-01", "2025-01-01"
SHELF = ("PRICE_TAG", "RECEIPT")
EXPECT = {"FR": "EUR", "DE": "EUR", "BE": "EUR", "IT": "EUR", "ES": "EUR", "AT": "EUR", "FI": "EUR", "HR": "EUR", "NL": "EUR", "PT": "EUR", "IE": "EUR", "LU": "EUR", "GR": "EUR", "SK": "EUR", "SI": "EUR",
          "LT": "EUR", "LV": "EUR", "EE": "EUR", "MT": "EUR", "CY": "EUR", "MC": "EUR", "US": "USD", "NO": "NOK", "SE": "SEK", "GB": "GBP", "PL": "PLN", "DK": "DKK", "CH": "CHF", "CA": "CAD", "JP": "JPY",
          "RU": "RUB", "IN": "INR", "UA": "UAH", "MX": "MXN", "IL": "ILS", "CZ": "CZK", "HU": "HUF", "RO": "RON", "BG": "BGN", "AU": "AUD", "NZ": "NZD", "BR": "BRL", "TR": "TRY"}
EURO = [c for c, v in EXPECT.items() if v == "EUR"]
CHAINS = [("E.Leclerc", ["leclerc"]), ("Auchan", ["auchan"]), ("Intermarché", ["intermarché", "intermarche"]), ("Carrefour", ["carrefour"]), ("Système U", ["super u", "hyper u", "u express", "système u", "systeme u", "marché u"]),
          ("Monoprix", ["monoprix"]), ("Franprix", ["franprix"]), ("Lidl", ["lidl"]), ("Aldi", ["aldi"]), ("Biocoop", ["biocoop"]), ("Naturalia", ["naturalia"]), ("Casino", ["casino"]), ("Picard", ["picard"]),
          ("Grand Frais", ["grand frais"]), ("Cora", ["cora"]), ("Match", ["match"]), ("G20", ["g20"]), ("La Vie Claire", ["la vie claire"])]
SIZE_RX = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(g|gr|gram|grams|kg|ml|cl|dl|l|litre|liter|litres|liters)\s*(?:e|℮)?\s*$", re.I)
SIZE_F = {"g": 1, "gr": 1, "gram": 1, "grams": 1, "kg": 1000, "ml": 1, "cl": 10, "dl": 100, "l": 1000, "litre": 1000, "liter": 1000, "litres": 1000, "liters": 1000}
EXPECT_LEN = (8, 12, 13, 14)
MIN_CAT_BARCODES = 30
FAR = 3.0          # a price at least FAR times or at most 1/FAR of its barcode median is a far outlier


def save(name, df):
    df.to_csv(R / f"{name}.csv", index=False)


def key(c):
    return c.lstrip("0")


def chain_of(display):
    first = display.split(",")[0].lower()
    for name, kws in CHAINS:
        if any(re.search(r"\b" + re.escape(k) + r"\b", first) for k in kws):
            return name
    return "other"


def quarter(d):
    return f"{d[:4]}-Q{(int(d[5:7]) - 1) // 3 + 1}"


def load():
    raw = pd.read_csv(D / "prices.csv.gz", dtype=str, keep_default_na=False)
    raw["id"] = raw["id"].astype(int)
    raw["p"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["pwd"] = pd.to_numeric(raw["price_without_discount"], errors="coerce")
    raw["disc"] = raw["price_is_discounted"] == "True"
    raw["cc"] = raw["location_osm_address_country_code"]
    raw["store"] = raw["location_id"]
    return raw


def fe_fit(y, groups, extra=None):
    """Least squares of y on an intercept, dummies for each group vector (first level of every group dropped) and the columns of extra (n x k).
    Returns (residuals, effects per group as arrays of the group's codes, extra coefficients)."""
    n = len(y); cols = [sparse.csr_matrix(np.ones((n, 1)))]; offs = []; pos = 1
    for g in groups:
        codes, uniq = pd.factorize(g); k = len(uniq)
        cols.append(sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, k))[:, 1:]); offs.append((pos, k, uniq)); pos += k - 1
    if extra is not None:
        cols.append(sparse.csr_matrix(extra)); ex0 = pos
    X = sparse.hstack(cols, format="csr")
    sol = lsqr(X, y, atol=1e-14, btol=1e-14, iter_lim=20000, conlim=1e12)[0]
    res = y - X @ sol
    eff = []
    for (p0, k, uniq) in offs:
        e = np.zeros(k); e[1:] = sol[p0:p0 + k - 1]; eff.append(pd.Series(e, index=uniq))
    return res, eff, (sol[ex0:] if extra is not None else None)


def main():
    R.mkdir(exist_ok=True)
    raw = load(); N = len(raw)
    # ---------- funnel ----------
    ok = pd.Series(True, index=raw.index); funnel = []

    def step(name, cond):
        nonlocal ok
        before = int(ok.sum()); ok = ok & cond; funnel.append({"step": name, "rows": int(ok.sum()), "dropped": before - int(ok.sum())})

    funnel.append({"step": "every row in the dump", "rows": N, "dropped": 0})
    step("a barcode product (drop raw produce with no barcode)", (raw["type"] == "PRODUCT") & (raw["product_code"] != ""))
    step("price above zero", raw["p"] > 0)
    step("currency present", raw["currency"] != "")
    step("date present", raw["date"] != "")
    step(f"date from {FIRST_OK} to the pull date", (raw["date"] >= FIRST_OK) & (raw["date"] <= PULL))
    step("store has a country", raw["cc"] != "")
    exp = raw["cc"].map(EXPECT)
    step("currency is the country's own", exp.isna() | (exp == raw["currency"]))
    pre = raw[ok].copy()
    dupkey = ["product_code", "store", "date", "p", "currency"]
    dup_surplus_mask = pre.sort_values("id").duplicated(subset=dupkey, keep="first").reindex(pre.index)
    A = pre[~dup_surplus_mask].copy()
    funnel.append({"step": "one row per (barcode, store, date, price, currency)", "rows": len(A), "dropped": len(pre) - len(A)})
    A["month"] = A["date"].str[:7]; A["quarter"] = A["date"].map(quarter)
    A["lp"] = np.log(A["p"])
    # far outliers: price >= 3x or <= 1/3 of the barcode's median in that currency, barcodes with >= 5 prices
    g = A.groupby(["product_code", "currency"])["p"]
    A["bn"] = g.transform("size"); A["bmed"] = g.transform("median"); A["ratio"] = A["p"] / A["bmed"]
    A["far"] = (A["bn"] >= 5) & ((A["ratio"] >= FAR) | (A["ratio"] <= 1 / FAR))
    A["suspect"] = (A["bn"] >= 5) & ((A["ratio"] >= 2) | (A["ratio"] <= 0.5))
    A["slip"] = A["far"] & A["ratio"].map(lambda r: any(abs(r / 10.0 ** k - 1) <= 0.03 for k in (1, 2, -1, -2)))
    S0 = A[(A["proof_type"].isin(SHELF)) & (~A["disc"]) & (~A["far"])]
    fr0 = (A["cc"] == "FR") & (A["currency"] == "EUR") & (A["date"] >= STUDY_FROM)
    funnel.append({"step": "tag or receipt proof (not loyalty export or shop import)", "rows": int(A["proof_type"].isin(SHELF).sum()), "dropped": len(A) - int(A["proof_type"].isin(SHELF).sum())})
    c1 = A["proof_type"].isin(SHELF) & ~A["disc"]
    funnel.append({"step": "not a promotion", "rows": int(c1.sum()), "dropped": int(A["proof_type"].isin(SHELF).sum()) - int(c1.sum())})
    c2 = c1 & ~A["far"]
    funnel.append({"step": "not a far outlier", "rows": int(c2.sum()), "dropped": int(c1.sum()) - int(c2.sum())})
    c3 = c2 & fr0
    funnel.append({"step": f"France, euro, {STUDY_FROM} on = study sample S", "rows": int(c3.sum()), "dropped": int(c2.sum()) - int(c3.sum())})
    fun = pd.DataFrame(funnel); fun["share_of_all"] = fun["rows"] / N
    save("audit", fun)

    # ---------- data-quality traps ----------
    created = raw["created"].str[:10]
    prod = raw[(raw["type"] == "PRODUCT") & (raw["product_code"] != "")]
    dq = []

    def add(name, n, denom=N, note=""):
        dq.append({"check": name, "rows": int(n), "share": n / denom, "denominator": int(denom), "note": note})

    add("raw produce with no barcode (priced per kg or per unit)", (raw["type"] == "CATEGORY").sum())
    add("price of zero or less", (raw["p"] <= 0).sum()); add("currency missing", (raw["currency"] == "").sum()); add("date missing", (raw["date"] == "").sum())
    add("no proof attached", (raw["proof_id"] == "").sum())
    add(f"dated before {FIRST_OK}", ((raw["date"] != "") & (raw["date"] < FIRST_OK)).sum())
    add("dated after the pull date", (raw["date"] > PULL).sum())
    dd = pd.to_datetime(raw["date"].where(raw["date"] != ""), errors="coerce"); cd = pd.to_datetime(created.where(created != ""), errors="coerce")
    add("date more than 1 day after the row was entered", ((dd - cd).dt.days > 1).sum())
    both = (raw["proof_currency"] != "") & (raw["currency"] != "")
    add("proof currency differs from price currency", (both & (raw["proof_currency"] != raw["currency"])).sum(), both.sum(), "denominator: rows with both currencies")
    has_exp = exp.notna() & (raw["currency"] != "")
    add("currency is not the store country's own", (has_exp & (exp != raw["currency"])).sum(), has_exp.sum(), "denominator: rows in a country with a known currency")
    add("exact duplicates (same barcode, store, date, price, currency): surplus rows", len(pre) - len(A), len(pre), "denominator: rows reaching the dedup step")
    dups = pre[pre.duplicated(subset=dupkey, keep=False)].sort_values("id")
    first_proof = dups.groupby(dupkey)["proof_id"].transform("first")
    same_proof = int((dups.duplicated(subset=dupkey, keep="first") & (dups["proof_id"] == first_proof)).sum())
    add("  of which entered twice from the same proof", same_proof, len(pre) - len(A), "denominator: surplus rows")
    add("barcode length not 8, 12, 13 or 14", (~prod["product_code"].str.len().isin(EXPECT_LEN)).sum(), len(prod), "denominator: barcode rows")
    add("loyalty-card export (GDPR_REQUEST) or shop import (SHOP_IMPORT) proofs", raw["proof_type"].isin(["GDPR_REQUEST", "SHOP_IMPORT"]).sum())
    add("discounted but no regular price recorded", (raw["disc"] & raw["pwd"].isna()).sum(), raw["disc"].sum(), "denominator: discounted rows")
    add("discounted but regular price not above the price", (raw["disc"] & (raw["pwd"] <= raw["p"])).sum(), raw["disc"].sum(), "denominator: discounted rows")
    add("barcodes with >= 5 prices (the only ones whose prices can be checked)", (A["bn"] >= 5).sum(), len(A), "denominator: cleaned rows A")
    add("suspect prices (>= 2x or <= 1/2 of the barcode median)", A["suspect"].sum(), (A["bn"] >= 5).sum(), "denominator: cleaned rows of barcodes with >= 5 prices")
    add("far outliers (>= 3x or <= 1/3 of the barcode median), dropped from the study", A["far"].sum(), (A["bn"] >= 5).sum(), "denominator: cleaned rows of barcodes with >= 5 prices")
    add("  of which a decimal slip (ratio 10, 100, 1/10 or 1/100 within 3%)", A["slip"].sum(), A["far"].sum(), "denominator: far outliers")
    aprod = A.copy(); aprod["k"] = aprod["product_code"].map(key)
    off = pd.read_csv(D / "off_products.csv.gz", dtype=str, keep_default_na=False)
    off["k"] = off["code"].map(key); off = off.drop_duplicates("k").set_index("k")
    def parse_size(s):
        m = SIZE_RX.match(s)
        return float(m.group(1).replace(",", ".")) * SIZE_F[m.group(2).lower()] if m else np.nan
    off["size"] = off["quantity"].map(parse_size); off.loc[off["size"] <= 0, "size"] = np.nan
    off["leaf"] = off["categories_tags"].map(lambda s: [t for t in s.split(",") if t][-1] if s else "")
    bar = aprod.drop_duplicates("k")[["k"]]
    add("barcodes of A not found in the Open Food Facts dump", (~bar["k"].isin(off.index)).sum(), len(bar), "denominator: distinct barcodes in A")
    inoff = bar[bar["k"].isin(off.index)]
    add("barcodes found but pack size not parseable as g, kg, ml, cl, dl, l", off.loc[inoff["k"], "size"].isna().sum(), len(bar), "denominator: distinct barcodes in A (multipacks '6 x 33 cl' and counts '10 pcs' are not parsed)")
    save("dq", pd.DataFrame(dq))

    # ---------- coverage ----------
    cc = A.groupby("cc").agg(prices=("id", "size"), stores=("store", "nunique"), barcodes=("product_code", "nunique")).sort_values("prices", ascending=False)
    cc["share"] = cc["prices"] / len(A); save("coverage_country", cc.head(12).reset_index().rename(columns={"cc": "country"}))
    cm = A[A["month"] >= "2024-01"].groupby("month").agg(prices=("id", "size"), stores=("store", "nunique"), barcodes=("product_code", "nunique")).reset_index()
    save("coverage_month", cm)
    own = raw.groupby("owner").size().sort_values(ascending=False); stor = raw.groupby("store").size().sort_values(ascending=False)
    conc = [("prices_in_dump", N), ("contributors", len(own)), ("stores_with_a_price", len(stor)), ("barcodes", raw.loc[raw["product_code"] != "", "product_code"].nunique()),
            ("top1_contributor_share", own.iloc[0] / N), ("top10_contributors_share", own.iloc[:10].sum() / N), ("top1pct_contributors_share", own.iloc[:math.ceil(0.01 * len(own))].sum() / N), ("top1pct_contributors_n", math.ceil(0.01 * len(own))),
            ("top10_stores_share", stor.iloc[:10].sum() / N), ("top100_stores_share", stor.iloc[:100].sum() / N)]
    for k, v in raw["proof_type"].replace("", "none").value_counts().items():
        conc.append((f"proof_share_{k}", v / N))
    for k, v in raw["location_osm_tag_value"].replace("", "unknown").value_counts().head(3).items():
        conc.append((f"store_type_share_{k}", v / N))
    biggest = A.groupby("month").size().sort_values(ascending=False)
    conc += [("busiest_month", biggest.index[0]), ("busiest_month_prices", int(biggest.iloc[0])), ("busiest_month_share_of_A", biggest.iloc[0] / len(A)), ("prices_2026_share_of_A", (A["date"] >= "2026-01-01").mean()),
             ("share_FR_in_A", (A["cc"] == "FR").mean())]
    save("concentration", pd.DataFrame(conc, columns=["key", "value"]))

    # ---------- dispersion ----------
    def cells_of(sample, store_quarter=True):
        u = sample.groupby(["product_code", "quarter", "store"], sort=False)["lp"].mean().reset_index()
        n = u.groupby(["product_code", "quarter"])["lp"].transform("size")
        return u[n >= 2]

    def spread_stats(u):
        g = u.groupby(["product_code", "quarter"])["lp"]
        sp = np.expm1(g.max() - g.min()); sd = g.std(ddof=1); ns = g.size()
        return sp, sd, ns

    FR = A[(A["cc"] == "FR") & (A["currency"] == "EUR") & (A["date"] >= STUDY_FROM)]
    shelf = FR[FR["proof_type"].isin(SHELF)]
    specs = {"S: tag or receipt, no promotions, no far outliers": shelf[(~shelf["disc"]) & (~shelf["far"])],
             "far outliers kept": shelf[~shelf["disc"]],
             "promotions kept (price paid)": shelf[~shelf["far"]],
             "price tags only": shelf[(shelf["proof_type"] == "PRICE_TAG") & (~shelf["disc"]) & (~shelf["far"])],
             "receipts only": shelf[(shelf["proof_type"] == "RECEIPT") & (~shelf["disc"]) & (~shelf["far"])]}
    rows = []
    for name, smp in specs.items():
        sp, sd, ns = spread_stats(cells_of(smp))
        rows.append({"spec": name, "cells": len(sp), "unit_rows": int(ns.sum()), "median_spread": sp.median(), "mean_spread": sp.mean(), "p25_spread": np.percentile(sp, 25), "p75_spread": np.percentile(sp, 75),
                     "p90_spread": np.percentile(sp, 90), "median_sd_log": sd.median(), "share_cells_over_50pct": (sp > 0.5).mean(), "share_cells_identical": (sp < 1e-9).mean()})
    save("dispersion_summary", pd.DataFrame(rows))
    sp, sd, ns = spread_stats(cells_of(specs["S: tag or receipt, no promotions, no far outliers"]))
    byn = pd.DataFrame({"sp": sp, "sd": sd, "n": ns.clip(upper=4)}).groupby("n").agg(cells=("sp", "size"), median_spread=("sp", "median"), median_sd_log=("sd", "median")).reset_index()
    byn["stores"] = byn["n"].map(lambda k: "2" if k == 2 else "3" if k == 3 else "4+"); save("dispersion_by_stores", byn.drop(columns="n"))
    # by country, each in its own currency, same rules
    rows = []
    for c, cur in EXPECT.items():
        s = A[(A["cc"] == c) & (A["currency"] == cur) & (A["date"] >= STUDY_FROM) & A["proof_type"].isin(SHELF) & (~A["disc"]) & (~A["far"])]
        if len(s) == 0: continue
        sp, sd, ns = spread_stats(cells_of(s))
        if len(sp) >= 100:
            rows.append({"country": c, "currency": cur, "cells": len(sp), "unit_rows": int(ns.sum()), "median_spread": sp.median(), "median_sd_log": sd.median(), "p90_spread": np.percentile(sp, 90)})
    save("dispersion_country", pd.DataFrame(rows).sort_values(["cells", "country"], ascending=[False, True]))

    # ---------- ladder A ----------
    S = specs["S: tag or receipt, no promotions, no far outliers"].copy()
    u = cells_of(S).reset_index(drop=True); u["cell"] = u["product_code"] + "|" + u["quarter"]
    sc = pd.factorize(u["store"])[0]; cc_ = pd.factorize(u["cell"])[0]; ns_, nc_ = sc.max() + 1, cc_.max() + 1
    gmat = sparse.csr_matrix((np.ones(len(u)), (sc, ns_ + cc_)), shape=(ns_ + nc_, ns_ + nc_)); _, lab = connected_components(gmat + gmat.T, directed=False)
    big = np.bincount(lab[sc]).argmax(); u = u[lab[sc] == big].reset_index(drop=True)
    y = u["lp"].to_numpy(); cellmean = u.groupby("cell")["lp"].transform("mean").to_numpy(); ssw = float(((y - cellmean) ** 2).sum())
    res, (eff_cell, eff_store), _ = fe_fit(y, [u["cell"].to_numpy(), u["store"].to_numpy()])
    rss = float((res ** 2).sum()); se = u["store"].map(eff_store).to_numpy(); se = se - se.mean()
    u["adj"] = y - se; gq_ = u.groupby("cell")["adj"]; sp_adj = np.expm1(gq_.max() - gq_.min()); gq0 = u.groupby("cell")["lp"]; sp_raw = np.expm1(gq0.max() - gq0.min())
    la = [("median_spread_raw", float(sp_raw.median())), ("median_spread_after_store", float(sp_adj.median())), ("p90_spread_raw", float(np.percentile(sp_raw, 90))), ("p90_spread_after_store", float(np.percentile(sp_adj, 90))),
          ("units_in_largest_component", len(u)), ("units_in_all_cells", len(cells_of(S))), ("cells", u["cell"].nunique()), ("stores", u["store"].nunique()), ("rms_within_cell", math.sqrt(ssw / len(u))),
          ("rms_after_store", math.sqrt(rss / len(u))), ("share_removed_by_store", 1 - rss / ssw), ("sd_store_effect", float(se.std(ddof=0)))]
    save("ladder_a", pd.DataFrame(la, columns=["key", "value"]))
    u["chain"] = u["store"].map(S.drop_duplicates("store").set_index("store")["location_osm_display_name"].map(chain_of)); u["se"] = se
    ch = u.groupby("chain").agg(stores=("store", "nunique"), units=("se", "size"), mean_effect=("se", "mean")).reset_index()
    ch["effect_pct"] = np.expm1(ch["mean_effect"]); ch = ch[(ch["stores"] >= 3) & (ch["units"] >= 300) & (ch["chain"] != "other")].sort_values(["effect_pct", "chain"])
    save("chains", ch)

    # ---------- ladder B ----------
    Sk = S.copy(); Sk["k"] = Sk["product_code"].map(key); Sk["size"] = Sk["k"].map(off["size"]); Sk["leaf"] = Sk["k"].map(off["leaf"])
    Sk = Sk[Sk["size"].notna() & (Sk["leaf"] != "")]
    nb = Sk.groupby("leaf")["k"].nunique(); keep = nb[nb >= MIN_CAT_BARCODES].index; Sk = Sk[Sk["leaf"].isin(keep)]
    ub = Sk.groupby(["k", "quarter", "store"], sort=False).agg(lp=("lp", "mean"), size=("size", "first"), leaf=("leaf", "first")).reset_index()
    y = ub["lp"].to_numpy(); x = np.log(ub["size"].to_numpy()); tss = float(((y - y.mean()) ** 2).sum()); n_ = len(ub)
    def rr(res): return float((res ** 2).sum())
    r1, _, _ = fe_fit(y, [ub["leaf"].to_numpy()])
    r2, _, b2 = fe_fit(y, [ub["leaf"].to_numpy()], extra=x.reshape(-1, 1))
    r3, _, b3 = fe_fit(y, [ub["leaf"].to_numpy(), ub["store"].to_numpy()], extra=x.reshape(-1, 1))
    yu = y - x + math.log(100.0); ru, _, _ = fe_fit(yu, [ub["leaf"].to_numpy()]); tssu = float(((yu - yu.mean()) ** 2).sum())
    lb = [{"model": "price: nothing", "r2": 0.0, "rms": math.sqrt(tss / n_), "slope_log_size": np.nan}, {"model": "price: category", "r2": 1 - rr(r1) / tss, "rms": math.sqrt(rr(r1) / n_), "slope_log_size": np.nan},
          {"model": "price: category + size", "r2": 1 - rr(r2) / tss, "rms": math.sqrt(rr(r2) / n_), "slope_log_size": b2[0]}, {"model": "price: category + size + store", "r2": 1 - rr(r3) / tss, "rms": math.sqrt(rr(r3) / n_), "slope_log_size": b3[0]},
          {"model": "price per 100 g or ml: nothing", "r2": 0.0, "rms": math.sqrt(tssu / n_), "slope_log_size": np.nan}, {"model": "price per 100 g or ml: category", "r2": 1 - rr(ru) / tssu, "rms": math.sqrt(rr(ru) / n_), "slope_log_size": np.nan}]
    lbd = pd.DataFrame(lb); ncat, nst = ub["leaf"].nunique(), ub["store"].nunique()
    kpar = {"price: nothing": 0, "price: category": ncat - 1, "price: category + size": ncat, "price: category + size + store": ncat + nst - 1, "price per 100 g or ml: nothing": 0, "price per 100 g or ml: category": ncat - 1}
    lbd["adj_r2"] = [1 - (1 - r) * (n_ - 1) / (n_ - 1 - kpar[m]) for m, r in zip(lbd["model"], lbd["r2"])]; lbd["units"] = n_; lbd["barcodes"] = ub["k"].nunique(); lbd["categories"] = ub["leaf"].nunique(); lbd["stores"] = ub["store"].nunique(); save("ladder_b", lbd)
    s_cov = S.copy(); s_cov["k"] = s_cov["product_code"].map(key); bc = s_cov.drop_duplicates("k")
    cov = [("study_barcodes", len(bc)), ("in_off_dump", int(bc["k"].isin(off.index).sum())), ("with_parsed_size", int(bc["k"].map(off["size"]).notna().sum())), ("with_size_and_category", int((bc["k"].map(off["size"]).notna() & (bc["k"].map(off["leaf"]) != "")).sum())),
           ("ladder_b_barcodes", int(ub["k"].nunique())), ("ladder_b_min_barcodes_per_category", MIN_CAT_BARCODES)]
    save("size_coverage", pd.DataFrame(cov, columns=["key", "value"]))

    # ---------- proof effect ----------
    Pf = FR[FR["proof_type"].isin(list(SHELF) + ["GDPR_REQUEST", "SHOP_IMPORT"]) & (~FR["far"])]
    rows = []
    for label, smp in (("no promotions", Pf[~Pf["disc"]]), ("promotions kept (price paid)", Pf)):
        smp = smp.copy(); smp["cell"] = smp["product_code"] + "|" + smp["quarter"]
        n = smp.groupby("cell")["store"].transform("nunique"); smp = smp[n >= 2].reset_index(drop=True)
        yy = smp["lp"].to_numpy(); Xd = np.column_stack([(smp["proof_type"] == t).to_numpy().astype(float) for t in ("RECEIPT", "GDPR_REQUEST", "SHOP_IMPORT")])
        gg = [smp["cell"].to_numpy(), smp["store"].to_numpy()]
        ry, _, _ = fe_fit(yy, gg); rx = np.column_stack([fe_fit(Xd[:, j], gg)[0] for j in range(3)])
        XtX = rx.T @ rx; b = np.linalg.solve(XtX, rx.T @ ry); e = ry - rx @ b
        cl = pd.factorize(smp["store"])[0]; sc_ = np.zeros((cl.max() + 1, 3)); np.add.at(sc_, cl, rx * e[:, None]); inv = np.linalg.inv(XtX); V = inv @ (sc_.T @ sc_) @ inv
        for j, t in enumerate(("RECEIPT", "GDPR_REQUEST", "SHOP_IMPORT")):
            rows.append({"sample": label, "proof": t, "rows": int(Xd[:, j].sum()), "total_rows": len(smp), "effect_pct": math.expm1(b[j]), "coef": b[j], "se_store_clustered": math.sqrt(V[j, j])})
    save("proof_effect", pd.DataFrame(rows))

    # ---------- discounts ----------
    dd_ = A[A["disc"]]; dep = 1 - dd_["p"] / dd_["pwd"]; dep = dep[dep.notna() & (dd_["pwd"] > dd_["p"])]
    disc = [("rows_A", len(A)), ("discounted_share", A["disc"].mean()), ("discounted_with_regular_price", int(len(dep))), ("median_depth", dep.median()), ("p90_depth", np.percentile(dep, 90)), ("mean_depth", dep.mean())]
    for t in ("PRICE_TAG", "RECEIPT", "GDPR_REQUEST", "SHOP_IMPORT"):
        disc.append((f"discounted_share_{t}", A.loc[A["proof_type"] == t, "disc"].mean()))
    save("discounts", pd.DataFrame(disc, columns=["key", "value"]))

    # ---------- inflation: same barcode, same store, about a year apart ----------
    H = json.loads((D / "hicp_fr.json").read_text()); ti = H["dimension"]["time"]["category"]["index"]; ci = H["dimension"]["coicop"]["category"]["index"]; nt = len(ti)
    hicp = {(c, t): H["value"][str(ci[c] * nt + i)] for c in ci for t, i in ti.items() if str(ci[c] * nt + i) in H["value"]}
    def hq(c, q):
        y, qq = int(q[:4]), int(q[-1]); ms = [f"{y}-{3 * (qq - 1) + k:02d}" for k in (1, 2, 3)]; ms0 = [f"{y - 1}-{3 * (qq - 1) + k:02d}" for k in (1, 2, 3)]
        if all((c, m) in hicp for m in ms + ms0): return sum(hicp[(c, m)] for m in ms) / sum(hicp[(c, m)] for m in ms0) - 1
        return np.nan
    Iq = A[(A["cc"] == "FR") & (A["currency"] == "EUR") & (A["date"] >= "2024-01-01") & A["proof_type"].isin(SHELF) & (~A["disc"]) & (~A["far"])]
    daily = Iq.groupby(["product_code", "store", "date"], sort=False)["lp"].median().reset_index(); daily["day"] = (pd.to_datetime(daily["date"]) - pd.Timestamp("2000-01-01")).dt.days
    pairs = []
    for (pc, st), grp in daily.groupby(["product_code", "store"], sort=False):
        if len(grp) < 2: continue
        days = grp["day"].to_numpy(); lps = grp["lp"].to_numpy(); dates = grp["date"].to_numpy()
        for j in range(len(days)):
            gap = days[j] - days[:j]; m = (gap >= 300) & (gap <= 430)
            if m.any():
                cand = np.where(m)[0]; best = cand[np.lexsort((cand, np.abs(gap[cand] - 365)))[0]]
                pairs.append((st, dates[j], lps[j] - lps[best], int(gap[best])))
    P = pd.DataFrame(pairs, columns=["store", "date1", "dlog", "gap"]); P["quarter"] = P["date1"].map(quarter)
    rows = []
    for q, gq in sorted(P.groupby("quarter"), key=lambda t: t[0]):
        t = gq[gq["dlog"].abs() <= math.log(3)]
        mu = t["dlog"].mean(); cl = t.groupby("store")["dlog"].apply(lambda v: (v - mu).sum()); se = math.sqrt((cl ** 2).sum()) / len(t)
        rows.append({"quarter": q, "pairs": len(gq), "stores": gq["store"].nunique(), "pairs_kept": len(t), "median_change": math.expm1(gq["dlog"].median()), "mean_change": math.expm1(mu), "se_mean_log": se,
                     "share_unchanged": float((gq["dlog"].abs() < 1e-9).mean()), "share_up": float((gq["dlog"] > 1e-9).mean()), "hicp_food_yoy": hq("CP01", q), "hicp_all_yoy": hq("CP00", q)})
    save("inflation", pd.DataFrame(rows))

    # ---------- cross-country (euro only) ----------
    X = A[A["currency"].eq("EUR") & A["cc"].isin(EURO) & (A["date"] >= STUDY_FROM) & A["proof_type"].isin(SHELF) & (~A["disc"]) & (~A["far"])]
    med = X.groupby(["product_code", "cc"])["lp"].median().unstack()
    rows = []
    for c in EURO:
        if c == "FR" or c not in med.columns: continue
        both = med[["FR", c]].dropna()
        if len(both) >= 15:
            d = both[c] - both["FR"]
            rows.append({"country": c, "barcodes": len(both), "median_ratio": math.exp(d.median()), "p25_ratio": math.exp(np.percentile(d, 25)), "p75_ratio": math.exp(np.percentile(d, 75)), "share_cheaper_than_france": float((d < 0).mean())})
    save("cross_country", pd.DataFrame(rows).sort_values(["barcodes", "country"], ascending=[False, True]))

    # ---------- raw produce: per kg or per unit ----------
    pr = raw[(raw["type"] == "CATEGORY") & (raw["currency"] == "EUR") & (raw["cc"] == "FR") & (raw["date"] >= STUDY_FROM) & (raw["p"] > 0) & raw["price_per"].isin(["KILOGRAM", "UNIT"])]
    pt = pr.pivot_table(index="category_tag", columns="price_per", values="p", aggfunc=["median", "size"])
    pt.columns = [f"{a}_{b}" for a, b in pt.columns]; pt = pt.dropna()
    pt = pt[(pt["size_KILOGRAM"] >= 10) & (pt["size_UNIT"] >= 10)]; pt["rows"] = pt["size_KILOGRAM"] + pt["size_UNIT"]
    pt = pt.reset_index().sort_values(["rows", "category_tag"], ascending=[False, True]).rename(columns={"median_KILOGRAM": "median_per_kg", "median_UNIT": "median_per_unit", "size_KILOGRAM": "rows_per_kg", "size_UNIT": "rows_per_unit"})
    pt["kg_over_unit"] = pt["median_per_kg"] / pt["median_per_unit"]; save("produce_units", pt.head(8))
    allp = raw[(raw["type"] == "CATEGORY") & (raw["p"] > 0)]
    save("produce_share", pd.DataFrame([("category_rows", int((raw["type"] == "CATEGORY").sum())), ("per_kilogram_share", float((allp["price_per"] == "KILOGRAM").mean())), ("per_unit_share", float((allp["price_per"] == "UNIT").mean())),
                                         ("categories_with_both_units", len(pt))], columns=["key", "value"]))
    print("analyze.py done")


if __name__ == "__main__":
    main()
