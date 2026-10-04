"""Step 5: the project's check. Recompute with different code, in plain Python straight from the downloaded files (csv, gzip, json, math and loops: no pandas, numpy or scipy), and compare with results/:

    ~/tools/venv/bin/python check.py

Recomputed: the cleaning funnel and every data-quality count, coverage and concentration, the same-barcode dispersion tables (five variants, by number of stores, by country), the store ladder (cell + store least squares solved by
alternating projections to 1e-13 instead of the sparse LSQR used in analyze.py; chain effects), the category/size/store ladder, the proof-type effects with store-clustered errors, discounts, the year-on-year matched pairs against HICP,
the France-versus-euro-neighbours comparison and the raw-produce units. Counts and shares are compared exactly (1e-9); the three least-squares families (ladder A, ladder B, proof effect) at a relative 1e-6, because LSQR and alternating
projections stop at different tiny residuals. Not independently rebuilt: the parquet -> csv conversion in download.py (pyarrow; row count asserted there).
"""
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
PULL = json.loads((R / "pull.json").read_text())["pulled"]
FIRST_OK, STUDY_FROM, FAR, MINCAT = "2018-01-01", "2025-01-01", 3.0, 30
SHELF = ("PRICE_TAG", "RECEIPT")
EXPECT = {c: "EUR" for c in "FR DE BE IT ES AT FI HR NL PT IE LU GR SK SI LT LV EE MT CY MC".split()}
EXPECT.update({"US": "USD", "NO": "NOK", "SE": "SEK", "GB": "GBP", "PL": "PLN", "DK": "DKK", "CH": "CHF", "CA": "CAD", "JP": "JPY", "RU": "RUB", "IN": "INR", "UA": "UAH", "MX": "MXN", "IL": "ILS", "CZ": "CZK",
               "HU": "HUF", "RO": "RON", "BG": "BGN", "AU": "AUD", "NZ": "NZD", "BR": "BRL", "TR": "TRY"})
EURO = [c for c, v in EXPECT.items() if v == "EUR"]
CHAINS = [("E.Leclerc", ["leclerc"]), ("Auchan", ["auchan"]), ("Intermarché", ["intermarché", "intermarche"]), ("Carrefour", ["carrefour"]), ("Système U", ["super u", "hyper u", "u express", "système u", "systeme u", "marché u"]),
          ("Monoprix", ["monoprix"]), ("Franprix", ["franprix"]), ("Lidl", ["lidl"]), ("Aldi", ["aldi"]), ("Biocoop", ["biocoop"]), ("Naturalia", ["naturalia"]), ("Casino", ["casino"]), ("Picard", ["picard"]),
          ("Grand Frais", ["grand frais"]), ("Cora", ["cora"]), ("Match", ["match"]), ("G20", ["g20"]), ("La Vie Claire", ["la vie claire"])]
import re
SIZE_RX = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(g|gr|gram|grams|kg|ml|cl|dl|l|litre|liter|litres|liters)\s*(?:e|℮)?\s*$", re.I)
SIZE_F = {"g": 1, "gr": 1, "gram": 1, "grams": 1, "kg": 1000, "ml": 1, "cl": 10, "dl": 100, "l": 1000, "litre": 1000, "liter": 1000, "litres": 1000, "liters": 1000}
n_ok = 0


def same(label, got, want, rel=1e-9):
    global n_ok
    n_ok += 1
    want = float(want)
    assert abs(got - want) <= rel * abs(want) + 1e-12, f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv", encoding="utf-8")))


def kv(name):
    return {r["key"]: r["value"] for r in table(name)}


def median(v):
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def pct(v, q):
    s = sorted(v); k = (len(s) - 1) * q / 100.0; f = math.floor(k); c = math.ceil(k)
    return s[f] + (s[c] - s[f]) * (k - f)


def mean(v):
    return sum(v) / len(v)


def sd1(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def key(c):
    return c.lstrip("0")


def quarter(d):
    return f"{d[:4]}-Q{(int(d[5:7]) - 1) // 3 + 1}"


def chain_of(display):
    first = display.split(",")[0].lower()
    for name, kws in CHAINS:
        if any(re.search(r"\b" + re.escape(k) + r"\b", first) for k in kws):
            return name
    return "other"


def ap(vals, groups, tol=1e-13, maxit=50000):
    """Residuals of each vector in vals after removing group means for every grouping in groups (alternating projections); also the summed means removed in the LAST grouping's steps."""
    out = []
    for v in vals:
        r = list(v); n = len(r); acc = [defaultdict(float) for _ in groups]
        for it in range(maxit):
            big = 0.0
            for gi, g in enumerate(groups):
                s = defaultdict(float); c = defaultdict(int)
                for i in range(n): s[g[i]] += r[i]; c[g[i]] += 1
                m = {k: s[k] / c[k] for k in s}
                for k, x in m.items():
                    acc[gi][k] += x
                    if abs(x) > big: big = abs(x)
                for i in range(n): r[i] -= m[g[i]]
            if big < tol: break
        else:
            raise RuntimeError("alternating projections did not converge")
        out.append((r, acc))
    return out


def tbl_eq(name, rows, keyf, fields, rel=1e-9):
    """compare results/name.csv with recomputed rows (dict keyed by keyf) on numeric fields"""
    got = table(name)
    assert len(got) == len(rows), f"{name}: {len(got)} rows in results, {len(rows)} recounted"
    for r in got:
        k = keyf(r); assert k in rows, f"{name}: {k} not recounted"
        for f in fields:
            same(f"{name}/{k}/{f}", rows[k][f], r[f] if r[f] != "" else "nan", rel) if not (isinstance(rows[k][f], float) and math.isnan(rows[k][f])) else None


def main():
    # ---------------- load ----------------
    raw = []
    with gzip.open(D / "prices.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["id"] = int(r["id"]); r["p"] = float(r["price"]) if r["price"] != "" else None; r["pwd"] = float(r["price_without_discount"]) if r["price_without_discount"] != "" else None
            r["disc"] = r["price_is_discounted"] == "True"; r["cc"] = r["location_osm_address_country_code"]; r["store"] = r["location_id"]
            raw.append(r)
    N = len(raw)
    # ---------------- funnel ----------------
    def stage(rows, cond):
        return [r for r in rows if cond(r)]
    fun = [("every row in the dump", N)]
    s1 = stage(raw, lambda r: r["type"] == "PRODUCT" and r["product_code"] != ""); fun.append(("a barcode product (drop raw produce with no barcode)", len(s1)))
    s2 = stage(s1, lambda r: r["p"] is not None and r["p"] > 0); fun.append(("price above zero", len(s2)))
    s3 = stage(s2, lambda r: r["currency"] != ""); fun.append(("currency present", len(s3)))
    s4 = stage(s3, lambda r: r["date"] != ""); fun.append(("date present", len(s4)))
    s5 = stage(s4, lambda r: FIRST_OK <= r["date"] <= PULL); fun.append((f"date from {FIRST_OK} to the pull date", len(s5)))
    s6 = stage(s5, lambda r: r["cc"] != ""); fun.append(("store has a country", len(s6)))
    s7 = stage(s6, lambda r: r["cc"] not in EXPECT or EXPECT[r["cc"]] == r["currency"]); fun.append(("currency is the country's own", len(s7)))
    seen = {}; A = []; surplus = []
    for r in sorted(s7, key=lambda r: r["id"]):
        k = (r["product_code"], r["store"], r["date"], r["p"], r["currency"])
        if k in seen: surplus.append((r, seen[k]))
        else: seen[k] = r["proof_id"]; A.append(r)
    fun.append(("one row per (barcode, store, date, price, currency)", len(A)))
    for r in A:
        r["month"] = r["date"][:7]; r["quarter"] = quarter(r["date"]); r["lp"] = math.log(r["p"])
    by = defaultdict(list)
    for r in A: by[(r["product_code"], r["currency"])].append(r["p"])
    med = {k: median(v) for k, v in by.items()}
    for r in A:
        n = len(by[(r["product_code"], r["currency"])]); ratio = r["p"] / med[(r["product_code"], r["currency"])]
        r["bn"] = n; r["far"] = n >= 5 and (ratio >= FAR or ratio <= 1 / FAR); r["suspect"] = n >= 5 and (ratio >= 2 or ratio <= 0.5)
        r["slip"] = r["far"] and any(abs(ratio / 10.0 ** k - 1) <= 0.03 for k in (1, 2, -1, -2))
    shelf_n = sum(1 for r in A if r["proof_type"] in SHELF); c1 = [r for r in A if r["proof_type"] in SHELF and not r["disc"]]; c2 = [r for r in c1 if not r["far"]]
    c3 = [r for r in c2 if r["cc"] == "FR" and r["currency"] == "EUR" and r["date"] >= STUDY_FROM]
    fun += [("tag or receipt proof (not loyalty export or shop import)", shelf_n), ("not a promotion", len(c1)), ("not a far outlier", len(c2)), (f"France, euro, {STUDY_FROM} on = study sample S", len(c3))]
    got = table("audit"); assert len(got) == len(fun)
    prev = None
    for (name, n), r in zip(fun, got):
        assert r["step"] == name, (r["step"], name); same(f"audit/{name}/rows", n, r["rows"]); same(f"audit/{name}/share", n / N, r["share_of_all"])
        if prev is not None and name not in ("tag or receipt proof (not loyalty export or shop import)", "not a promotion", "not a far outlier") and not name.startswith("France"): same(f"audit/{name}/dropped", prev - n, r["dropped"])
        prev = n
    # ---------------- data-quality traps ----------------
    off = {}
    with gzip.open(D / "off_products.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            k = key(r["code"])
            if k not in off:
                m = SIZE_RX.match(r["quantity"]); sz = float(m.group(1).replace(",", ".")) * SIZE_F[m.group(2).lower()] if m else None
                tags = [t for t in r["categories_tags"].split(",") if t]; off[k] = {"size": sz if sz and sz > 0 else None, "leaf": tags[-1] if tags else ""}
    def cdate(r):
        try: return date.fromisoformat(r["date"]), date.fromisoformat(r["created"][:10])
        except ValueError: return None
    late = 0
    for r in raw:
        if r["date"] != "" and r["created"] != "":
            d1, d2 = cdate(r)
            if (d1 - d2).days > 1: late += 1
    both = [r for r in raw if r["proof_currency"] != "" and r["currency"] != ""]
    has_exp = [r for r in raw if r["cc"] in EXPECT and r["currency"] != ""]
    prod = [r for r in raw if r["type"] == "PRODUCT" and r["product_code"] != ""]
    disc_rows = [r for r in raw if r["disc"]]
    bars = {key(r["product_code"]) for r in A}; ck = A and [r for r in A if r["bn"] >= 5]
    same_proof = sum(1 for r, first in surplus if r["proof_id"] == first)
    exp_dq = {
        "raw produce with no barcode (priced per kg or per unit)": (sum(1 for r in raw if r["type"] == "CATEGORY"), N),
        "price of zero or less": (sum(1 for r in raw if r["p"] is not None and r["p"] <= 0), N), "currency missing": (sum(1 for r in raw if r["currency"] == ""), N), "date missing": (sum(1 for r in raw if r["date"] == ""), N),
        "no proof attached": (sum(1 for r in raw if r["proof_id"] == ""), N), f"dated before {FIRST_OK}": (sum(1 for r in raw if r["date"] != "" and r["date"] < FIRST_OK), N),
        "dated after the pull date": (sum(1 for r in raw if r["date"] > PULL), N), "date more than 1 day after the row was entered": (late, N),
        "proof currency differs from price currency": (sum(1 for r in both if r["proof_currency"] != r["currency"]), len(both)),
        "currency is not the store country's own": (sum(1 for r in has_exp if EXPECT[r["cc"]] != r["currency"]), len(has_exp)),
        "exact duplicates (same barcode, store, date, price, currency): surplus rows": (len(surplus), len(s7)), "  of which entered twice from the same proof": (same_proof, len(surplus)),
        "barcode length not 8, 12, 13 or 14": (sum(1 for r in prod if len(r["product_code"]) not in (8, 12, 13, 14)), len(prod)),
        "loyalty-card export (GDPR_REQUEST) or shop import (SHOP_IMPORT) proofs": (sum(1 for r in raw if r["proof_type"] in ("GDPR_REQUEST", "SHOP_IMPORT")), N),
        "discounted but no regular price recorded": (sum(1 for r in disc_rows if r["pwd"] is None), len(disc_rows)), "discounted but regular price not above the price": (sum(1 for r in disc_rows if r["pwd"] is not None and r["pwd"] <= r["p"]), len(disc_rows)),
        "barcodes with >= 5 prices (the only ones whose prices can be checked)": (len(ck), len(A)),
        "suspect prices (>= 2x or <= 1/2 of the barcode median)": (sum(1 for r in A if r["suspect"]), len(ck)), "far outliers (>= 3x or <= 1/3 of the barcode median), dropped from the study": (sum(1 for r in A if r["far"]), len(ck)),
        "  of which a decimal slip (ratio 10, 100, 1/10 or 1/100 within 3%)": (sum(1 for r in A if r["slip"]), sum(1 for r in A if r["far"])),
        "barcodes of A not found in the Open Food Facts dump": (sum(1 for b in bars if b not in off), len(bars)),
        "barcodes found but pack size not parseable as g, kg, ml, cl, dl, l": (sum(1 for b in bars if b in off and off[b]["size"] is None), len(bars))}
    got = table("dq"); assert len(got) == len(exp_dq)
    for r in got:
        n, den = exp_dq[r["check"]]; same(f"dq/{r['check']}/rows", n, r["rows"]); same(f"dq/{r['check']}/denominator", den, r["denominator"]); same(f"dq/{r['check']}/share", n / den, r["share"])
    # ---------------- coverage ----------------
    cc = defaultdict(lambda: [0, set(), set()])
    for r in A:
        c = cc[r["cc"]]; c[0] += 1; c[1].add(r["store"]); c[2].add(r["product_code"])
    got = table("coverage_country"); top = sorted(cc, key=lambda k: (-cc[k][0], k))[:12]
    assert [r["country"] for r in got] == top or {r["country"] for r in got} == set(top)
    for r in got:
        c = cc[r["country"]]; same(f"cov/{r['country']}/prices", c[0], r["prices"]); same(f"cov/{r['country']}/stores", len(c[1]), r["stores"]); same(f"cov/{r['country']}/barcodes", len(c[2]), r["barcodes"]); same(f"cov/{r['country']}/share", c[0] / len(A), r["share"])
    cm = defaultdict(lambda: [0, set(), set()])
    for r in A:
        if r["month"] >= "2024-01": c = cm[r["month"]]; c[0] += 1; c[1].add(r["store"]); c[2].add(r["product_code"])
    got = table("coverage_month"); assert len(got) == len(cm)
    for r in got:
        c = cm[r["month"]]; same(f"covm/{r['month']}/prices", c[0], r["prices"]); same(f"covm/{r['month']}/stores", len(c[1]), r["stores"]); same(f"covm/{r['month']}/barcodes", len(c[2]), r["barcodes"])
    own = defaultdict(int); stor = defaultdict(int)
    for r in raw: own[r["owner"]] += 1; stor[r["store"]] += 1
    ov = sorted(own.values(), reverse=True); sv = sorted(stor.values(), reverse=True); k1 = math.ceil(0.01 * len(ov))
    cg = kv("concentration")
    for k, v in (("prices_in_dump", N), ("contributors", len(ov)), ("stores_with_a_price", len(sv)), ("barcodes", len({r["product_code"] for r in raw if r["product_code"] != ""})), ("top1_contributor_share", ov[0] / N),
                 ("top10_contributors_share", sum(ov[:10]) / N), ("top1pct_contributors_share", sum(ov[:k1]) / N), ("top1pct_contributors_n", k1), ("top10_stores_share", sum(sv[:10]) / N), ("top100_stores_share", sum(sv[:100]) / N)):
        same(f"conc/{k}", v, cg[k])
    for t in ("PRICE_TAG", "RECEIPT", "GDPR_REQUEST", "SHOP_IMPORT"): same(f"conc/proof_{t}", sum(1 for r in raw if r["proof_type"] == t) / N, cg[f"proof_share_{t}"])
    same("conc/proof_none", sum(1 for r in raw if r["proof_type"] == "") / N, cg["proof_share_none"])
    st = defaultdict(int)
    for r in raw: st[r["location_osm_tag_value"] or "unknown"] += 1
    for k, v in sorted(st.items(), key=lambda t: -t[1])[:3]: same(f"conc/store_type_{k}", v / N, cg[f"store_type_share_{k}"])
    mc = defaultdict(int)
    for r in A: mc[r["month"]] += 1
    bm = max(mc, key=lambda k: mc[k]); assert cg["busiest_month"] == bm; same("conc/busiest_n", mc[bm], cg["busiest_month_prices"]); ow = defaultdict(int)
    for r in A:
        if r["month"] == bm: ow[r["owner"]] += 1
    same("conc/busiest_owner", max(ow.values()) / mc[bm], cg["busiest_month_top_contributor_share"]); same("conc/busiest_share", mc[bm] / len(A), cg["busiest_month_share_of_A"])
    same("conc/2026", sum(1 for r in A if r["date"] >= "2026-01-01") / len(A), cg["prices_2026_share_of_A"]); same("conc/frshare", sum(1 for r in A if r["cc"] == "FR") / len(A), cg["share_FR_in_A"])
    # ---------------- dispersion ----------------
    def units(sample):
        g = defaultdict(list)
        for r in sample: g[(r["product_code"], r["quarter"], r["store"])].append(r["lp"])
        u = {k: mean(v) for k, v in g.items()}; cells = defaultdict(dict)
        for (pc, q, s), v in u.items(): cells[(pc, q)][s] = v
        return {c: d for c, d in cells.items() if len(d) >= 2}

    def disp(cells):
        sp = []; sd = []; ns = []
        for d in cells.values():
            v = list(d.values()); sp.append(math.expm1(max(v) - min(v))); sd.append(sd1(v)); ns.append(len(v))
        return sp, sd, ns

    FRs = [r for r in A if r["cc"] == "FR" and r["currency"] == "EUR" and r["date"] >= STUDY_FROM and r["proof_type"] in SHELF]
    specs = {"S: tag or receipt, no promotions, no far outliers": [r for r in FRs if not r["disc"] and not r["far"]], "far outliers kept": [r for r in FRs if not r["disc"]], "promotions kept (price paid)": [r for r in FRs if not r["far"]],
             "price tags only": [r for r in FRs if r["proof_type"] == "PRICE_TAG" and not r["disc"] and not r["far"]], "receipts only": [r for r in FRs if r["proof_type"] == "RECEIPT" and not r["disc"] and not r["far"]]}
    rows = {}
    for name, smp in specs.items():
        sp, sd, ns = disp(units(smp))
        rows[name] = {"cells": len(sp), "unit_rows": sum(ns), "median_spread": median(sp), "mean_spread": mean(sp), "p25_spread": pct(sp, 25), "p75_spread": pct(sp, 75), "p90_spread": pct(sp, 90), "median_sd_log": median(sd),
                      "share_cells_over_50pct": sum(1 for x in sp if x > 0.5) / len(sp), "share_cells_identical": sum(1 for x in sp if x < 1e-9) / len(sp)}
    tbl_eq("dispersion_summary", rows, lambda r: r["spec"], list(next(iter(rows.values()))))
    sp, sd, ns = disp(units(specs["S: tag or receipt, no promotions, no far outliers"])); g = defaultdict(lambda: ([], []))
    for a, b, n in zip(sp, sd, ns): g["2" if n == 2 else "3" if n == 3 else "4+"][0].append(a); g["2" if n == 2 else "3" if n == 3 else "4+"][1].append(b)
    tbl_eq("dispersion_by_stores", {k: {"cells": len(v[0]), "median_spread": median(v[0]), "median_sd_log": median(v[1])} for k, v in g.items()}, lambda r: r["stores"], ["cells", "median_spread", "median_sd_log"])
    rows = {}
    for c, cur in EXPECT.items():
        smp = [r for r in A if r["cc"] == c and r["currency"] == cur and r["date"] >= STUDY_FROM and r["proof_type"] in SHELF and not r["disc"] and not r["far"]]
        if not smp: continue
        sp, sd, ns = disp(units(smp))
        if len(sp) >= 100: rows[c] = {"cells": len(sp), "unit_rows": sum(ns), "median_spread": median(sp), "median_sd_log": median(sd), "p90_spread": pct(sp, 90)}
    tbl_eq("dispersion_country", rows, lambda r: r["country"], ["cells", "unit_rows", "median_spread", "median_sd_log", "p90_spread"])
    # ---------------- ladder A ----------------
    S = specs["S: tag or receipt, no promotions, no far outliers"]
    cells = units(S); ulist = [(c, s, v) for c, d in cells.items() for s, v in d.items()]
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for c, d in cells.items():
        ss = list(d); [parent.__setitem__(find(s), find(ss[0])) for s in ss[1:]]
    cnt = defaultdict(int)
    for c, s, v in ulist: cnt[find(s)] += 1
    big = max(cnt, key=lambda k: cnt[k]); U = [(c, s, v) for c, s, v in ulist if find(s) == big]
    y = [v for _, _, v in U]; cg_ = {c: i for i, c in enumerate(dict.fromkeys(c for c, _, _ in U))}; sg_ = {s: i for i, s in enumerate(dict.fromkeys(s for _, s, _ in U))}
    cc_ = [cg_[c] for c, _, _ in U]; ss_ = [sg_[s] for _, s, _ in U]; n = len(U)
    cm_ = defaultdict(list)
    for i in range(n): cm_[cc_[i]].append(y[i])
    cmean = {k: mean(v) for k, v in cm_.items()}; ssw = sum((y[i] - cmean[cc_[i]]) ** 2 for i in range(n))
    [(res, acc)] = ap([y], [cc_, ss_]); rss = sum(e * e for e in res); seff = [acc[1][ss_[i]] for i in range(n)]; m0 = mean(seff); seff = [x - m0 for x in seff]
    adj = [y[i] - seff[i] for i in range(n)]; cm2 = defaultdict(list); cm3 = defaultdict(list)
    for i in range(n): cm2[cc_[i]].append(adj[i]); cm3[cc_[i]].append(y[i])
    spa = [math.expm1(max(v) - min(v)) for v in cm2.values()]; spr = [math.expm1(max(v) - min(v)) for v in cm3.values()]
    sdm = math.sqrt(sum(x * x for x in seff) / n)
    la = kv("ladder_a"); L = 1e-6
    for k, v in (("median_spread_raw", median(spr)), ("median_spread_after_store", median(spa)), ("p90_spread_raw", pct(spr, 90)), ("p90_spread_after_store", pct(spa, 90)), ("units_in_largest_component", n), ("units_in_all_cells", len(ulist)),
                 ("cells", len(cg_)), ("stores", len(sg_)), ("rms_within_cell", math.sqrt(ssw / n)), ("rms_after_store", math.sqrt(rss / n)), ("share_removed_by_store", 1 - rss / ssw), ("sd_store_effect", sdm)):
        same(f"ladder_a/{k}", v, la[k], L if k not in ("units_in_largest_component", "units_in_all_cells", "cells", "stores") else 1e-12)
    # normal-equation property: residuals sum to zero in every cell and every store
    for grp in (cc_, ss_):
        t = defaultdict(float)
        for i in range(n): t[grp[i]] += res[i]
        assert max(abs(x) for x in t.values()) < 1e-9
    disp_name = {}
    for r in S: disp_name.setdefault(r["store"], r["location_osm_display_name"])
    chs = defaultdict(lambda: [set(), 0, 0.0])
    for (c, s, v), e in zip(U, seff): ch = chains = chs[chain_of(disp_name[s])]; ch[0].add(s); ch[1] += 1; ch[2] += e
    rows = {k: {"stores": len(v[0]), "units": v[1], "mean_effect": v[2] / v[1], "effect_pct": math.expm1(v[2] / v[1])} for k, v in chs.items() if len(v[0]) >= 3 and v[1] >= 300 and k != "other"}
    tbl_eq("chains", rows, lambda r: r["chain"], ["stores", "units"]); tbl_eq("chains", rows, lambda r: r["chain"], ["mean_effect", "effect_pct"], L)
    # ---------------- ladder B ----------------
    Sk = []
    for r in S:
        o = off.get(key(r["product_code"]))
        if o and o["size"] is not None and o["leaf"] != "": Sk.append((key(r["product_code"]), o["leaf"], o["size"], r))
    nb = defaultdict(set)
    for k, leaf, sz, r in Sk: nb[leaf].add(k)
    keepc = {l for l, v in nb.items() if len(v) >= MINCAT}; ug = defaultdict(list); info = {}
    for k, leaf, sz, r in Sk:
        if leaf in keepc: ug[(k, r["quarter"], r["store"])].append(r["lp"]); info[(k, r["quarter"], r["store"])] = (sz, leaf)
    keys = list(ug); y = [mean(ug[k]) for k in keys]; x = [math.log(info[k][0]) for k in keys]; n = len(keys)
    lg = {l: i for i, l in enumerate(dict.fromkeys(info[k][1] for k in keys))}; sgb = {s: i for i, s in enumerate(dict.fromkeys(k[2] for k in keys))}; Lc = [lg[info[k][1]] for k in keys]; Sg = [sgb[k[2]] for k in keys]
    ncat, nst = len(lg), len(sgb); ym = mean(y); tss = sum((v - ym) ** 2 for v in y); yu = [y[i] - x[i] + math.log(100.0) for i in range(n)]; yum = mean(yu); tssu = sum((v - yum) ** 2 for v in yu)
    (r1, _), (xr1, _), (yur, _) = ap([y, x, yu], [Lc])
    b2 = sum(a * b for a, b in zip(xr1, r1)) / sum(a * a for a in xr1); rss2 = sum((r1[i] - b2 * xr1[i]) ** 2 for i in range(n)); rss1 = sum(v * v for v in r1); rssu = sum(v * v for v in yur)
    (r3, _), (xr3, _) = ap([y, x], [Lc, Sg]); b3 = sum(a * b for a, b in zip(xr3, r3)) / sum(a * a for a in xr3); rss3 = sum((r3[i] - b3 * xr3[i]) ** 2 for i in range(n))
    def adjr(r2, k): return 1 - (1 - r2) * (n - 1) / (n - 1 - k)
    lbm = {"price: nothing": (0.0, math.sqrt(tss / n), None, 0), "price: category": (1 - rss1 / tss, math.sqrt(rss1 / n), None, ncat - 1), "price: category + size": (1 - rss2 / tss, math.sqrt(rss2 / n), b2, ncat),
           "price: category + size + store": (1 - rss3 / tss, math.sqrt(rss3 / n), b3, ncat + nst - 1), "price per 100 g or ml: nothing": (0.0, math.sqrt(tssu / n), None, 0), "price per 100 g or ml: category": (1 - rssu / tssu, math.sqrt(rssu / n), None, ncat - 1)}
    got = table("ladder_b"); assert len(got) == len(lbm)
    for r in got:
        r2_, rms_, sl, kp = lbm[r["model"]]; same(f"ladder_b/{r['model']}/r2", r2_, r["r2"], 1e-6 if r2_ else 1e-12); same(f"ladder_b/{r['model']}/rms", rms_, r["rms"], 1e-6); same(f"ladder_b/{r['model']}/adj", adjr(r2_, kp), r["adj_r2"], 1e-6 if r2_ else 1e-12)
        if sl is not None: same(f"ladder_b/{r['model']}/slope", sl, r["slope_log_size"], 1e-6)
        else: assert r["slope_log_size"] == ""
        same(f"ladder_b/{r['model']}/units", n, r["units"], 1e-12); same(f"ladder_b/{r['model']}/barcodes", len({k[0] for k in keys}), r["barcodes"], 1e-12); same(f"ladder_b/{r['model']}/cats", ncat, r["categories"], 1e-12); same(f"ladder_b/{r['model']}/stores", nst, r["stores"], 1e-12)
    bcs = {}
    for r in S: bcs.setdefault(key(r["product_code"]), None)
    sc = kv("size_coverage")
    for k, v in (("study_barcodes", len(bcs)), ("in_off_dump", sum(1 for b in bcs if b in off)), ("with_parsed_size", sum(1 for b in bcs if b in off and off[b]["size"] is not None)),
                 ("with_size_and_category", sum(1 for b in bcs if b in off and off[b]["size"] is not None and off[b]["leaf"] != "")), ("ladder_b_barcodes", len({k[0] for k in keys})), ("ladder_b_min_barcodes_per_category", MINCAT)):
        same(f"size_cov/{k}", v, sc[k], 1e-12)
    # ---------------- proof effect ----------------
    FRp = [r for r in A if r["cc"] == "FR" and r["currency"] == "EUR" and r["date"] >= STUDY_FROM and r["proof_type"] in SHELF + ("GDPR_REQUEST", "SHOP_IMPORT") and not r["far"]]
    pe = {}
    for label, smp in (("no promotions", [r for r in FRp if not r["disc"]]), ("promotions kept (price paid)", FRp)):
        cs = defaultdict(set)
        for r in smp: cs[(r["product_code"], r["quarter"])].add(r["store"])
        smp = [r for r in smp if len(cs[(r["product_code"], r["quarter"])]) >= 2]; n = len(smp)
        cg2 = {c: i for i, c in enumerate(dict.fromkeys((r["product_code"], r["quarter"]) for r in smp))}; sg2 = {s: i for i, s in enumerate(dict.fromkeys(r["store"] for r in smp))}
        G = [[cg2[(r["product_code"], r["quarter"])] for r in smp], [sg2[r["store"]] for r in smp]]; types = ("RECEIPT", "GDPR_REQUEST", "SHOP_IMPORT")
        outs = ap([[r["lp"] for r in smp]] + [[1.0 if r["proof_type"] == t else 0.0 for r in smp] for t in types], G)
        ry = outs[0][0]; rx = [o[0] for o in outs[1:]]
        xtx = [[sum(rx[a][i] * rx[b][i] for i in range(n)) for b in range(3)] for a in range(3)]; xty = [sum(rx[a][i] * ry[i] for i in range(n)) for a in range(3)]
        def solve(M, bvec):
            k = len(M); m = [M[i][:] + [bvec[i]] for i in range(k)]
            for c in range(k):
                p = max(range(c, k), key=lambda r: abs(m[r][c])); m[c], m[p] = m[p], m[c]; d = m[c][c]; m[c] = [v / d for v in m[c]]
                for r in range(k):
                    if r != c: f = m[r][c]; m[r] = [v - f * w for v, w in zip(m[r], m[c])]
            return [m[i][k] for i in range(k)]
        b = solve(xtx, xty); inv = [solve(xtx, [1.0 if i == j else 0.0 for i in range(3)]) for j in range(3)]
        e = [ry[i] - sum(b[a] * rx[a][i] for a in range(3)) for i in range(n)]; sc_ = defaultdict(lambda: [0.0, 0.0, 0.0])
        for i in range(n):
            for a in range(3): sc_[smp[i]["store"]][a] += rx[a][i] * e[i]
        meat = [[sum(v[a] * v[c] for v in sc_.values()) for c in range(3)] for a in range(3)]
        V = [[sum(inv[a][p] * meat[p][q] * inv[q][c] for p in range(3) for q in range(3)) for c in range(3)] for a in range(3)]
        for j, t in enumerate(types): pe[(label, t)] = {"rows": sum(1 for r in smp if r["proof_type"] == t), "total_rows": n, "effect_pct": math.expm1(b[j]), "coef": b[j], "se": math.sqrt(V[j][j])}
    got = table("proof_effect"); assert len(got) == len(pe)
    for r in got:
        v = pe[(r["sample"], r["proof"])]; same(f"proof/{r['sample']}/{r['proof']}/rows", v["rows"], r["rows"], 1e-12); same(f"proof/{r['sample']}/{r['proof']}/total", v["total_rows"], r["total_rows"], 1e-12)
        same(f"proof/{r['sample']}/{r['proof']}/coef", v["coef"], r["coef"], L); same(f"proof/{r['sample']}/{r['proof']}/pct", v["effect_pct"], r["effect_pct"], L); same(f"proof/{r['sample']}/{r['proof']}/se", v["se"], r["se_store_clustered"], L)
    # ---------------- discounts ----------------
    dep = [1 - r["p"] / r["pwd"] for r in A if r["disc"] and r["pwd"] is not None and r["pwd"] > r["p"]]; ds = kv("discounts")
    for k, v in (("rows_A", len(A)), ("discounted_share", sum(1 for r in A if r["disc"]) / len(A)), ("discounted_with_regular_price", len(dep)), ("median_depth", median(dep)), ("p90_depth", pct(dep, 90)), ("mean_depth", mean(dep))): same(f"disc/{k}", v, ds[k])
    for t in ("PRICE_TAG", "RECEIPT", "GDPR_REQUEST", "SHOP_IMPORT"):
        rr = [r for r in A if r["proof_type"] == t]; same(f"disc/share_{t}", sum(1 for r in rr if r["disc"]) / len(rr), ds[f"discounted_share_{t}"])
    # ---------------- inflation ----------------
    H = json.load(open(D / "hicp_fr.json")); ti = H["dimension"]["time"]["category"]["index"]; ci = H["dimension"]["coicop"]["category"]["index"]; nt = len(ti); hic = {}
    for c, a in ci.items():
        for t, i in ti.items():
            if str(a * nt + i) in H["value"]: hic[(c, t)] = H["value"][str(a * nt + i)]
    def hq(c, q):
        y_, qq = int(q[:4]), int(q[-1]); ms = [f"{y_}-{3 * (qq - 1) + k:02d}" for k in (1, 2, 3)]; m0 = [f"{y_ - 1}-{3 * (qq - 1) + k:02d}" for k in (1, 2, 3)]
        return sum(hic[(c, m)] for m in ms) / sum(hic[(c, m)] for m in m0) - 1 if all((c, m) in hic for m in ms + m0) else None
    dd = defaultdict(list)
    for r in A:
        if r["cc"] == "FR" and r["currency"] == "EUR" and r["date"] >= "2024-01-01" and r["proof_type"] in SHELF and not r["disc"] and not r["far"]: dd[(r["product_code"], r["store"], r["date"])].append(r["lp"])
    ser = defaultdict(list)
    for (pc, s, d), v in dd.items(): ser[(pc, s)].append((date.fromisoformat(d).toordinal(), d, median(v)))
    pairs = []
    for (pc, s), lst in ser.items():
        lst.sort()
        for j in range(len(lst)):
            best = None
            for i in range(j):
                gap = lst[j][0] - lst[i][0]
                if 300 <= gap <= 430 and (best is None or abs(gap - 365) < abs(best[0] - 365)): best = (gap, lst[i][2])
            if best: pairs.append((s, lst[j][1], lst[j][2] - best[1]))
    byq = defaultdict(list)
    for s, d, dl in pairs: byq[quarter(d)].append((s, dl))
    got = table("inflation"); assert len(got) == len(byq)
    for r in got:
        q = r["quarter"]; lst = byq[q]; kept = [(s, dl) for s, dl in lst if abs(dl) <= math.log(3)]; mu = mean([dl for _, dl in kept]); cl = defaultdict(float)
        for s, dl in kept: cl[s] += dl - mu
        same(f"infl/{q}/pairs", len(lst), r["pairs"]); same(f"infl/{q}/stores", len({s for s, _ in lst}), r["stores"]); same(f"infl/{q}/kept", len(kept), r["pairs_kept"])
        same(f"infl/{q}/median", math.expm1(median([dl for _, dl in lst])), r["median_change"]); same(f"infl/{q}/mean", math.expm1(mu), r["mean_change"]); same(f"infl/{q}/se", math.sqrt(sum(v * v for v in cl.values())) / len(kept), r["se_mean_log"])
        same(f"infl/{q}/unch", sum(1 for _, dl in lst if abs(dl) < 1e-9) / len(lst), r["share_unchanged"]); same(f"infl/{q}/up", sum(1 for _, dl in lst if dl > 1e-9) / len(lst), r["share_up"])
        for c, f in (("CP01", "hicp_food_yoy"), ("CP00", "hicp_all_yoy")):
            h = hq(c, q); assert (h is None) == (r[f] == "");
            if h is not None: same(f"infl/{q}/{f}", h, r[f])
    # ---------------- cross-country ----------------
    mm = defaultdict(list)
    for r in A:
        if r["currency"] == "EUR" and r["cc"] in EURO and r["date"] >= STUDY_FROM and r["proof_type"] in SHELF and not r["disc"] and not r["far"]: mm[(r["product_code"], r["cc"])].append(r["lp"])
    mmed = {k: median(v) for k, v in mm.items()}; rows = {}
    for c in EURO:
        if c == "FR": continue
        dl = [mmed[(pc, c)] - mmed[(pc, "FR")] for (pc, cc2) in mmed if cc2 == c and (pc, "FR") in mmed]
        if len(dl) >= 15: rows[c] = {"barcodes": len(dl), "median_ratio": math.exp(median(dl)), "p25_ratio": math.exp(pct(dl, 25)), "p75_ratio": math.exp(pct(dl, 75)), "share_cheaper_than_france": sum(1 for v in dl if v < 0) / len(dl)}
    tbl_eq("cross_country", rows, lambda r: r["country"], ["barcodes", "median_ratio", "p25_ratio", "p75_ratio", "share_cheaper_than_france"])
    # ---------------- raw produce ----------------
    pr = defaultdict(lambda: {"KILOGRAM": [], "UNIT": []})
    for r in raw:
        if r["type"] == "CATEGORY" and r["currency"] == "EUR" and r["cc"] == "FR" and r["date"] >= STUDY_FROM and r["p"] is not None and r["p"] > 0 and r["price_per"] in ("KILOGRAM", "UNIT"): pr[r["category_tag"]][r["price_per"]].append(r["p"])
    full = {k: v for k, v in pr.items() if len(v["KILOGRAM"]) >= 10 and len(v["UNIT"]) >= 10}
    order = sorted(full, key=lambda k: (-(len(full[k]["KILOGRAM"]) + len(full[k]["UNIT"])), k))[:8]; got = table("produce_units"); assert [r["category_tag"] for r in got] == order
    for r in got:
        v = full[r["category_tag"]]; same(f"prod/{r['category_tag']}/kg", median(v["KILOGRAM"]), r["median_per_kg"]); same(f"prod/{r['category_tag']}/unit", median(v["UNIT"]), r["median_per_unit"]); same(f"prod/{r['category_tag']}/nkg", len(v["KILOGRAM"]), r["rows_per_kg"])
        same(f"prod/{r['category_tag']}/nunit", len(v["UNIT"]), r["rows_per_unit"]); same(f"prod/{r['category_tag']}/ratio", median(v["KILOGRAM"]) / median(v["UNIT"]), r["kg_over_unit"])
    allp = [r for r in raw if r["type"] == "CATEGORY" and r["p"] is not None and r["p"] > 0]; ps = kv("produce_share")
    same("prod/rows", sum(1 for r in raw if r["type"] == "CATEGORY"), ps["category_rows"]); same("prod/kgshare", sum(1 for r in allp if r["price_per"] == "KILOGRAM") / len(allp), ps["per_kilogram_share"])
    same("prod/unitshare", sum(1 for r in allp if r["price_per"] == "UNIT") / len(allp), ps["per_unit_share"]); same("prod/both", len(full), ps["categories_with_both_units"])
    print(f"check.py: {n_ok} recomputed numbers agree with results/")


if __name__ == "__main__":
    main()
