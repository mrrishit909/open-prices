"""Step 1: the Open Prices database (crowdsourced shelf prices), the Open Food Facts product table for pack sizes, and the official French consumer price index.

    ~/tools/venv/bin/python download.py   -> data/prices.csv.gz  data/off_products.csv.gz  data/hicp_fr.json, results/pull.json

Open Prices (Open Food Facts), daily bulk export of every price: https://huggingface.co/datasets/openfoodfacts/open-prices/resolve/main/prices.parquet (the official dump, published by the openfoodfacts organisation on Hugging Face and described at
https://huggingface.co/datasets/openfoodfacts/open-prices ; live API https://prices.openfoodfacts.org/api/v1/prices). Licence: Open Database Licence (ODbL 1.0): credit Open Prices / Open Food Facts contributors,
share alike for derived databases. Only this project's own aggregates (results/) are published here, never the raw rows. The parquet is converted to gzip CSV with pyarrow (lossless, selected columns) so check.py can read it in plain Python.
Open Food Facts product table (ODbL, contents DbCL): https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz (1.3 GB, streamed and filtered to the barcodes that have a price; the 1.3 GB is never stored).
Eurostat HICP (monthly index, 2015=100, France, all items CP00 and food CP01), table prc_hicp_midx, reuse permitted with source: https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_midx
Cite: Open Food Facts contributors, Open Prices, ODbL 1.0; Eurostat, HICP. Descriptive User-Agent below. Raw data stays in data/ (gitignored); results/pull.json holds URLs, sizes and checksums.
"""
import csv
import gzip
import hashlib
import io
import json
import ssl
import sys
import time
import urllib.request
import zlib
from datetime import date
from pathlib import Path

import certifi
import pyarrow.parquet as pq

HERE = Path(__file__).parent
DATA = HERE / "data"
UA = {"User-Agent": "research-portfolio (Rishit Raj Mathur)"}
CTX = ssl.create_default_context(cafile=certifi.where())
PRICES_URL = "https://huggingface.co/datasets/openfoodfacts/open-prices/resolve/main/prices.parquet"
OFF_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
HICP_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_midx?geo=FR&coicop=CP00&coicop=CP01&unit=I15&sinceTimePeriod=2023-01&format=JSON"
PRICE_COLS = ["id", "type", "product_code", "category_tag", "price", "price_is_discounted", "price_without_discount", "discount_type", "price_per", "currency", "location_id", "location_osm_id", "date",
              "owner", "proof_id", "proof_type", "proof_date", "proof_currency", "created", "location_osm_display_name", "location_osm_tag_value", "location_osm_address_country_code"]
OFF_COLS = ["code", "product_name", "brands", "quantity", "product_quantity", "categories_tags", "unique_scans_n"]


def get(url, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), context=CTX, timeout=120)
        except OSError:
            if k == tries - 1: raise
            time.sleep(5)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def prices(info):
    raw = DATA / "prices.parquet"
    with get(PRICES_URL) as r, open(raw, "wb") as f:
        while b := r.read(1 << 20): f.write(b)
    pf = pq.ParquetFile(raw); n = 0
    with gzip.open(DATA / "prices.csv.gz", "wt", newline="", compresslevel=6) as out:
        w = csv.writer(out); w.writerow(PRICE_COLS)
        for batch in pf.iter_batches(batch_size=50000, columns=PRICE_COLS):
            cols = [batch.column(c).to_pylist() for c in PRICE_COLS]
            for row in zip(*cols):
                w.writerow(["" if v is None else (v.isoformat() if hasattr(v, "isoformat") else str(v)) for v in row]); n += 1
    assert n == pf.metadata.num_rows
    info["files"]["prices.parquet"] = {"url": PRICES_URL, "bytes": raw.stat().st_size, "sha256": sha_file(raw), "rows": n}
    info["files"]["prices.csv.gz"] = {"url": "derived from prices.parquet by pyarrow (selected columns, lossless)", "bytes": (DATA / "prices.csv.gz").stat().st_size, "sha256": sha_file(DATA / "prices.csv.gz"), "rows": n}
    raw.unlink()


def key(code):
    return code.lstrip("0")


def off_products(info):
    want = set()
    with gzip.open(DATA / "prices.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["product_code"]: want.add(key(r["product_code"]))
    for attempt in range(3):
        try:
            resp = get(OFF_URL); h = hashlib.sha256(); nbytes = [0]; kept = 0; seen = 0
            class Tap(io.RawIOBase):
                def readable(self): return True
                def readinto(self, b):
                    d = resp.read(len(b))
                    h.update(d); nbytes[0] += len(d); b[:len(d)] = d; return len(d)
            text = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BufferedReader(Tap(), 1 << 20)), encoding="utf-8", newline="", errors="replace")
            rd = csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
            head = next(rd); ix = [head.index(c) for c in OFF_COLS]; seen_codes = set()
            with gzip.open(DATA / "off_products.csv.gz", "wt", newline="", compresslevel=6) as out:
                w = csv.writer(out); w.writerow(OFF_COLS)
                for row in rd:
                    seen += 1
                    if len(row) <= max(ix): continue
                    k = key(row[ix[0]])
                    if k in want and k not in seen_codes:
                        seen_codes.add(k); w.writerow([row[i] for i in ix]); kept += 1
            info["files"]["off_products.csv.gz"] = {"url": OFF_URL, "bytes_streamed": nbytes[0], "sha256_of_stream": h.hexdigest(), "rows_in_dump": seen, "rows_kept": kept,
                                                    "note": "filtered to the barcodes present in prices.csv.gz, leading zeros stripped for matching; the full file is not stored", "stored_bytes": (DATA / "off_products.csv.gz").stat().st_size}
            return
        except (OSError, zlib.error, EOFError) as e:
            print("off stream failed, retry", e, file=sys.stderr); time.sleep(10)
    raise RuntimeError("OFF stream failed 3 times")


def hicp(info):
    raw = get(HICP_URL).read(); (DATA / "hicp_fr.json").write_bytes(raw)
    info["files"]["hicp_fr.json"] = {"url": HICP_URL, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "updated": json.loads(raw)["updated"]}


def main():
    DATA.mkdir(exist_ok=True); (HERE / "results").mkdir(exist_ok=True)
    info = {"pulled": date.today().isoformat(), "files": {}}
    hicp(info); prices(info); off_products(info)
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v.get("bytes", v.get("bytes_streamed")) for k, v in info["files"].items()}))


if __name__ == "__main__":
    main()
