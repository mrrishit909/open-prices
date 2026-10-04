# Does one barcode cost the same everywhere?

Real shelf prices that volunteers entered into Open Food Facts' Open Prices database (photos of price tags and receipts, one barcode in one shop on one day), audited and then asked five questions:
- who and what the crowd covers (countries, shops, products, contributors), and how lopsided that is;
- how far apart the prices of the same barcode are in different shops, and how much of the gap survives dropping outliers and promotions;
- how much of that gap is the shop's price level (its chain), and how much is left;
- across different barcodes of one category, how much of a price is pack size and shop, and what happens when a kilo is entered as a piece;
- whether the crowd's repeat prices rise like official French food inflation, and what traps (duplicates, units, currencies, receipts, loyalty-card exports) the file holds.

The questions follow GeeksforGeeks' "Product Price Scraping and Analysis". Scraping retailers' sites is not allowed here, so the data is the compliant stand-in: the crowdsourced Open Prices database, with a bulk file instead of thousands of requests. The page with the results and the charts is the build log:

Build log: https://mrrishit909.github.io/projects/open-prices/

## Data

Open Prices, run by Open Food Facts: every price in the database, from the official daily bulk export published by the `openfoodfacts` organisation on Hugging Face (https://huggingface.co/datasets/openfoodfacts/open-prices, `prices.parquet`, 320,399 prices when pulled on 2026-10-04; live API https://prices.openfoodfacts.org/api/v1/prices). Licence: Open Database Licence (ODbL 1.0): credit Open Prices and the Open Food Facts contributors; derived databases must be shared alike. This repository publishes only its own aggregates (`results/`), never the rows, photos or contributor ids. The Open Food Facts product table (https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz, ODbL, contents under the Database Contents Licence; 1.3 GB streamed and filtered to the 116,467 barcodes that have a price, never stored whole) gives each barcode's pack-size text and categories. The official price index is Eurostat's HICP for France (table prc_hicp_midx, food and all items, 2015 = 100, to December 2025; https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_midx). The stored files are about 49 MB, fetched by `download.py` (checksums in `results/pull.json`), not committed.

```
~/tools/venv/bin/python download.py    # data/* (the 1.3 GB product table is streamed and filtered; a few minutes)
~/tools/venv/bin/python analyze.py     # results/*   (about 20 seconds, 2 GB of memory)
~/tools/venv/bin/python check.py       # plain-Python recount of every number (several minutes)
~/tools/venv/bin/python charts.py
```

## Steps

1. **Download** the Open Prices parquet (converted to gzip CSV, lossless), the Open Food Facts products for the priced barcodes, and Eurostat's French HICP, with checksums.
2. **Audit.** A cleaning funnel (a barcode product, a positive price, a currency, a date from 2018 to the pull date, a store with a country, the country's own currency, one row per barcode-store-date-price-currency) and every trap counted: duplicates, currency mismatches, odd barcodes, discounts without a regular price, loyalty-card exports, far outliers (3 times or a third of the barcode's median, where a barcode has 5 or more prices).
3. **Coverage.** By country and month; the concentration of contributors, stores and proof types.
4. **Dispersion.** For France in the euro from 2025, tag or receipt proofs, no promotions, no far outliers (the study sample): one value per barcode, store and quarter (mean log price); a case is a barcode in one quarter sold in 2 or more shops; the gap is the dearest shop over the cheapest, minus 1. Five variants, and each country in its own currency.
5. **Shop ladder.** log price = case effect + shop effect + residual, fitted by least squares (sparse LSQR) in the largest connected set of shops: how much of the within-case variance the shop's price level removes, and the chain levels.
6. **Size ladder.** Across barcodes: log price on category (the most specific Open Food Facts tag, 30 or more barcodes), log pack size (g or ml parsed from "250 g", "1 l"), and shop; the same in price per 100 g or ml.
7. **Proof effect.** Receipts, loyalty-card exports and shop imports against price tags, inside barcode-quarter and shop, store-clustered errors.
8. **Inflation.** The same barcode in the same shop 300 to 430 days apart (closest to a year), by quarter of the later price, against HICP France food and all items over the same quarter.
9. **Cross-country and raw produce.** The same barcode in France and another euro country (no conversion needed); per-kilogram and per-unit prices of raw produce.
10. **Recount** every table with `check.py`: plain Python (csv, gzip, json, math, loops), the least-squares fits by alternating projections instead of LSQR.

## Results

- **Coverage is lopsided.** 320,399 prices, 140,875 barcodes, 6,977 shops, 3,659 contributors; the top 37 contributors (1%) entered 90% of the prices and the top 10 shops 25%. France holds 67% of the cleaned prices and the US 13%; 84% are from supermarkets; 2026-02 alone has 34,233 (12% of the cleaned prices, 73% of them from one contributor); 48% are dated 2026.
- **The file is mostly clean, with a few traps.** 291,812 rows (91%) survive the funnel. Dropped: 10,154 raw-produce rows with no barcode, 15,815 duplicates (5.1% of the rows reaching that step; 23% of them from the same proof), 435 prices in a currency that is not the store country's. Kept but flagged: 36,277 loyalty-card exports and shop imports (11%), 3,146 discounted prices with no regular price (10% of the discounted). Only 36% of the cleaned rows belong to a barcode with 5 or more prices, the only ones whose outliers can be seen: 104 far outliers (0.10% of them), 2 of them decimal slips.
- **The same barcode is 12% dearer in the dearest shop (median).** 17,488 barcode-quarters sold in 2 or more French shops (49,378 barcode-shop-quarters): the gap is 12.2% at the median, 19% on average, 46% at the 90th percentile; 8% of cases have the same price everywhere. It grows with the number of shops compared: 7.3% for 2, 17.6% for 3, 31.9% for 4 or more. Keeping the far outliers changes the median to 12.2%, keeping promotions to 13.0%, using price tags only to 12.3%, receipts only to 7.5% (441 cases). In their own currencies: US 12%, Norway 4%, Sweden 4%, Germany 0% (a median of 0 means that in at least half of the 273 cases every shop has the same price).
- **The shop's price level explains 60% of the variance within a barcode-quarter.** 49,350 barcode-shop-quarters, 839 shops: the root-mean-square gap to the case mean falls from 10.7% to 6.8%, the median dearest-over-cheapest gap from 12.2% to 7.6%, the 90th percentile from 46% to 26%. Chains, against the same barcodes elsewhere: E.Leclerc -11% (73 shops), Intermarché -2%, Auchan 0%, Système U +3%, Carrefour +5%, Monoprix +14%, Franprix +27%.
- **Across barcodes, pack size explains little.** 27,885 barcode-shop-quarters, 10,440 barcodes in 156 categories (of 57,549 study barcodes, 35,130 have a pack size that parses): category explains 43.5% of the variance of log price, pack size adds 3.1% (price rises by 0.17 per unit of log size, where proportional pricing would give 1), the shop adds 8.2%: 45% is left (brand, quality, pack type). Price per 100 g or ml is more spread inside a category (root-mean-square 0.69 in logs) than the price itself (0.45).
- **Receipts read lower than price tags.** Inside barcode-quarter and shop, a receipt price is -1.3% against a price tag (store-clustered error 0.4%; 3,437 receipts); with promotions kept -2.9%. Loyalty-card exports -2.0% (error 3.0%) and shop imports +3.9% (error 3.3%) cannot be told from tags. 9.7% of cleaned prices are discounted, by a median of 23% (90th percentile 47%).
- **The crowd's year-on-year prices drift below official food inflation in 2025, on thin data.** Same barcode, same shop, about a year apart: in 2025-Q4 the mean change was -1.7% (672 pairs in 27 shops, store-clustered 95% band -4.9% to +1.6%) against Eurostat's +1.6% for French food and +0.8% for all items; the median pair did not change (23% of pairs are unchanged to the cent). The band includes the official food figure, so the data cannot say the crowd is wrong, only that it is too thin to confirm it. 2026 pairs run at -1.5% to -2.4% with no official figure to compare in the Eurostat file (it ends in December 2025).
- **Neighbours and units.** Median price of the same barcode in another euro country over France: Germany 0.91 (153 barcodes), Belgium 1.12 (119); no other country shares more than 48 barcodes with France. Raw produce is 74% priced per kilogram and 26% per unit, and the two are not comparable: lemons' median is 3.95 per kg and 0.79 per unit, limes 5.99 and 0.59, while bananas "per unit" have the same median (1.99) as bananas per kilogram, which suggests per-kilogram prices entered as per-unit.

## Not done, and caveats

- The contributors choose what, where and when to record: the data is 67% French, led by a few dozen contributors and a few chains, with a spike in 2026-02 (73% of it from one contributor). Nothing here is representative of any country's shelves, and the dispersion is of the products and shops the volunteers chose.
- The gap between shops mixes real price differences, promotions the tag does not show, shop formats (a Franprix is not a Leclerc hypermarket), differences in the day within a quarter, and entry errors; the store ladder removes the shop's level but cannot separate format from chain, and the chain levels come from the shops the volunteers visited.
- Receipts show what was paid (loyalty prices, unannounced discounts) and tags what was shown; the proof effect is an association inside barcode-quarter and shop, not a cause. A discounted price is the price paid; where the regular price was missing (10% of the discounted) the row is only used in the "promotions kept" variant.
- Outlier checks need 5 or more prices of a barcode (36% of the rows): a wrong price on a barcode seen once is invisible. Pack size parses for 35,130 of the 57,549 study barcodes; multipacks, counts and ounces are not parsed, and the category is the most specific Open Food Facts tag, which is noisy.
- The inflation comparison has 64 to 925 pairs per quarter in 13 to 37 shops, repeated shops are not a random sample, and HICP is spending-weighted and includes fresh food without barcodes. Eurostat's file ends in December 2025, so 2026 has no official figure here. The cross-country comparison is a handful of barcodes and compares different shop formats.
- The data is updated daily; a rerun gives slightly different numbers. The figures here are for the file pulled on 2026-10-04.
- `check.py` recounts every number in `results/` in plain Python (653 numbers). The three least-squares families are compared at a relative 1e-6 (LSQR and alternating projections stop at different tiny residuals), everything else at 1e-9. Not independently rebuilt: the parquet to CSV conversion (pyarrow; the row count is asserted in `download.py`).
- Attribution: price data from Open Prices / Open Food Facts contributors (ODbL 1.0); this repository's aggregates are shared under the same licence.
