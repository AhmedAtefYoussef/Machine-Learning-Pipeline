# Audit C: dataset deep-dive (analyst, 2026-10-16). Diagnostic only; nothing here enters the chain.
Sources: exp/audit_c/facts.json (facts.py), probe.json (probe.py, sha 511e39518d9b), probe2.json (probe2.py, sha acfdd3e364ab),
probe3.json (probe3.py). test.csv read for dates/feature ranges only (no label exists, nothing tuned on it).
Base model for probes = P3 target design (C6, 282 cols), power target 0.1, closed-form ridge 1e-4 (exp/v2/lib).
Seeded-validation looks spent: 6 (evaluate() prints it; no decision was taken on it).

## 1 Time coverage
- train = days 1-19 of all 24 months (2011-01-01 .. 2012-12-19), 456 days; test = day 20 of every month, 24 days, 0 shared dates.
- every test day's previous day (the 19th) is in train (100%); its next day (21st) is in neither file (0%).
  -> hidden test = interpolation inside known months, at the edge of each 19-day block. Not a future block.
- test: 17 working days / 7 non-working; weekdays 0..6 = 4,3,4,3,4,4,2 days.
- day-of-month has no effect: OOF obs/pred ratio by day of month 0.973..1.037, no edge drift at 17-19.

## 2 Missing hours
- train: 58 missing of 456x24; 42 days < 24 rows; 91% of missing hours fall in 0-6h (hr3: 23, hr4: 14, hr2: 8).
  -> quiet hours are absent, not 0 (cnt >= 1 always; 105 rows with cnt = 1; cnt <= 5 = 6.4% of rows, all at 0-7h).
- one real outage: 2011-01-18 hours 0-11 missing (12 h run); the day's total is 0.48 x its month/day-type median.
  All other runs <= 2 h, at 2-5h.
- test: 2 missing hours (2011-02-20 5h, 2012-02-20 4h): quiet hours, same pattern.
- 83 train rows sit next to an in-day gap (minor; no test analogue).

## 3 Operational quirks
- weathersit 4: 1 row (2012-01-09 18h, cnt 164); test 0. Merge into 3 is right.
- hum = 0: 22 rows, only 2011-03-10, and 20 of them are weathersit 3 (rain). Fill = train median of positive hum 0.62,
  but rainy-hour median is 0.87 and the neighbouring days average 0.71. With hum^2, hum^3 in the model (partial effect of
  hum > 0.9 = -0.42 log) the fill treats a rain day as a dry one. Train-only rows, 0.2%: defensibility, not score.
- windspeed = 0: 12.1% train / 17.8% test; next level 0.0896 (28 levels, a quantised sensor). More frequent at night
  (17.0% at 0-5h vs 9.0% at 12-17h); 46% of zero rows in runs >= 3 h, longest 15 h (possible stuck sensor), but cnt at
  wind 0 vs the next level differs by -0.015 log inside hour x day-type cells: behaves as calm. Leaving it is right.
- atemp: 2012-08-17 = constant 0.2424 for all 24 rows while temp 0.62-0.86 (train only). Test atemp is clean
  (max |atemp - fit(temp)| 0.088, same as train 0.085). atemp is a P4 candidate only: harmless.
- worst days are rain days (2011-12-07 ratio 0.20, 2011-04-16 0.28, 2011-03-06 0.28, 2011-03-10 0.31, max ws 3, hum >= 0.89).

## 4 Redundancy (exact, train + test)
- weekday = dteday weekday (0 = Sun), yr = year - 2011, mnth = month, season = calendar quarter of mnth: all 100%.
  (season is NOT the meteorological season its labels suggest: "winter" = Jan-Mar.)
- workingday = weekday in 1..5 and not holiday: 0 exceptions; no holiday falls on a weekend.
- instant is monotone in time (r with day count 0.99999): a trend proxy; correctly never used as a raw input.
- holidays: train 13 dates (MLK x2, Emancipation x2, Jul 4 x2, Labor x2, Columbus x2, Veterans x2, 2012-01-02);
  test 1: 2012-02-20 Presidents' Day, 23 rows (4% of test), a holiday TYPE never seen in train.

## 5 Collinearity
- r(temp, atemp) 0.985; VIF temp 39.8 / atemp 35.6; everything else VIF <= 3.8 (doy_cos 3.8, r(temp, doy_cos) -0.835).
- r(hum, windspeed) -0.32, r(hum, hr) -0.28, r(trend, temp) 0.18. Only temp/atemp is a real conditioning issue.

## 6 Non-linear effects (partial log cnt after hour x day type x year means)
- hr x workingday: working-day peaks 8h 480, 17h 529, 18h 495; non-working hump 12-14h 379-388.
  ratio wd/non-wd: 4.28 at 8h, 1.56 at 17h, 0.51 at 13h -> the interaction is the largest single effect (eda: +0.17 R2).
- temp: rises monotonically to 0.7-0.8 (+0.36) and falls above 0.9 (-0.03): concave, peak about 30-32 C. Cubic is right.
- temp slope depends on day part: 1.31 (am peak) .. 1.45 (midday) .. 1.81 (pm peak) .. 2.27 (evening). hr x temp justified.
- hum: flat 0.2-0.8, then -0.42 above 0.9 (rain proxy). Cubic hum justified.

## 7 Target
- cnt mean 191.6, median 145, skew 1.24; skew after sqrt 0.27, power 0.1 -0.61, log1p -0.85.
- mean-variance law in 192 cells (yr x wd x hr x season, n >= 20): log var = b log mean, b = 1.56 -> variance-stabilising
  power 1 - b/2 = 0.22. Chosen 0.1 (ADR-017, by validation) sits between that and log.
- OOF residual sd on the 0.1 scale falls 0.63 -> 0.27 across predicted deciles (bike scale rises 3.8 -> 71.9): the
  transform over-compresses the top; consistent with b. Top 10% of rows carry 53% of squared deviation of cnt.
- working-day rush rows (7-8h, 17-19h) = 14% of rows, 37% of OOF squared error.

## 8 Trend and growth
- monthly mean 2011: 55 74 87 111 175 197 204 183 175 175 155 134; 2012: 124 146 208 257 264 287 267 286 293 281 232 217.
- 2012/2011 ratio by month: 2.28 1.98 2.40 2.32 | 1.51 1.46 1.31 1.56 1.68 1.60 1.49 1.62. Growth is NOT constant:
  about 2.2x in Jan-Apr (ramp-up of a young system) vs 1.5x afterwards. Card quotes only the pooled 1.654.
- consequence (OOF on held-out days): linear trend + annual harmonics force one year-on-year ratio, so obs/pred by
  year-month = 0.82 0.91 0.87 0.89 in 2011 Jan-Apr (over-predicted), 1.06 1.13 1.13 1.10 in 2012 Jan-Apr (under),
  0.94-0.96 in 2012 May-Jul. Month-level bias of 5-18% is the largest structured residual left.
- chrono split: pred/obs mean ratio 1.062 after 2012-07-01 (exponential trend overshoot); irrelevant to the hidden test.

## 9 Random-split optimism
- seeded val rows sharing a day with train rows: 100%; with the adjacent hour in train: 95.6%; both neighbours: 61.6%;
  mean 18.3 same-day train rows per val row.
- day level = 22% of OOF squared error (day-mean residual): what a same-day neighbour reveals and the hidden test hides.
  Linear models barely exploit it (seeded 0.9413 vs held-out days 0.9462); the tree ceiling does (gap 0.016, ADR-018).
- INTERVALS: every pipeline interval is a ROW bootstrap (common.bootstrap_r2; p1, p2, p3, p4). On OOF predictions:
  row-bootstrap 95% CI 0.9423-0.9501 (width 0.0078), DAY-cluster bootstrap 0.9394-0.9523 (width 0.0129, 1.65x).
  A 24-day test-sized sample: 95% range 0.912-0.967, sd 0.0147; 19 day-of-month hold-outs (each = 24 days, one per
  month, the closest copy of the hidden test): mean 0.9447, sd 0.0135, range 0.911-0.964.
  -> the hidden-test R2 is uncertain by about +-0.03 from day sampling alone; quote it.

## 10 Pipeline vs facts: what is right
features.py / common.py: hum fill learned on fit rows (H2 ok); ws 4 -> 3; linear trend + 2 harmonics; season, mnth,
weekday, yr, instant, atemp only as P4 candidates (exact copies); weather memory from inputs by timestamp, test sees the
19th's 23h through the train inputs (no label). No leakage found. DATA_CARD numbers re-measured: all correct.

## 11 What the pipeline MISSES or gets wrong (ranked)
| # | item | evidence | fix | gain held-out days / test-mimic / chrono | compliant |
|---|---|---|---|---|---|
| 1 | intervals are row bootstrap; days are the sampling unit | 1.65x width; 24-day sd 0.0147 | day-cluster bootstrap in common (resample dates), quote expected test spread 0.91-0.97 | 0 (honesty) | yes |
| 2 | growth differs by month (2.2x Jan-Apr vs 1.5x) -> 5-18% year-month bias | sec 8 | yr x 4 doy harmonics (4 cols) | +0.0035 (se 0.0005, 5/5) / +0.0040 (se 0.0006, 19/19) / -0.342 | yes, but fails ADR-018 chrono rule |
| 2b | same, coarser | sec 8 | 23 year-month levels | +0.0030 (0.0006, 5/5) / +0.0032 (0.0008, 14/19) / -0.054 | yes, same rule conflict |
| 3 | hum fill 0.62 for a rain day (ws3 median 0.87) | sec 3 | fill by train median of hum within weathersit | ~0 (22 train rows) | yes |
| 4 | test holiday type unseen (Presidents' Day) | OOF holiday ratios 0.80-1.29 | none possible with 13 dates; disclose +-15% on 23 rows | - | - |
| 5 | season label semantics (quarter) and wind-0 = calm, not missing | sec 3-4 | one line each in DATA_CARD / report | 0 | yes |
| 6 | lead weather (next hour, worst of next 3 h) | probe.json | reject | +0.0008 (se 0.0003) / day19 -0.004 / -0.0005 | yes (look-ahead, doubtful) |

## 12 Judgement
- #2 is the only representation gain above its paired noise (6.7 se on the test-mimic validator, 19/19 wins), and it
  targets exactly the hidden test's structure (interpolation inside known months). It is blocked only by the team's own
  chrono non-inferiority rule, which guards a risk (future months) that the hidden test does not carry. But: +0.004 is
  0.3 sd of the test's own day-sampling noise; adopting it means changing an acceptance rule after seeing results,
  re-running P3-P6 (~25 min + prose), and a final model whose chrono score collapses (0.94 -> 0.60), which the report
  must then explain. Recommendation: do not adopt two days before the deadline; disclose it as the measured remaining
  gap ("year-specific seasonality: +0.004 on held-out days, not identifiable chronologically").
- #1 is cheap and makes the report more defensible; recommend (P1-P4 intervals change, numbers re-trace; no metric moves).
- Otherwise: no further gain above noise in the data itself.
