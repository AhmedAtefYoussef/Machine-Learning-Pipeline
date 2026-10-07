# DATA_CARD (chief, measured once on the real files with seed 44615; source exp/eda0/eda.json, eda1.json). Agents read this, not the CSVs.

## Files and time structure
- train.csv 10,886 x 15, test.csv 574 x 14, no NaN, `instant` unique, no instant shared between the files (train max 17093, test 432..17117).
- train = days 1-19 of every month, Jan 2011 - Dec 2012: 456 days, 42 of them with fewer than 24 rows. test = the 20th of each month: 24 days (22 with 24 rows, 2 with 23). The hidden test is whole unseen days interleaved through the timeline, not a future block.
- Fewest rows at hours 3 (433), 4 (442), 2 (448) of 456 days; cnt >= 1 always (0 zeros). Quiet hours are absent, not recorded as 0.
- weekday: 0 = Sunday ... 6 = Saturday. season = calendar quarter of mnth (1 = Jan-Mar). 13 holiday dates in train (311 rows), mostly Mondays; one holiday date in test (2012-02-20, 23 rows).

## Seeded split
8708 train / 2178 validation rows; mean cnt 191.6 / 191.4. Every validation row has same-day rows in the training portion (100%).
Chronological cut 2012-07-01: about 75% / 25% of train.csv.

## Target
cnt mean 191.6, median 145, sd 181.1, skew 1.24, min 1, max 977; quantiles 75% 284, 90% 452, 99% 774. 2011 mean 144.2, 2012 mean 238.6 (ratio 1.654).

## Demand structure
- Working days: peaks at 17h (529), 18h (495), 8h (480). Non-working days: midday hump 12-14h (379-388).
- weathersit rows: 1: 7192, 2: 2834, 3: 859, 4: 1 (test: 358 / 187 / 29 / 0).

## Redundancy (train)
r(temp, atemp) 0.985; r(season, mnth) 0.972 (exact function); workingday = (weekday in 1..5) and not holiday, 0 exceptions; r(instant, days since start) 0.99999; r(yr, days) 0.867.

## Quirks
hum = 0: 22 rows, all 2011-03-10. windspeed = 0: 12.1% of train, 17.8% of test; 28 distinct wind levels. |atemp - temp| > 0.2: 24 rows, all 2012-08-17 (train only).

## Calibration from exp/eda0 (closed-form fits; bug detectors, NOT results to quote)
Bike-scale R2, log1p target, no back-transform factor, trend + 2 harmonics:
| design | p | train | seeded val | day-block | chrono |
|---|---|---|---|---|---|
| no hour | 13 | 0.168 | 0.164 | 0.163 | 0.029 |
| additive (Phase 1 base) | 36 | 0.726 | 0.718 | 0.724 | 0.608 |
| + workingday x hour | 59 | 0.892 | 0.892 | 0.890 | 0.736 |
| + cubic temp/hum/wind | 65 | 0.913 | 0.911 | 0.911 | 0.858 |
| + hour x temp, hour x hum | 111 | 0.927 | 0.926 | 0.923 | 0.887 |
| ... + weekday x hour, wd x hour x temp | ~400 | 0.941 | 0.935 | 0.933 | 0.857 |
| ... + month x hour, month x wd x hour | ~900 | 0.944 | 0.928 | 0.929 | 0.832 |
Seeded and day-block estimates agree to 0.003 for these linear models; the chronological estimate is 0.04-0.10 lower. GD (lr = 1/lambda_max): base design lambda_max 1.974, condition 45, 475 iterations; wd x hour + cubic: condition about 520, about 4200 iterations.
Back-transform on validation: none 0.9108, ls 0.9108, Duan 0.9040.
