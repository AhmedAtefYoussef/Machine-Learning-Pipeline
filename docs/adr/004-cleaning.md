# ADR-004 Cleaning rules
status: accepted        phase: p0 (frozen at F1)        owner: chief
context: operational quirks (description 1.4): corrupted or missing sensor values.
options: per quirk: leave / flag / impute (statistics from train only)
evidence: exp/eda0/eda.json. hum = 0 on 22 rows, all on 2011-03-10 (sensor failure): imputing the train median changes validation R2 by -0.001 (noise) but removes an impossible value. windspeed = 0 in 12.1% of train rows (17.8% of test), 28 distinct levels: a wind_zero flag changes validation R2 by +0.001 (noise). weathersit 4: one train row, none in test. atemp: 24 corrupt rows on 2012-08-17 (|atemp - temp| > 0.2), train only. cnt >= 1 always and hours 2-4 have the fewest rows: quiet hours are missing rather than recorded as 0.
decision: hum 0 -> train median of positive hum; weathersit 4 -> 3; windspeed left as measured, no flag; atemp left as is (it is a Phase 4 candidate only, and the 24 rows are 0.2% of train); no rows dropped, no hours invented.
consequences: the model has never seen a zero-demand hour and will over-predict the quietest hours slightly; said in the report.
