# %% [markdown]
# ## Expectation — Phase 3: Bias–Variance Tradeoff
#
# *Written before running the phase (see the git history of this file).*
#
# **What we do.** We take the Phase 2 design (its degree and feature list, loaded from `artifacts/p2.json`) as the anchor and move complexity both below and above it on a nested ladder of ten designs, from "no hour information" (13 weights) up to "month × working day × hour" (about 900 weights). We also vary the pure polynomial degree, and the amount of training data. Every design is scored three ways: on our seeded validation set, on whole held-out days inside the training portion, and on one chronological split (train before 1 July 2012, validate after).
#
# **What we expect, and why.**
#
# 1. *Phase 2 is not over-fit.* Its training and validation R² should differ by less than 0.01. If anything we expect it to be slightly under-fit: letting the weather effect depend on the hour should still raise validation R² by 0.01–0.02.
# 2. *Degree alone will look flat.* Raising the power of temperature and humidity from 3 to 8 should move neither curve by more than 0.002. A flat line is not evidence of a good fit, which is why we need the wider ladder.
# 3. *Variance appears late.* On the ladder we expect validation R² to peak somewhere around 250–450 weights and then fall while training R² keeps rising. The first clearly harmful levels should be the month × hour blocks, where many cells hold only a handful of training rows.
# 4. *Learning curves.* At Phase 2 complexity, validation R² should stop improving after roughly half the training days (a bias signature). At the top of the ladder the train–validation gap should be visibly wider with little data and shrink as data is added (a variance signature).
# 5. *The time question.* We expect the seeded estimate and the held-out-day estimate to be almost equal (within 0.005). Our model has no way to memorise a particular day, so sharing days between training and validation should hardly help it. The chronological estimate should be clearly lower, by 0.04–0.08, because the model has to extrapolate a growth trend into six months it has never seen, having observed July–December only once.
# 6. *Which estimate to trust.* For the hidden test set, which is the 20th of each month and so lies inside the observed period, the held-out-day estimate is the right one. For genuinely future months the chronological estimate is the honest one. We expect the diagnosis to stay "bias first, variance only at the top of the ladder" under all three, but the chronological split should prefer a simpler model than the seeded split does, especially disliking the blocks that let the trend vary by hour.
