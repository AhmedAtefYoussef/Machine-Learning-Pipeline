

German International University of Applied Sciences
Informatics and Computer Science
## Dr. Caroline Sabty
TA Nouran Khaled
TA Sandra Samuel
## Machine Learning, Winter 2026
## Project 1
1  Rush Hour: Predicting Bike-Sharing Demand - The Chained
## Pipeline Project
## 1.1  Overview
A city bike-sharing operator has to decide, every hour, how many bikes should be waiting at its stations.
Too few and customers find empty docks and leave. Too many and the operator wastes money moving
idle bikes around the city with trucks. Every one of those decisions starts with the same question:how
many bikes will be rented in this hour?
In this project you will answer that question. You are given two years of hourly records from a bike-
sharing system, with the weather, the calendar, and the number of bikes rented in each hour (cnt). Your
job is to predictcntand, at the end, to decide automatically whether an hour is a high-demand hour.
Unlike a typical single-model project, this project is structured as achain of five phases. Each phase
implements one algorithm from the course, and the output of each phase becomes a required input to
the next. You will build regression models with a focus ongeneralizationthroughregularization
techniques, and finish by converting the problem into a classification task withlogistic regression.
The task requires accuracy, robustness, and the ability to justify every modeling choice you make, not
every step is spelled out for you, and that is intentional.
Read this before you start.Not every column is informative, and some columns carry the same
information as others. Deciding which is which (with evidence from your own models) is part of your
work. The data also has atime orderand a fewoperational quirks. A pipeline that treats every
row as an independent, ordinary observation will look better on paper than it really is. Look at your
data before you model it.
## 1.2  Project Timeline
•Team Formation:
## – Start Date: September 29th
– End Date: October 1stat 12:00 pm
•Project Deadlines
## – Start Date: September 29th
## – Final Submission Deadline: October 18th.
– Evaluation(all team members): in the tutorial slots of the week after the submission; the
schedule will be announced. Teams with members from different tutorials are discussed on
their days off (see Team Formation).
## 1

## 1.3  Team Formation
To participate in this project, you must first complete theteam registration form. Teams consist of2–
3 members, who may come from the same tutorial or from different tutorials. Registration is mandatory
for participation. Please fill in the form at the following link: https://forms.gle/ffri4CFWb6DKy2Ab9.
Note:teams whose members come fromdifferent tutorialsare evaluated (live discussion) on their
days offrather than on a regular tutorial day. The schedule will be announced.
## 1.4  Dataset Description
The dataset containshourlyrecords of a bike-sharing system (Capital Bikeshare, Washington D.C.) over
two years(2011–2012). Each row describes one hour: the calendar, the weather, and the total number
of bikes rented in that hour.
Files you receive
You receive three files uploaded to the CMS:
•train.csv-10,886 hourly records with all features and the targetcnt. This is theonly labeled
data you have; your train/validation split comes from this file.
•test.csv- 574 hourly records with the same features butnocnt. You will submit predictions for
these hours (the hidden test set).
•sample_submission.csv- the exact format of your prediction file:instant,cnt.
## Columns
•instant: record index (unique identifier of the hourly record)
•dteday: calendar date (YYYY-MM-DD)
•season: season (1: winter, 2: spring, 3: summer, 4: fall)
•yr: year (0: 2011, 1: 2012)
•mnth: month (1 to 12)
•hr: hour of the day (0 to 23)
•holiday: whether the day is a public holiday (1) or not (0)
•weekday: day of the week
•workingday: 1 if the day is neither a weekend nor a holiday, otherwise 0
•weathersit: weather situation
–1: clear, few clouds, partly cloudy
–2: mist + cloudy, mist + broken clouds, mist + few clouds, mist
–3: light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds
–4: heavy rain + ice pallets + thunderstorm + mist, snow + fog
•temp: normalized temperature in Celsius,(t−t
min
## )/(t
max
## −t
min
## )witht
min
## =−8,t
max
## = +39
•atemp: normalized “feels-like” temperature in Celsius, witht
min
## =−16,t
max
## = +50
•hum: normalized humidity (divided by 100)
•windspeed: normalized wind speed (divided by 67)
## 2

•cnt: number of bikes rented in that hour (target variable;train.csvonly)
Some hours are missing from the timeline. Several columns are strongly related to one another, several
features act on demand in a clearly non-linear way, anddtedayis not a feature you can use as-is. Handling
all of this appropriately is part of the project.
## 1.5  Objective
The objective is to build a chain of regression and classification models that together (a) predictcntas
accurately as possible while remaining robust to overfitting (including on data it has never seen) and (b)
produce a final binary classifier built on top of the regression pipeline’s own results.
## 1.6  Evaluation Metric
Regression phases (Phases 1–4) are evaluated using theR-squared (R
## 2
## )score:
## R
## 2
## = 1−
## P
n
i=1
## (y
i
## −ˆy
i
## )
## 2
## P
n
i=1
## (y
i
## − ̄y)
## 2
•nis the number of samples.
## •ˆy
i
is the predicted number of rentals.
## •y
i
is the actual number of rentals.
- ̄yis the mean of the actual rentals.
All metrics in the phases are computed on yourvalidation set(the 20% oftrain.csvheld out with
your team seed), reported asR
## 2
andRMSE. The final phase (Phase 5) is evaluated on the validation
set usingaccuracy,F1-score, andROC-AUC, since it is a classification task.
The predictions you submit fortest.csvarenotused to tune or score your phases. The teaching team
compares them with the hidden labels only as a final check, during the evaluation of your team, that
your final model avoided overfitting and underfitting.
## 1.7  Requirements Checklist
Everything in this subsection is stated plainly and without ambiguity. The phase descriptions later in
this document deliberately do not repeat these requirements in full, they assume you already know them
from here.
Algorithms you must implement, all five, in this order:
•Gradient Descent- implemented from scratch (nosklearn .fit()/.predict()for this phase).
Must support a configurable learning rate and report convergence behavior.
•Polynomial Regression- feature expansion to a self-chosen degree, fit using your own Phase 1
gradient descent code, notsklearn’s optimizer.
•Bias-Variance Tradeoff- an empirical diagnosis of your Phase 2 model’s fit quality, produced by
varying model complexity and comparing training vs. held-out performance.
•Regularization- all three of L1 (Lasso), L2 (Ridge), and Elastic Net must be implemented and
compared. Not one, not two, but all three.
•Logistic Regression- a binary classifier trained on a target you derive yourself fromcnt, using
only the features that survive Phase 4’s selection.
## 3

Non-negotiable structural expectations:
•Chaining is mandatory.Each phase must consume the numeric artifact(s) produced by the pre-
vious phase (see the summary table below for exactly which artifact). A phase built independently of
your own team’s previous phase will not receive credit for pipeline integrity, even if it is individually
correct.
•Team seed.Your team must use the seed defined in Section 1.8. Using a shared, default, or
unseeded split is a violation of these requirements.
•No prescribed method beyond this checklist.Beyond what is stated here, your team is
expected to determine the correct technique, plot, threshold, or search strategy yourselves. The
absence of a detailed recipe in a phase’s instructions is intentional, not an omission.
•Written justification required.Every phase deliverable must include a short justification, in
your own words, for every choice your team made that wasn’t handed to you (learning rate, degree,
which features to expand, regularization strength, classification threshold, etc.).
•Expectation before result.Each phase in your notebook starts with a short markdown cell titled
Expectation, writtenbeforeyou run the phase: what do you expect to happen, and why? It ends
with anOutcomecell: what actually happened, and what surprised you. Wrong expectations cost
nothing (missing or obviously after-the-fact ones do).
•Test predictions.Submit the filled-insample_submission.csvwith your predictions fortest.csv,
produced by the regression model your pipeline ends up recommending.
•Evaluation.Every team member must be able to explain any part of the notebook and change
and re-run it on the spot.
## 1.8  Team Seed & Data Split
Because teams have 2–3 members, the seed is derived from the whole team rather than a single student,
so it cannot be freely chosen and does not depend on who types it in or in what order.
•Take every team member’s student ID (as astring),sort them, and concatenate them into a
single stringconcatenated_stringwith underscores (e.g."34521_40218_41190“).
•Computeseed = int(hashlib.sha256(concatenated_string.encode()).hexdigest(), 16) %
- For the example IDs above, the seed is41698; use this to check your code.
•Use thisseedintrain_test_split(..., random_state=seed)ontrain.csvto create an80%
train / 20% validationsplit (a single call withtest_size=0.20). This is the only seeded split.
There is no third split to make: all your evaluation and tuning uses the validation set, andtest.csv
is only used for the final predictions.
•Phase 3 additionally asks forone chronological splitoftrain.csv. That split does not use the
seed; it is determined by the dates.
Using the test file.test.csvhas no labels, so never split it, shuffle it, or use it to tune anything. Fit
every scaler, encoder, polynomial expansion and feature selection on yourtrainingportion only, evaluate
and tune on the validation portion, and apply exactly the same fitted transformations totest.csvat
the very end. Refitting your final model on train + validation before predicting is allowed if you justify
it.
Reference code(replace the IDs with your own team’s):
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
## 4

# --- team seed ---
ids = sorted(["34521", "40218", "41190"])        # 1. your team’s student IDs (strings), sorted
concatenated_string = "_".join(ids)              # 2. "34521_40218_41190"
seed = int(hashlib.sha256(concatenated_string.encode()).hexdigest(), 16) % 100000
# --- data ---
train_all = pd.read_csv("train.csv")             # labeled
test_X = pd.read_csv("test.csv")                 # unlabeled (hidden test set)
# --- the only seeded split: train / validation (80 / 20) ---
train_df, val_df = train_test_split(train_all, test_size=0.20, random_state=seed)
# Fit scalers / encoders / feature selection on train_df ONLY, then apply them
# to val_df and to test_X. Tune everything on val_df.
# --- at the end: fill in sample_submission.csv with your test predictions ---
# preds = final_model.predict(prepared_test_X)
# sub = pd.read_csv("sample_submission.csv")
# sub["cnt"] = preds                             # same row order as test.csv
# sub.to_csv("sample_submission.csv", index=False)
Because the seed depends on your team’s exact roster, no two teams will produce the same train/validati-
on split, the same gradient descent trajectory, the same chosen polynomial degree, the same regularization
results, or the same classification threshold. A generic answer produced without your team’s own inter-
mediate numbers will not match what your pipeline actually needs at each step.
1.9  Phase-by-Phase Instructions
These instructions intentionally stop short of a full recipe. You already know, from the Requirements
Checklist above, exactly which algorithm each phase requires and which structural rules apply. What
method, plot, split, or threshold logic best satisfies that requirement is for your team to determine and
justify.
## Phase 1 - Gradient Descent
Fit a linear model to your prepared feature set using your own gradient descent implementation.
•Consumes: your team’s seeded train/val split.
•Must produce, for Phase 2 to consume: a final weight vector and the hyperparameters that got you
there.
•Left for you to determine: how to prepare and scale the raw features; how to representhr,season,
weathersit, and the information hidden indteday; what learning rate and stopping criterion to
use; and how to demonstrate that your chosen settings actually converge rather than diverge or
stall.
Optional bonus (+5%) - the operator’s real cost.For the operator, under-predicting demand
(empty stations, lost customers) is worse than over-predicting it (a few idle bikes). Suppose under-
predicting costs3×as much as over-predicting by the same amount. Modify your from-scratch
gradient descent to minimize this asymmetric loss instead of MSE, derive its gradient in your report,
and show how the fitted predictions shift compared to your MSE model.
## Phase 2 - Polynomial Regression
Extend Phase 1’s model to capture non-linear structure in the data, continuing to use your own gradient
descent code rather than a library optimizer.
•Consumes: Phase 1’s weight vector, used as your starting point rather than a random initialization.
## 5

•Must produce, for Phase 3 to consume: a chosen polynomial degree and the resulting train/validation
performance at that degree.
•Left for you to determine: which features are worth expanding (expanding every numeric column
is neither required nor advisable), what degree to use, and how to initialize the expanded weight
vector from Phase 1’s shorter one.
Phase 3 - Bias-Variance Tradeoff
Determine whether the model your team built in Phase 2 is under-fit, over-fit, or reasonably fit – empi-
rically, not by assertion.
•Consumes: Phase 2’s degree and feature-expansion choice, which should anchor whatever range you
investigate.
•Must produce, for Phase 4 to consume: a specific, evidence-based diagnosis, and a target degree/-
complexity to carry forward (this may or may not match Phase 2’s).
•Also required the time question: your seeded split mixes hours from the same days into training and
validation, whereas new data will arrive as unseen time periods. Re-evaluate your model onone
chronological splitoftrain.csv(train on the earlier dates, validate on the most recent ones)
and compare the two estimates. Explain any gap. Which estimate should you trust for unseen data,
and does it change your diagnosis?
•Left for you to determine: what to vary, what to measure, where to place the chronological cut, and
what evidence (numeric or visual) would actually distinguish bias from variance on this dataset.
Phase 4 - Regularization (L1, L2, Elastic Net)
Address whatever fitting problem Phase 3 diagnosed, using all three regularization methods listed in the
## Requirements Checklist.
•Consumes: Phase 3’s diagnosis and target degree/complexity (this should directly inform how you
search for regularization strength).
•Must produce, for Phase 5 to consume: a final hyperparameter choice per method and the feature
subset that L1-type regularization leaves non-zero.
•Also required - a verdict for every original column: for each input column in the dataset, state
whether your evidence says it isuseful,redundant(its information is already carried by another
column), oruninformative, and show the numbers from your own models that support each
verdict. “Lasso set it to zero” is evidence; “it sounds irrelevant” is not.
•Left for you to determine: the search range and strategy forλ(andl1_ratio), and how to fairly
compare the three methods against each other.
## Phase 5 - Logistic Regression
Convert the regression problem into a binary classification problem (is this ahigh-demand hour?) and
solve it using only the features Phase 4 judged worth keeping.
•Consumes: the surviving feature subset from Phase 4.
•Must produce: a trained classifier, an evaluation on your validation set, and a short pipeline retro-
spective tying all five phases’ numbers together.
•Left for you to determine: how to define “high demand” fromcntin a way that is meaningful to
the operator and not trivially biased (a single global threshold has an obvious flaw, identifying
and fixing that flaw is part of the exercise), and which evaluation metrics best suit whatever class
balance results.
## 6

## 1.10  Deliverables Summary
PhaseConsumes from previous pha-
se
Produces for next phase
- Gradient DescentTeam’s seeded train/val splitWeight vector, learning rate, itera-
tion count
## 2.  Polynomial  Re-
gression
Phase 1 weight vector (as GD init)Chosen degree, expanded feature
list
3.Bias–Variance
## Tradeoff
Phase 2 degree, feature listDiagnosis (random vs. chronologi-
cal), revised target degree
- RegularizationConfirmed degree, diagnosisλper method, surviving features,
verdict per column
## 5. Logistic Regressi-
on
Surviving feature setFinal classifier, pipeline retrospec-
tive
## 1.11  Submission
Your submission (one per team) is made through https://forms.gle/7zJr24GAnrpCYimdA. It contains:
a)Notebook(.ipynb) that runs top-to-bottom without errors and contains all five phases, including
the Expectation and Outcome cells.
b)Short report(PDF, at most 6 pages) that explains, for every phase, the choices made where the
instructions left something open and the rationale behind them, referencing your own team’s actual
numbers, not general theory.
c)Filled-insample_submission.csvwith your final model’s predictions fortest.csv(columns
instant,cnt; same rows, same order). The teaching team evaluates it against the hidden labels
as part of your team’s evaluation, to check that your final model really avoided overfitting and
underfitting.
## 1.12  Rules
•External Data: The use of external data is not allowed. All models must be trained solely on the
provided dataset.
•Team Size: Teams of 2–3 members are allowed, from the same tutorial or from different tutorials.
Teams with members from different tutorials are evaluated on their days off.
•Seed Integrity: Submissions may be spot-checked by recomputing your team’s seed from your
registered roster and re-running your Phase 1 code to confirm the weights carried into Phase 2
match. Mismatches indicate a phase was completed independently of your own team’s pipeline.
## 7