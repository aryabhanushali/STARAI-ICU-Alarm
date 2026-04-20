STARAI-ICU-Alarm

This project was built for the STAR AI Data Competition (Track 4), where the goal is to predict ICU patient mortality using high-resolution physiological data from the HiRID dataset.

The main idea behind this project is not just to build a model that predicts whether a patient will die, but to think about how that prediction would actually be used in a real ICU. That means looking at things like uncertainty, alarm fatigue, and fairness, not just accuracy.

What this project does

At a high level, the pipeline:

takes raw ICU time-series data (vitals, labs, etc.)
converts it into structured features for each patient
trains a model to predict mortality
evaluates how reliable those predictions are
simulates how alerts would be triggered
checks if the model performs differently across patient groups

So instead of just outputting a prediction, the system tries to answer:
can we trust this prediction, and should we act on it?

Pipeline overview
1. Data processing

The HiRID dataset is very large, so it’s read in chunks instead of all at once. The data is resampled into 2-minute windows to reduce noise and better match how ICU monitoring systems work.

Basic patient info like age and sex is also added here.

2. Feature engineering

Each patient has a different length of stay, so the time-series data is converted into a fixed set of features.

These include:

mean, min, max, standard deviation
trends over time (slopes)
missingness (how often a variable is recorded)
clinical flags like low blood pressure or oxygen

The idea is to summarize each patient’s condition in a way that still reflects real clinical signals.

3. Model training

A gradient boosting model (LightGBM) is trained using 5-fold cross-validation.

the dataset is imbalanced (~6% mortality), so class weights are used
a logistic regression model is also trained as a baseline
predictions from both are later combined
4. Prediction + ensembling

The 5 models from cross-validation are combined using rank normalization and weighted averaging.

Then the final prediction is a blend of:

LightGBM (main model)
logistic regression (adds some linear signal)
What makes this project different
Uncertainty module

Instead of trusting every prediction equally, the model checks how much the different folds disagree.

if all models agree → prediction is more reliable
if they disagree → prediction is uncertain

Uncertain cases can be flagged instead of blindly acted on.

Alarm policy simulation

The model output is used to simulate different alert strategies, like:

trigger alert if risk > threshold
require both high risk and abnormal vitals
multi-organ failure triggers

Each policy is evaluated based on:

how many real cases it catches
how many false alarms it produces

This is important because too many false alarms leads to alarm fatigue in ICUs.

Fairness analysis

The model is evaluated across different groups:

age
sex
ICU stay length

This helps check if the model performs worse on certain types of patients.

Results (high level)
Cross-validation AUC: ~0.87
Kaggle public AUC: ~0.95

Some observations:

recent vital signs (last values) are very important
missing data is actually informative
simple models like logistic regression still perform well
Notes on data

The full dataset is not included in this repo because:

it’s very large
it contains sensitive clinical data

Only small reference files are included. Everything else is excluded using .gitignore.

Limitations
performance is worse for short ICU stays (not enough data early on)
data comes from a single hospital
not a real-time system (works on full patient stays)
How to run
git clone https://github.com/aryabhanushali/STARAI-ICU-Alarm.git
cd STARAI-ICU-Alarm
pip install -r requirements.txt
python star_ai_pipeline/run_pipeline.py
