# STARAI-ICU-Alarm

This project was built for the STAR AI Data Competition (Track 4), where the goal is to predict ICU patient mortality using high-resolution physiological data. The competition focuses on real-world clinical challenges, especially how to make predictions that are not only accurate but also useful in practice, where false alarms and unreliable predictions can have serious consequences.

Instead of just training a model and reporting accuracy, this project is designed as a full pipeline that tries to reflect how a system like this would actually be used in a hospital setting. That includes not only predicting mortality, but also measuring uncertainty, simulating alarm policies, and analyzing fairness across different patient groups.

The core idea is to move beyond a single prediction and think about how decisions are made from those predictions.

---

## What the project does

At a high level, the pipeline takes ICU patient data, processes it into features, trains a model to predict mortality, and then adds additional layers to evaluate how reliable and clinically useful those predictions are.

The main components are:

* A mortality prediction model trained on ICU time-series data
* An uncertainty module that flags predictions the model is not confident about
* An alarm policy engine that simulates when alerts would be triggered
* A fairness analysis that checks performance across subgroups

This structure follows the STAR AI competition guidelines, where teams are encouraged to think about deterioration signals, alarm burden, and real-world usability instead of just leaderboard performance.

---

## How this relates to STAR AI

The STAR AI competition evaluates models based on AUC for mortality prediction, but also emphasizes deeper analysis such as early warning signals, alarm fatigue, and system reliability.

This project aligns with that by:

* Predicting mortality as the main task (competition target)
* Incorporating uncertainty to identify unreliable predictions
* Simulating alarm policies to study tradeoffs between sensitivity and false alarms
* Analyzing fairness across patient subgroups

The goal is not just to get a strong AUC, but to understand how the model behaves and how it would perform in a real ICU environment.

---

## Pipeline overview

The pipeline is structured as a sequence of steps:

1. Data loading
   Patient data is loaded from preprocessed HiRID files. These include time-series vital signs and clinical measurements.

2. Feature engineering
   Raw time-series data is aggregated into features that summarize each patient’s ICU stay. This includes statistics like averages, trends, and variability.

3. Model training
   A gradient boosting model (LightGBM-style) is trained to predict whether a patient dies during their ICU stay.

4. Prediction
   The trained model is used to generate probabilities for mortality on validation or test data.

5. Uncertainty estimation
   Multiple models (e.g., cross-validation folds) are compared to measure disagreement. High disagreement indicates uncertainty, and those cases can be flagged for human review.

6. Alarm policy simulation
   Different rules are tested for triggering alerts (for example, threshold-based triggers). This helps evaluate how often the system would raise alarms and how many would be false positives.

7. Analysis and evaluation
   The pipeline computes performance metrics such as AUC and also evaluates fairness across groups like age, sex, and ICU stay length.

---

## Project structure

The main code lives in the `star_ai_pipeline` folder:

* `data_loader.py` handles loading and preparing the dataset
* `features.py` builds features from raw data
* `train.py` trains the model
* `predict.py` runs inference
* `uncertainty_module.py` measures prediction uncertainty
* `alarm_policy_engine.py` simulates alert strategies
* `analysis.py` computes metrics and evaluations
* `equity_dashboard.py` looks at fairness across subgroups
* `run_pipeline.py` ties everything together

There are also:

* `results/` for outputs (not included in the repo)
* `logs/` for runtime logs (not included)
* small reference CSV files to help understand variables

Large datasets and generated outputs are intentionally excluded from the repository.

---

## How to run the project

1. Clone the repository

```
git clone https://github.com/aryabhanushali/STARAI-ICU-Alarm.git
cd STARAI-ICU-Alarm
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the pipeline

```
python star_ai_pipeline/run_pipeline.py
```

Note: The original HiRID dataset is not included due to size and access restrictions. You will need to provide your own data in the expected format.

---

## Results and evaluation

The model is evaluated using AUC, which is the main metric used in the competition. In addition to that, the project looks at:

* Model disagreement as a measure of uncertainty
* Tradeoffs between true positives and false alarms under different alarm policies
* Performance differences across patient subgroups

This makes it possible to understand not just how accurate the model is, but also how reliable and usable it would be in practice.

---

## Notes on data

This repository does not include the full dataset. The HiRID dataset is large and contains sensitive clinical information, so only small reference files are included.

Anything that is large, generated, or potentially sensitive (such as processed patient data or model outputs) is excluded using `.gitignore`.

---

## Future improvements

Some directions for improvement include:

* Modeling intermediate deterioration signals (e.g., respiratory or circulatory failure)
* Improving feature engineering for time-series data
* Exploring sequence models like LSTMs or Transformers
* Designing more adaptive alarm policies
* Improving uncertainty estimation with better calibration

---

## Summary

This project builds a mortality prediction system that goes beyond a standard ML model by focusing on uncertainty, alarm design, and fairness. It is designed to reflect the kinds of considerations that matter in real clinical systems, not just benchmark performance.
