# Predicting problematic internet use in children

Kaggle competition: [Child Mind Institute – Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) (2024).

## Why it matters
Problematic internet use in children is linked to anxiety, depression and poor sleep, but it is usually measured with a 20-question clinical questionnaire (PCIAT). This project predicts the resulting **Severity Impairment Index (SII: 0 none, 1 mild, 2 moderate, 3 severe)** from data that is cheaper to collect: physical exams, fitness tests, a sleep questionnaire and screen-time habits. The aim is early screening.

## Data
- 3,960 children, of whom 2,736 have an SII label (58% none, 27% mild, 14% moderate, 1% severe).
- About 1,000 children also wore a wrist accelerometer for several weeks.
- **Leakage guard:** the 22 PCIAT questionnaire columns define the target and are absent at test time, so they are excluded from every model.

## Method
1. **Features:** physical, fitness, body-composition and questionnaire columns, plus a few age-adjusted interactions (BMI × age, internet hours × age, fat-mass ratios). Accelerometer recordings are summarised per child from worn periods only: activity level, night vs day movement, light exposure and wear time. This is computed lazily with Polars.
2. **Models:** LightGBM, XGBoost and CatBoost, trained twice: once on SII, and once with a Tweedie objective on the 0–80 PCIAT total, a skewed, zero-heavy score. The two blends are standardised and averaged.
3. **From regression to classes:** three cut-off thresholds are tuned to maximise quadratic weighted kappa (QWK), the competition metric.
4. **Validation:** stratified 5-fold cross-validation repeated with 3 seeds. Thresholds tuned on the same out-of-fold predictions they are scored on give an optimistic score, so the main number uses **nested thresholds**: tuned on 4/5 of the out-of-fold predictions and scored on the held-out 1/5.

## Results (out-of-fold, 2,736 children, 5-fold × 5 seeds)

| Version | Features / targets | **QWK (nested thresholds)** |
|---|---|---|
| v1 | tabular, sii regression, 3 seeds | 0.452 |
| v1 | + accelerometer | 0.446 |
| **v2** | tabular, **sii model + PCIAT-total Tweedie model, blended**, 5 seeds | **0.460** |
| v2 | same + accelerometer | 0.454 |

Within v2: the sii-only blend scores 0.453 and the PCIAT-total model alone 0.461. Predicting the finer 0–80 questionnaire total gives the trees more signal than the four coarse classes. For reference, the strongest public solutions scored around 0.45–0.48 on the private leaderboard.

**What the experiments showed**
- Adding accelerometer summaries did **not** improve cross-validated QWK. Only about a third of children have recordings, and simple summaries add noise. The simpler tabular model is the one to prefer.
- The most important features are internet hours (age-adjusted), age, the sleep-disturbance score (SDS), height and weight.
- **Severe cases:** v1 never predicted class 3. v2 catches 8 of the 34 severe children (1% of the data), and most of the rest land in *moderate*. A screening tool would still need a high-risk flag or more severe-case data.

## Run it

All the code is in one notebook, [problematic_internet_use.ipynb](problematic_internet_use.ipynb), with an explanation before each step of what it does, why and how it works.

```bash
pip install pandas polars lightgbm xgboost catboost scikit-learn scipy jupyter
kaggle competitions download -c child-mind-institute-problematic-internet-use -p data && (cd data && unzip -q *.zip)
jupyter notebook problematic_internet_use.ipynb   # Run All: writes results/metrics.json, feature importance, submission.csv
```
The notebook also runs on Kaggle with the competition attached; it finds `/kaggle/input` automatically. I ran it on Kaggle because the accelerometer data is about 6 GB.

`results/metrics.json` holds all scores, thresholds and the confusion matrix.

![Confusion matrix and feature importance](results/cmi_results.png)
