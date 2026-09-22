# Lesson 02: pandas and data thinking

## Goal
Learn how ML data moves from messy tables into a clean feature matrix X and target vector y.

## Core ideas
A **sample** is one observation. A **feature** is an input variable. A **target/label** is what a supervised model should predict.

```python
import pandas as pd

df = pd.DataFrame({
    "frequency_mhz": [5, 10, 15, 20, 25],
    "snr_db": [3.2, 8.1, None, 14.5, 18.0],
    "detected": [0, 0, 1, 1, 1],
})

print(df)
print(df.info())
print(df.describe())
print(df.isna().sum())
```

## Selecting data
```python
X = df[["frequency_mhz", "snr_db"]]
y = df["detected"]

high_freq = df[df["frequency_mhz"] >= 15]
```

## Missing values
Never blindly delete or fill missing values. Ask why they are missing and whether the missingness itself contains information.

```python
df["snr_db"] = df["snr_db"].fillna(df["snr_db"].median())
```

## Exercise
Create a DataFrame with 100 synthetic measurements containing frequency, amplitude, noise power and a binary detection label. Deliberately insert five missing values.

Then:
1. inspect types and dimensions;
2. count missing values;
3. compute summary statistics;
4. select measurements above 20 MHz;
5. create an SNR feature;
6. split columns into X and y.

## Critical concept: leakage
Data leakage happens when training information contains something that would not actually be available at prediction time. A spectacular validation score can therefore be evidence of a broken experiment.

## Done when
You can turn a table into features and targets and explain missing data and leakage.
