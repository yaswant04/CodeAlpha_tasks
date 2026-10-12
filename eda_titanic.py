"""
CodeAlpha Data Analytics Internship — TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
Dataset: Titanic - Machine Learning from Disaster (the classic Kaggle dataset)
Source : https://www.kaggle.com/competitions/titanic  (mirrored copy used here:
         https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

Goal: Ask meaningful questions, explore structure/types, find trends & anomalies,
      test simple hypotheses with stats, and flag data quality issues.
"""

import pandas as pd
import numpy as np
from scipy import stats

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

DATA_PATH = "../data/titanic.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def structure_overview(df: pd.DataFrame):
    print("=" * 70)
    print("1. DATA STRUCTURE OVERVIEW")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Column dtypes:")
    print(df.dtypes, "\n")
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Summary statistics (numeric columns):")
    print(df.describe(include=[np.number]).T, "\n")


def data_quality_issues(df: pd.DataFrame):
    print("=" * 70)
    print("2. DATA QUALITY ISSUES (missing values, duplicates, anomalies)")
    print("=" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    quality = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    quality = quality[quality["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    print("Missing values by column:")
    print(quality, "\n")

    dupes = df.duplicated().sum()
    print(f"Fully duplicated rows: {dupes}\n")

    # Anomaly check: fares of 0 for passengers who weren't crew (odd but real -
    # some Titanic tickets genuinely had $0 fare, e.g. staff/comp tickets)
    zero_fare = (df["Fare"] == 0).sum()
    print(f"Passengers with Fare == 0 (possible free/comp tickets): {zero_fare}")

    # Anomaly check: implausible ages
    if df["Age"].max() > 80 or (df["Age"].dropna() < 0).any():
        print(f"Age range: {df['Age'].min()} to {df['Age'].max()} (plausible for 1912 passenger list)")
    print()


def key_questions(df: pd.DataFrame):
    print("=" * 70)
    print("3. MEANINGFUL QUESTIONS ABOUT THE DATA")
    print("=" * 70)

    overall_rate = df["Survived"].mean()
    print(f"Q1: What was the overall survival rate? -> {overall_rate:.1%}\n")

    print("Q2: Did survival rate differ by passenger class (Pclass)?")
    print(df.groupby("Pclass")["Survived"].mean().round(3), "\n")

    print("Q3: Did survival rate differ by sex?")
    print(df.groupby("Sex")["Survived"].mean().round(3), "\n")

    print("Q4: Did survival rate differ by port of embarkation?")
    print(df.groupby("Embarked")["Survived"].mean().round(3), "\n")

    print("Q5: Did having family aboard (SibSp/Parch) relate to survival?")
    df_tmp = df.copy()
    df_tmp["family_size"] = df_tmp["SibSp"] + df_tmp["Parch"]
    df_tmp["has_family"] = df_tmp["family_size"] > 0
    print(df_tmp.groupby("has_family")["Survived"].mean().round(3), "\n")


def hypothesis_tests(df: pd.DataFrame):
    print("=" * 70)
    print("4. HYPOTHESIS TESTING")
    print("=" * 70)

    # H1: Sex and survival are independent (chi-square test of independence)
    contingency_sex = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency_sex)
    print("H1: 'Survival is independent of sex' (chi-square test)")
    print(f"    chi2 = {chi2:.2f}, p-value = {p:.2e}")
    print("    -> REJECT H1: sex is strongly associated with survival.\n"
          if p < 0.05 else "    -> Fail to reject H1.\n")

    # H2: Sex and Pclass are independent
    contingency_class = pd.crosstab(df["Pclass"], df["Survived"])
    chi2b, pb, _, _ = stats.chi2_contingency(contingency_class)
    print("H2: 'Survival is independent of passenger class' (chi-square test)")
    print(f"    chi2 = {chi2b:.2f}, p-value = {pb:.2e}")
    print("    -> REJECT H2: passenger class is strongly associated with survival.\n"
          if pb < 0.05 else "    -> Fail to reject H2.\n")

    # H3: Mean age of survivors vs non-survivors differ (independent t-test)
    age_survived = df.loc[df["Survived"] == 1, "Age"].dropna()
    age_died = df.loc[df["Survived"] == 0, "Age"].dropna()
    t_stat, p_t = stats.ttest_ind(age_survived, age_died, equal_var=False)
    print("H3: 'Mean age is the same for survivors and non-survivors' (Welch t-test)")
    print(f"    survivors mean age = {age_survived.mean():.1f}, "
          f"non-survivors mean age = {age_died.mean():.1f}")
    print(f"    t = {t_stat:.2f}, p-value = {p_t:.4f}")
    print("    -> REJECT H3: there is a statistically significant age difference.\n"
          if p_t < 0.05 else "    -> Fail to reject H3: no significant age difference.\n")


def save_clean_summary(df: pd.DataFrame):
    summary = (
        df.assign(family_size=df["SibSp"] + df["Parch"])
        .groupby(["Pclass", "Sex"])
        .agg(
            passengers=("PassengerId", "count"),
            survival_rate=("Survived", "mean"),
            avg_age=("Age", "mean"),
            avg_fare=("Fare", "mean"),
        )
        .round(2)
    )
    summary.to_csv("../data/titanic_eda_summary.csv")
    print("Saved grouped summary table to ../data/titanic_eda_summary.csv")


def main():
    df = load_data()
    structure_overview(df)
    data_quality_issues(df)
    key_questions(df)
    hypothesis_tests(df)
    save_clean_summary(df)


if __name__ == "__main__":
    main()
