"""
CodeAlpha Data Analytics Internship — TASK 3: DATA VISUALIZATION
====================================================================
Dataset: Titanic passenger data (same dataset used in Task 2 EDA)
Goal: Turn the raw data into clear charts that tell a story and support
      decision-making / insight generation. Built with Matplotlib + Seaborn.

Outputs (saved to ../outputs/):
  1. titanic_dashboard.png   - 6-panel overview dashboard
  2. survival_by_class_sex.png - focused insight chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
DATA_PATH = "../data/titanic.csv"
OUT_DIR = "../outputs"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Survived_label"] = df["Survived"].map({0: "Died", 1: "Survived"})
    df["family_size"] = df["SibSp"] + df["Parch"]
    return df


def build_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Titanic Passenger Dataset — Exploratory Dashboard", fontsize=18, fontweight="bold")

    # 1. Overall survival counts
    ax = axes[0, 0]
    sns.countplot(data=df, x="Survived_label", ax=ax, hue="Survived_label", legend=False)
    ax.set_title("Overall Survival Count")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Passengers")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)

    # 2. Survival rate by passenger class
    ax = axes[0, 1]
    class_rate = df.groupby("Pclass")["Survived"].mean().reset_index()
    sns.barplot(data=class_rate, x="Pclass", y="Survived", ax=ax, hue="Pclass", legend=False)
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)

    # 3. Survival rate by sex
    ax = axes[0, 2]
    sex_rate = df.groupby("Sex")["Survived"].mean().reset_index()
    sns.barplot(data=sex_rate, x="Sex", y="Survived", ax=ax, hue="Sex", legend=False)
    ax.set_title("Survival Rate by Sex")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)

    # 4. Age distribution by survival
    ax = axes[1, 0]
    sns.kdeplot(data=df, x="Age", hue="Survived_label", fill=True, common_norm=False, alpha=0.4, ax=ax)
    ax.set_title("Age Distribution: Survivors vs. Non-survivors")
    ax.set_xlabel("Age")

    # 5. Fare distribution by class (log scale to tame outliers)
    ax = axes[1, 1]
    sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax, hue="Pclass", legend=False)
    ax.set_yscale("symlog")
    ax.set_title("Fare Distribution by Class (log scale)")
    ax.set_xlabel("Passenger Class")

    # 6. Survival rate by family size
    ax = axes[1, 2]
    fam_rate = df.groupby("family_size")["Survived"].mean().reset_index()
    sns.lineplot(data=fam_rate, x="family_size", y="Survived", marker="o", ax=ax)
    ax.set_title("Survival Rate vs. Family Size Aboard")
    ax.set_xlabel("Family Size (SibSp + Parch)")
    ax.set_ylabel("Survival Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f"{OUT_DIR}/titanic_dashboard.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_focused_insight_chart(df: pd.DataFrame):
    """A single, presentation-ready chart: survival rate by class AND sex."""
    pivot = df.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=pivot, x="Pclass", y="Survived", hue="Sex", ax=ax)
    ax.set_title("Survival Rate by Passenger Class and Sex\n(Titanic, 1912)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)
    ax.legend(title="Sex")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0%%" if False else lambda v: f"{v:.0%}")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/survival_by_class_sex.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    df = load_data()
    build_dashboard(df)
    build_focused_insight_chart(df)
    print("\nData story: Being female and travelling 1st class were the two strongest")
    print("predictors of survival on the Titanic — a 1st class woman had roughly a")
    print("96% survival rate, while a 3rd class man had roughly a 14% survival rate.")


if __name__ == "__main__":
    main()
