"""
╔══════════════════════════════════════════════════════════════╗
║         TITANIC SURVIVAL PREDICTION — MODEL PIPELINE        ║
║         Data Cleaning → Feature Engineering → Training      ║
║         Cross-Validation → Ensemble → Submission CSV        ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN
──────────
1. Install dependencies (once):
       pip install pandas numpy scikit-learn xgboost matplotlib seaborn

2. Place these files in the SAME folder as this script:
       train.csv   (or train__1_.csv — update DATA_DIR below)
       test.csv    (or test__1_.csv)
       gender_submission.csv  (optional reference)

3. Run:
       python titanic_model.py

4. Output files produced:
       submission.csv            ← Kaggle-ready predictions
       feature_importance.png    ← Feature importance bar chart
       model_comparison.png      ← CV score comparison chart
       confusion_matrix.png      ← Best model confusion matrix
"""

# ───────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these paths if needed
# ───────────────────────────────────────────────────────────────
TRAIN_PATH  = "train__1_.csv"      # path to training CSV
TEST_PATH   = "test__1_.csv"       # path to test CSV
OUTPUT_PATH = "submission.csv"     # where predictions are saved
RANDOM_SEED = 42
CV_FOLDS    = 5
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (cross_val_score,
                                     StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix,
                              roc_auc_score)
import xgboost as xgb

# ══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  TITANIC SURVIVAL PREDICTION PIPELINE")
print("=" * 60)

print("\n[1/6] Loading data...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

print(f"  Train : {train_raw.shape[0]:,} rows × {train_raw.shape[1]} columns")
print(f"  Test  : {test_raw.shape[0]:,} rows  × {test_raw.shape[1]} columns")
print(f"  Survival rate: {train_raw['Survived'].mean()*100:.1f}%  "
      f"({train_raw['Survived'].sum()} survived / "
      f"{(train_raw['Survived']==0).sum()} perished)")

# ══════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[2/6] Preprocessing & feature engineering...")

def preprocess(df):
    """
    Full cleaning + feature engineering pipeline.
    Applied identically to both train and test sets.
    """
    df = df.copy()

    # ── 2a. Extract Title from Name ──────────────────────────
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    title_map = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
        "Capt": "Rare", "Countess": "Rare", "Lady": "Rare",
        "Sir": "Rare", "Don": "Rare", "Jonkheer": "Rare"
    }
    df["Title"] = df["Title"].map(title_map).fillna("Rare")

    # ── 2b. Fill missing Age (group-wise median) ─────────────
    group_median = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(group_median).fillna(df["Age"].median())

    # ── 2c. Fill missing Fare & Embarked ─────────────────────
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # ── 2d. Cabin binary flag (presence = high class proxy) ──
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # ── 2e. Family-size features ─────────────────────────────
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # ── 2f. Interaction & log features ───────────────────────
    df["Age*Class"] = df["Age"] * df["Pclass"]   # older + lower class = higher risk
    df["Fare_log"]  = np.log1p(df["Fare"])        # right-skewed → log-normal

    # ── 2g. Encode categoricals ──────────────────────────────
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])         # female=0, male=1
    df["Embarked"] = le.fit_transform(df["Embarked"])    # C=0, Q=1, S=2
    df["Title"]    = le.fit_transform(df["Title"])

    # ── 2h. Select final feature set ─────────────────────────
    features = [
        "Pclass",      # ticket class (1=1st, 2=2nd, 3=3rd)
        "Sex",         # gender (encoded)
        "Age",         # age in years (imputed)
        "SibSp",       # siblings/spouses aboard
        "Parch",       # parents/children aboard
        "Fare_log",    # log of ticket fare
        "Embarked",    # port of embarkation (encoded)
        "HasCabin",    # cabin record present
        "Title",       # social title (encoded)
        "FamilySize",  # total family size
        "IsAlone",     # traveling solo flag
        "Age*Class",   # interaction feature
    ]
    return df[features]


X      = preprocess(train_raw)
y      = train_raw["Survived"]
X_test = preprocess(test_raw)

print(f"  Features used   : {X.shape[1]}")
print(f"  Feature names   : {list(X.columns)}")
print(f"  Missing in X    : {X.isnull().sum().sum()}")
print(f"  Missing in X_test: {X_test.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — DEFINE & CROSS-VALIDATE MODELS
# ══════════════════════════════════════════════════════════════
print("\n[3/6] Cross-validating individual models...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=500, random_state=RANDOM_SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=4,
        random_state=RANDOM_SEED
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0
    ),
    "SVM (RBF)": SVC(
        C=2.0, kernel="rbf", gamma="scale",
        probability=True, random_state=RANDOM_SEED
    ),
}

cv_results = {}
print(f"\n  {'Model':<25}  {'Mean CV Acc':>11}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
print("  " + "─" * 60)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25}  {scores.mean():.4f}       "
          f"±{scores.std():.4f}  {scores.min():.4f}  {scores.max():.4f}")

best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n  ✓ Best individual model: {best_model_name}  "
      f"({cv_results[best_model_name].mean()*100:.2f}%)")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — ENSEMBLE (VOTING CLASSIFIER)
# ══════════════════════════════════════════════════════════════
print("\n[4/6] Building Voting Ensemble (soft voting)...")

# Use the three best tree-based models — drop SVM (lower CV) and
# Logistic Regression (linear, less expressive)
ensemble = VotingClassifier(
    estimators=[
        ("rf",  models["Random Forest"]),
        ("gb",  models["Gradient Boosting"]),
        ("xgb", models["XGBoost"]),
    ],
    voting="soft",       # averages predicted probabilities
    weights=[1, 1, 1],   # equal weights
    n_jobs=-1,
)

ensemble_scores = cross_val_score(ensemble, X, y, cv=cv,
                                   scoring="accuracy", n_jobs=-1)
cv_results["Ensemble (RF+GB+XGB)"] = ensemble_scores
print(f"  Ensemble CV Accuracy: {ensemble_scores.mean():.4f} "
      f"± {ensemble_scores.std():.4f}")

# Decide final model
all_means = {k: v.mean() for k, v in cv_results.items()}
final_model_name = max(all_means, key=all_means.get)
final_model = ensemble if "Ensemble" in final_model_name else models[final_model_name]
print(f"  ✓ Final model selected: {final_model_name}  "
      f"({all_means[final_model_name]*100:.2f}%)")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — TRAIN FINAL MODEL & EVALUATE
# ══════════════════════════════════════════════════════════════
print("\n[5/6] Training final model on full training set...")

final_model.fit(X, y)
train_preds = final_model.predict(X)
train_acc   = accuracy_score(y, train_preds)
print(f"  Training accuracy (in-sample): {train_acc*100:.2f}%")
print(f"  CV accuracy (generalisation) : {all_means[final_model_name]*100:.2f}%")
print(f"  Overfitting gap              : "
      f"{(train_acc - all_means[final_model_name])*100:.2f}pp")

# Classification report on out-of-fold predictions (honest estimate)
oof_preds = cross_val_predict(final_model, X, y, cv=cv)
print("\n  Classification Report (out-of-fold predictions):")
print(classification_report(y, oof_preds,
                             target_names=["Perished", "Survived"],
                             digits=4))

# ROC-AUC (needs probabilities)
try:
    oof_proba = cross_val_predict(final_model, X, y, cv=cv, method="predict_proba")
    auc = roc_auc_score(y, oof_proba[:, 1])
    print(f"  ROC-AUC (OOF): {auc:.4f}")
except Exception:
    pass

# Feature importances from XGBoost (always trained solo for fi)
print("\n  Feature Importances (XGBoost):")
xgb_solo = models["XGBoost"]
xgb_solo.fit(X, y)
fi = pd.Series(xgb_solo.feature_importances_,
               index=X.columns).sort_values(ascending=False)
for feat, imp in fi.items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<15}  {imp:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════
# SECTION 6 — PREDICT ON TEST SET & SAVE SUBMISSION
# ══════════════════════════════════════════════════════════════
print(f"\n[6/6] Generating test predictions → {OUTPUT_PATH}")

test_preds = final_model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_raw["PassengerId"],
    "Survived":    test_preds.astype(int)
})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"  Submission rows   : {len(submission)}")
print(f"  Predicted survived: {test_preds.sum()} "
      f"({test_preds.mean()*100:.1f}%)")
print(f"  Predicted perished: {(test_preds==0).sum()} "
      f"({(1-test_preds.mean())*100:.1f}%)")
print(f"\n  ✓ Saved → {OUTPUT_PATH}")
print("\n  First 10 predictions:")
print(submission.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════
# SECTION 7 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════
print("\n  Generating charts...")
plt.style.use("dark_background")
fig_color = "#0d1117"
accent    = "#58a6ff"
green_c   = "#3ecf8e"
red_c     = "#f06b6b"
orange_c  = "#f5a623"

# ── Chart A: CV Model Comparison ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(fig_color)
ax.set_facecolor("#161b22")

names  = list(cv_results.keys())
means  = [cv_results[n].mean() * 100 for n in names]
stds   = [cv_results[n].std()  * 100 for n in names]
colors = [green_c if "Ensemble" in n else accent for n in names]

bars = ax.barh(names, means, xerr=stds,
               color=colors, edgecolor="none",
               height=0.55, capsize=4,
               error_kw={"elinewidth": 1.5, "ecolor": "white", "alpha": 0.5})

for bar, mean in zip(bars, means):
    ax.text(mean + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f}%", va="center", ha="left",
            color="white", fontsize=10, fontweight="bold")

ax.set_xlim(65, 91)
ax.set_xlabel("Cross-Validation Accuracy (%)", fontsize=11, color="#8b949e")
ax.set_title("Model Comparison — 5-Fold Stratified CV Accuracy",
             fontsize=13, fontweight="bold", color="white", pad=14)
ax.tick_params(colors="white")
ax.spines[:].set_visible(False)
ax.axvline(x=80, color="#30363d", linewidth=1, linestyle="--", alpha=0.6)

patch = mpatches.Patch(color=green_c, label="Ensemble (final model)")
ax.legend(handles=[patch], loc="lower right", fontsize=9,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig_color)
plt.close()
print("  Saved → model_comparison.png")

# ── Chart B: Feature Importance ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(fig_color)
ax.set_facecolor("#161b22")

fi_sorted = fi.sort_values()
bar_colors = [green_c if fi_sorted[f] == fi_sorted.max() else accent
              for f in fi_sorted.index]
ax.barh(fi_sorted.index, fi_sorted.values * 100,
        color=bar_colors, edgecolor="none", height=0.6)
for i, (feat, val) in enumerate(fi_sorted.items()):
    ax.text(val * 100 + 0.3, i, f"{val*100:.1f}%",
            va="center", ha="left", color="white", fontsize=9)

ax.set_xlabel("Importance (%)", fontsize=11, color="#8b949e")
ax.set_title("Feature Importances (XGBoost)",
             fontsize=13, fontweight="bold", color="white", pad=14)
ax.tick_params(colors="white")
ax.spines[:].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor=fig_color)
plt.close()
print("  Saved → feature_importance.png")

# ── Chart C: Confusion Matrix ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(fig_color)
ax.set_facecolor(fig_color)

cm = confusion_matrix(y, oof_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Perished", "Survived"],
            yticklabels=["Perished", "Survived"],
            linewidths=0.5, linecolor="#30363d",
            annot_kws={"size": 16, "weight": "bold", "color": "white"},
            ax=ax)
ax.set_xlabel("Predicted", fontsize=11, color="#8b949e", labelpad=10)
ax.set_ylabel("Actual",    fontsize=11, color="#8b949e", labelpad=10)
ax.set_title("Confusion Matrix (Out-of-Fold Predictions)",
             fontsize=12, fontweight="bold", color="white", pad=12)
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight",
            facecolor=fig_color)
plt.close()
print("  Saved → confusion_matrix.png")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Final model        : {final_model_name}")
print(f"  CV Accuracy        : {all_means[final_model_name]*100:.2f}%  "
      f"(± {cv_results[final_model_name].std()*100:.2f}%)")
print(f"  Test predictions   : {len(submission)} passengers")
print(f"  Top feature        : {fi.idxmax()}  ({fi.max()*100:.1f}% importance)")
print()
print("  Output files:")
print(f"    {OUTPUT_PATH:<30} ← submit to Kaggle")
print("    model_comparison.png        ← CV accuracy chart")
print("    feature_importance.png      ← XGBoost importances")
print("    confusion_matrix.png        ← prediction quality")
print("=" * 60)
