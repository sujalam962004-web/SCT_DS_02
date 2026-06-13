"""
Titanic Dataset — Data Cleaning & Exploratory Data Analysis
============================================================
Source: Kaggle Titanic Competition (train.csv, test.csv)
Author: Data Analysis Pipeline
Steps:  11 documented cleaning steps + full EDA with findings
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════
print("=" * 62)
print("TITANIC — DATA CLEANING & EDA")
print("=" * 62)

print("\n[STEP 1] Loading datasets...")
train = pd.read_csv('/mnt/user-data/uploads/train__1_.csv')
test  = pd.read_csv('/mnt/user-data/uploads/test__1_.csv')

print(f"  Train: {train.shape[0]} rows × {train.shape[1]} columns")
print(f"  Test : {test.shape[0]} rows × {test.shape[1]} columns")
print(f"  Columns: {train.columns.tolist()}")

# ═══════════════════════════════════════════════════
# STEP 2: INITIAL DATA INSPECTION
# ═══════════════════════════════════════════════════
print("\n[STEP 2] Initial inspection...")
print("\n  Data Types:")
print(train.dtypes.to_string())
print("\n  First 5 rows:")
print(train.head().to_string())
print("\n  Statistical Summary:")
print(train.describe().round(2).to_string())

# ═══════════════════════════════════════════════════
# STEP 3: NULL VALUE AUDIT
# ═══════════════════════════════════════════════════
print("\n[STEP 3] Null value audit...")
null_counts = train.isnull().sum()
null_pct    = (train.isnull().mean() * 100).round(1)
null_df = pd.DataFrame({'Missing Count': null_counts, 'Missing %': null_pct})
null_df = null_df[null_df['Missing Count'] > 0]
print(null_df.to_string())
# Key findings:
# - Cabin: 687 missing (77.1%) → too sparse, create binary flag
# - Age: 177 missing (19.9%) → impute with group-wise median
# - Embarked: 2 missing (0.2%) → fill with mode

# ═══════════════════════════════════════════════════
# STEP 4: REMOVE DUPLICATES
# ═══════════════════════════════════════════════════
print("\n[STEP 4] Checking for duplicate rows...")
dupes = train.duplicated(subset='PassengerId').sum()
print(f"  Duplicate PassengerIds: {dupes}")
train = train.drop_duplicates(subset='PassengerId')
print(f"  Shape after dedup: {train.shape}")

# ═══════════════════════════════════════════════════
# STEP 5: IMPUTE AGE (group-wise median)
# ═══════════════════════════════════════════════════
print("\n[STEP 5] Imputing Age with group-wise median (Pclass × Sex)...")
age_before = train['Age'].isna().sum()
group_medians = train.groupby(['Pclass','Sex'])['Age'].median()
print("  Group medians:\n", group_medians.to_string())

def fill_age(row):
    if pd.isna(row['Age']):
        return group_medians.get((row['Pclass'], row['Sex']), train['Age'].median())
    return row['Age']

train['Age'] = train.apply(fill_age, axis=1)
age_after = train['Age'].isna().sum()
print(f"  Age nulls: {age_before} → {age_after}")

# ═══════════════════════════════════════════════════
# STEP 6: IMPUTE EMBARKED
# ═══════════════════════════════════════════════════
print("\n[STEP 6] Imputing Embarked with mode...")
mode_emb = train['Embarked'].mode()[0]
print(f"  Mode = '{mode_emb}' (Southampton)")
train['Embarked'] = train['Embarked'].fillna(mode_emb)
print(f"  Embarked nulls remaining: {train['Embarked'].isna().sum()}")

# ═══════════════════════════════════════════════════
# STEP 7: CABIN → HasCabin BINARY FLAG
# ═══════════════════════════════════════════════════
print("\n[STEP 7] Creating HasCabin binary feature...")
train['HasCabin'] = train['Cabin'].notna().astype(int)
cabin_surv = train.groupby('HasCabin')['Survived'].mean().mul(100).round(1)
print(f"  Survival rate by HasCabin:\n  {cabin_surv.to_dict()}")
# HasCabin=1 → 66.7% survival vs HasCabin=0 → 30.0%

# ═══════════════════════════════════════════════════
# STEP 8: EXTRACT TITLE FROM NAME
# ═══════════════════════════════════════════════════
print("\n[STEP 8] Extracting Title from Name...")
train['Title'] = train['Name'].str.extract(r',\s*([^\.]+)\.')
title_map = {
    'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master',
    'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare',
    'Mlle':'Miss','Mme':'Mrs','Don':'Rare','Lady':'Rare',
    'Countess':'Rare','Capt':'Rare','Jonkheer':'Rare','Sir':'Rare'
}
train['Title'] = train['Title'].map(title_map).fillna('Rare')
title_surv = train.groupby('Title')['Survived'].mean().mul(100).round(1)
print("  Survival by Title:")
print(title_surv.to_string())

# ═══════════════════════════════════════════════════
# STEP 9: ENGINEER FAMILY SIZE
# ═══════════════════════════════════════════════════
print("\n[STEP 9] Engineering FamilySize and IsAlone...")
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone']    = (train['FamilySize'] == 1).astype(int)
family_surv = train.groupby('FamilySize')['Survived'].mean().mul(100).round(1)
print("  Survival by FamilySize:")
print(family_surv.to_string())

# ═══════════════════════════════════════════════════
# STEP 10: BIN CONTINUOUS VARIABLES
# ═══════════════════════════════════════════════════
print("\n[STEP 10] Binning Age and Fare...")
train['AgeBand'] = pd.cut(train['Age'],
    bins=[0,12,18,30,50,80],
    labels=['Child','Teen','Young Adult','Adult','Senior'])

train['FareBand'] = pd.qcut(train['Fare'], q=4,
    labels=['Low','Mid-Low','Mid-High','High'],
    duplicates='drop')

print("  AgeBand distribution:")
print(train['AgeBand'].value_counts().sort_index().to_string())
print("  FareBand survival:")
print(train.groupby('FareBand', observed=True)['Survived'].mean().mul(100).round(1).to_string())

# ═══════════════════════════════════════════════════
# STEP 11: DROP NON-PREDICTIVE COLUMNS
# ═══════════════════════════════════════════════════
print("\n[STEP 11] Dropping non-predictive columns...")
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_clean = train.drop(columns=drop_cols)
print(f"  Dropped: {drop_cols}")
print(f"  Final shape: {train_clean.shape}")
print(f"  Final columns: {train_clean.columns.tolist()}")
print(f"  Remaining nulls: {train_clean.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════
# EDA: KEY FINDINGS
# ═══════════════════════════════════════════════════
print("\n" + "=" * 62)
print("EDA — KEY FINDINGS")
print("=" * 62)

print(f"\n  Overall survival rate: {train['Survived'].mean()*100:.1f}%")
print(f"  Survived: {train['Survived'].sum()} | Perished: {(train['Survived']==0).sum()}")

print("\n  [1] Gender Effect (strongest predictor):")
print(train.groupby('Sex')['Survived'].mean().mul(100).round(1).to_string())

print("\n  [2] Class Effect:")
print(train.groupby('Pclass')['Survived'].mean().mul(100).round(1).to_string())

print("\n  [3] Age Group Effect:")
train['AgeGroup'] = pd.cut(train['Age'],bins=[0,12,18,30,40,50,60,80],
    labels=['Child(0-12)','Teen(13-18)','YoungAdult(19-30)','Adult(31-40)','Middle(41-50)','Senior(51-60)','Elder(61+)'])
print(train.groupby('AgeGroup',observed=True)['Survived'].mean().mul(100).round(1).to_string())

print("\n  [4] Gender × Class interaction (survival %):")
pivot = train.pivot_table(values='Survived', index='Pclass', columns='Sex', aggfunc='mean').mul(100).round(1)
print(pivot.to_string())

print("\n  [5] Fare bracket effect:")
fare_bins = pd.cut(train['Fare'],bins=[0,10,20,40,80,200,600],labels=['<$10','$10-20','$20-40','$40-80','$80-200','>$200'])
print(train.groupby(fare_bins,observed=True)['Survived'].mean().mul(100).round(1).to_string())

print("\n  [6] Embarkation port effect:")
print(train.groupby('Embarked')['Survived'].mean().mul(100).round(1).to_string())

print("\n  [7] Family size (optimal = 2-4):")
print(train.groupby('FamilySize')['Survived'].mean().mul(100).round(1).to_string())

print("\n  [8] Correlation to Survived (numeric features):")
num_cols = ['Pclass','Age','SibSp','Parch','Fare','FamilySize','HasCabin']
print(train[num_cols+['Survived']].corr()['Survived'].drop('Survived').round(3).sort_values().to_string())

print("\n  [9] Age: survived vs perished:")
print(f"  Survived — mean: {train[train['Survived']==1]['Age'].mean():.1f}, median: {train[train['Survived']==1]['Age'].median()}")
print(f"  Perished — mean: {train[train['Survived']==0]['Age'].mean():.1f}, median: {train[train['Survived']==0]['Age'].median()}")

print("\n  [10] Cabin presence effect:")
print(train.groupby('HasCabin')['Survived'].mean().mul(100).round(1).to_string())

# ═══════════════════════════════════════════════════
# SAVE CLEAN DATA
# ═══════════════════════════════════════════════════
train_clean.to_csv('/home/claude/titanic_clean.csv', index=False)
print("\n" + "=" * 62)
print("OUTPUTS SAVED")
print("  titanic_clean.csv — cleaned training dataset")
print(f"  Shape: {train_clean.shape[0]} rows × {train_clean.columns.size} features")
print("=" * 62)
