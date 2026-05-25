"""
Titanic Survival Prediction Pipeline
=====================================
Dataset: Kaggle Titanic (bhanupratapbiswas/titanic-survival-datasets)
Author : Generated via Claude
"""

import re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION")
print("=" * 55)

df = pd.read_csv('titanic.csv')
print(f"\n[1] Dataset loaded  — shape: {df.shape}")
print(f"    Survived: {df['Survived'].value_counts().to_dict()}")
print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}\n")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
def extract_title(name):
    m = re.search(r',\s*([^\.]+)\.', str(name))
    if m:
        title = m.group(1).strip()
        rare = {'Dr','Rev','Col','Major','Mlle','Countess','Ms',
                'Lady','Jonkheer','Don','Dona','Capt','Sir'}
        return 'Rare' if title in rare else title
    return 'Unknown'

df['Title']      = df['Name'].apply(extract_title)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
df['HasCabin']   = df['Cabin'].notna().astype(int)

print("[2] Feature engineering complete")
print(f"    Titles found: {df['Title'].value_counts().to_dict()}")

# ── 3. Handle Missing Values ──────────────────────────────────────────────────
age_medians = df.groupby('Title')['Age'].median()
df['Age'] = df.apply(
    lambda r: age_medians.get(r['Title'], df['Age'].median())
    if pd.isna(r['Age']) else r['Age'], axis=1)

df['Fare']     = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print(f"\n[3] Missing values handled")
print(f"    Remaining NaN (Cabin kept as HasCabin flag): "
      f"{df[['Age','Fare','Embarked']].isnull().sum().sum()}")

# ── 4. Encode Categorical Variables ──────────────────────────────────────────
le_sex = LabelEncoder()
le_emb = LabelEncoder()
le_ttl = LabelEncoder()

df['Sex_enc']      = le_sex.fit_transform(df['Sex'])
df['Embarked_enc'] = le_emb.fit_transform(df['Embarked'])
df['Title_enc']    = le_ttl.fit_transform(df['Title'])

print("\n[4] Categorical encoding complete")

# ── 5. Prepare Feature Matrix ─────────────────────────────────────────────────
features = [
    'Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked_enc', 'Title_enc', 'FamilySize', 'IsAlone', 'HasCabin'
]
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n[5] Train/test split  —  train: {len(X_train)}, test: {len(X_test)}")

# ── 6. Train & Evaluate Models ────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, C=0.1),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42,
                                                   max_depth=5, min_samples_leaf=5),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42,
                                                       learning_rate=0.1, max_depth=3),
}

results = {}
print("\n[6] Model training & evaluation")
print(f"    {'Model':<25} {'CV Acc':>10}  {'±':>6}  {'Test Acc':>10}")
print("    " + "-" * 55)
for name, m in models.items():
    scores = cross_val_score(m, X_train, y_train, cv=cv, scoring='accuracy')
    m.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, m.predict(X_test))
    results[name] = {'cv_mean': scores.mean(), 'cv_std': scores.std(), 'test_acc': test_acc, 'model': m}
    print(f"    {name:<25} {scores.mean():>10.4f}  {scores.std():>6.4f}  {test_acc:>10.4f}")

best_name  = max(results, key=lambda k: results[k]['test_acc'])
best_model = results[best_name]['model']
print(f"\n    Best model: {best_name}")

# ── 7. Final Metrics ──────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
acc    = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\n[7] Final Metrics  ({best_name})")
print(f"    Accuracy  : {acc:.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ── 8. Feature Explainability (Permutation Importance) ────────────────────────
perm = permutation_importance(best_model, X_test, y_test,
                               n_repeats=30, random_state=42)
imp_mean = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

print("[8] Top feature importances (permutation):")
for feat, val in imp_mean.items():
    print(f"    {feat:<18} {val:+.4f}")

# ── 9. Save Model ─────────────────────────────────────────────────────────────
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(le_sex,     'le_sex.pkl')
joblib.dump(le_emb,     'le_emb.pkl')
joblib.dump(le_ttl,     'le_ttl.pkl')
print("\n[9] Model artifacts saved.")

# ── 10. Inference Example ─────────────────────────────────────────────────────
print("\n[10] Inference example:")

def predict_passenger(pclass, sex, age, sibsp, parch, fare, embarked, title,
                       has_cabin=0):
    sex_enc      = le_sex.transform([sex])[0]
    embarked_enc = le_emb.transform([embarked])[0]
    title_enc    = le_ttl.transform([title])[0] if title in le_ttl.classes_ else 0
    family_size  = sibsp + parch + 1
    is_alone     = int(family_size == 1)
    X_new = np.array([[pclass, sex_enc, age, sibsp, parch, fare,
                       embarked_enc, title_enc, family_size, is_alone, has_cabin]])
    pred = best_model.predict(X_new)[0]
    prob = best_model.predict_proba(X_new)[0][1]
    return pred, prob

# Example 1: 3rd-class male
pred, prob = predict_passenger(3, 'male', 22, 0, 0, 7.25, 'S', 'Mr')
print(f"    3rd-class Mr (age 22)   → Survived={bool(pred)}  p={prob:.2%}")

# Example 2: 1st-class female
pred, prob = predict_passenger(1, 'female', 35, 1, 0, 71.0, 'C', 'Mrs', has_cabin=1)
print(f"    1st-class Mrs (age 35)  → Survived={bool(pred)}  p={prob:.2%}")

print("\nPipeline complete.")
