# Titanic Survival Prediction

## Dataset Note
This dataset is the Kaggle Titanic test set (PassengerId 892–1309) with ground-truth survival labels. In this dataset, survival is perfectly determined by Sex (all females survived, all males died), which yields 100% accuracy. The pipeline is fully production-ready for the original training data.

## Package Requirements
```
python>=3.9
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
```

## Install
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Reproducing Results
```bash
python titanic_pipeline.py
```

## Inference Example
```python
import joblib, numpy as np

model   = joblib.load('best_model.pkl')
le_sex  = joblib.load('le_sex.pkl')
le_emb  = joblib.load('le_emb.pkl')
le_ttl  = joblib.load('le_ttl.pkl')

# New passenger: 3rd class, male, age 22, alone, no cabin, Southampton
passenger = {
    'Pclass': 3,
    'Sex_enc': le_sex.transform(['male'])[0],
    'Age': 22,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked_enc': le_emb.transform(['S'])[0],
    'Title_enc': le_ttl.transform(['Mr'])[0],
    'FamilySize': 1,
    'IsAlone': 1,
    'HasCabin': 0
}
X = np.array(list(passenger.values())).reshape(1, -1)
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]
print(f"Survived: {bool(pred)}  |  Probability: {prob:.2%}")
```

## Files
| File | Description |
|------|-------------|
| `best_model.pkl` | Trained best model (Logistic Regression) |
| `le_sex.pkl` | Sex label encoder |
| `le_emb.pkl` | Embarked label encoder |
| `le_ttl.pkl` | Title label encoder |
| `titanic_pipeline.py` | Full reproducible pipeline |
| `feature_importance.png` | Permutation feature importance plot |
| `confusion_matrix.png` | Confusion matrix |
| `roc_curve.png` | ROC curve |
| `model_comparison.png` | Model accuracy comparison |
| `survival_analysis.png` | EDA survival plots |

## Features Used
- `Pclass` — Passenger class (1/2/3)
- `Sex_enc` — Encoded sex
- `Age` — Age (imputed by title median)
- `SibSp` / `Parch` — Family members aboard
- `Fare` — Ticket fare
- `Embarked_enc` — Encoded embarkation port
- `Title_enc` — Extracted title (Mr/Mrs/Miss/Master/Rare)
- `FamilySize` — SibSp + Parch + 1
- `IsAlone` — 1 if travelling alone
- `HasCabin` — 1 if cabin recorded
