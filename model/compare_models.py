"""Compara los modelos con validación cruzada y métricas de clasificación.
Ejecutar desde la carpeta deploy-diabetes:  python model/compare_models.py
"""
import pathlib
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv(pathlib.Path('data/diabetes_prediction_dataset.csv'))
df = df.drop_duplicates()
df = df[df['gender'] != 'Other']
y = df.pop('diabetes')
X = df

prep = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'smoking_history'])],
    remainder='passthrough')

modelos = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                            n_jobs=-1, random_state=0),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             n_jobs=4, random_state=0),
}
scoring = {'roc_auc': 'roc_auc', 'pr_auc': 'average_precision',
           'recall': 'recall', 'precision': 'precision', 'f1': 'f1',
           'mae': 'neg_mean_absolute_error'}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

filas = []
for nombre, modelo in modelos.items():
    pipe = Pipeline([('prep', prep), ('model', modelo)])
    r = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
    filas.append({'Modelo': nombre,
                  'ROC-AUC': r['test_roc_auc'].mean(),
                  'PR-AUC': r['test_pr_auc'].mean(),
                  'Recall': r['test_recall'].mean(),
                  'Precision': r['test_precision'].mean(),
                  'F1': r['test_f1'].mean(),
                  'MAE': -r['test_mae'].mean()})
print(pd.DataFrame(filas).round(4).to_string(index=False))
