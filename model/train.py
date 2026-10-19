"""Entrena el modelo (preprocesamiento + XGBoost) y lo guarda en model/.
Ejecutar desde la carpeta deploy-diabetes:  python model/train.py
"""
import pathlib
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv(pathlib.Path('data/diabetes_prediction_dataset.csv'))
df = df.drop_duplicates()
df = df[df['gender'] != 'Other']          # solo 18 casos
y = df.pop('diabetes')
X = df

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)

prep = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'smoking_history'])],
    remainder='passthrough')

clf = Pipeline([
    ('prep', prep),
    ('model', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                            n_jobs=4, random_state=0)),
])

print('Training model..')
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred, target_names=['No diabetes', 'Diabetes']))
print('ROC-AUC:', round(roc_auc_score(y_test, proba), 4))

# Reentrenar con todos los datos antes de guardar
clf.fit(X, y)
print('Saving model..')
dump(clf, pathlib.Path('model/diabetes-v1.joblib'))
print('Listo: model/diabetes-v1.joblib')
