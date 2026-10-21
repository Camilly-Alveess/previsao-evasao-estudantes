"""Retreina o pipeline de evasão com a versão atual do scikit-learn.

Usa exatamente os mesmos dados de treino/teste, o mesmo pré-processamento e os mesmos
hiperparâmetros do notebook (Random Forest, random_state=42), apenas salvando o modelo
numa versão do scikit-learn compatível com o Streamlit Cloud.

Uso (na pasta do projeto):  python retreinar_modelo.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from outlier_clipper import OutlierClipper

treino = pd.read_csv("data/dados_treino.csv")
teste = pd.read_csv("data/dados_teste.csv")

X_train, y_train = treino.drop(columns="evasao"), treino["evasao"]
X_test, y_test = teste.drop(columns="evasao"), teste["evasao"]

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clipper", OutlierClipper(factor=1.5)),
    ("scaler", StandardScaler()),
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols),
])
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100,
                                          class_weight="balanced", n_jobs=-1)),
])

# Validação cruzada (com score de treino, para o diagnóstico de overfitting)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
r = cross_validate(model_pipeline, X_train, y_train, cv=cv, n_jobs=-1,
                   scoring=["accuracy", "f1", "roc_auc"], return_train_score=True)
for m in ("accuracy", "f1", "roc_auc"):
    print(f"CV {m}: treino={r[f'train_{m}'].mean():.4f} | validação={r[f'test_{m}'].mean():.4f}")

model_pipeline.fit(X_train, y_train)
pred = model_pipeline.predict(X_test)
proba = model_pipeline.predict_proba(X_test)[:, 1]
print(f"\nTeste: acurácia={accuracy_score(y_test, pred):.4f} | ROC-AUC={roc_auc_score(y_test, proba):.4f}")
print(classification_report(y_test, pred, target_names=["Graduado", "Desistente"]))

joblib.dump({"model_pipeline": model_pipeline, "input_features": list(X_train.columns)},
            "model/modelo_evasao_pipeline.joblib")
print("Modelo salvo em model/modelo_evasao_pipeline.joblib")
