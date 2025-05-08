import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import optuna
import joblib
import matplotlib.pyplot as plt

# === 1. Rutas y carga de datos ===
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data")
clean_csv_path = os.path.join(data_dir, "clean_data.csv")
print(f"Current directory: {current_dir}")
print(f"Data directory: {data_dir}")
print(f"Clean CSV path: {clean_csv_path}")

df = pd.read_csv(clean_csv_path)

# === 2. Preprocesamiento ===
df['ProductRelated_Duration'] = np.log1p(df['ProductRelated_Duration'])  # log(1+x)
df['Revenue'] = df['Revenue'].astype(int)

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 3. Optuna: búsqueda de hiperparámetros ===
def objective(trial):
    params = {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": "balanced",
        "random_state": 42
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_tr, y_tr)

        model = LGBMClassifier(**params)
        model.fit(X_res, y_res)

        y_val_proba = model.predict_proba(X_val)[:, 1]

        best_f1 = 0
        for t in np.linspace(0.3, 0.7, 15):
            y_val_pred = (y_val_proba > t).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            best_f1 = max(best_f1, f1)

        f1_scores.append(best_f1)

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# === 4. Guardar el estudio ===
joblib.dump(study, "optuna_study.pkl")
print("📁 Estudio guardado en 'optuna_study.pkl'")

best_params = study.best_params
best_params.update({"n_estimators": 500, "class_weight": "balanced", "random_state": 42})
print("\n🎯 Mejores hiperparámetros encontrados:")
print(best_params)

# === 5. Entrenamiento final ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

final_model = LGBMClassifier(**best_params)
final_model.fit(X_train_res, y_train_res)

# === 6. Guardar modelo entrenado ===
joblib.dump(final_model, "lightgbm_model.pkl")
print("📁 Modelo LightGBM guardado en 'lightgbm_model.pkl'")

# === 7. Buscar mejor threshold ===
y_val_proba = final_model.predict_proba(X_train)[:, 1]
best_f1 = 0
best_threshold = 0.5
for t in np.linspace(0.3, 0.7, 50):
    y_val_pred = (y_val_proba > t).astype(int)
    f1 = f1_score(y_train, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n🔍 Mejor threshold encontrado: {best_threshold:.3f}")

# === 8. Evaluación en test ===
y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba > best_threshold).astype(int)

print("\n📊 Evaluación en test:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_test_proba))

# === 9. Evaluación en entrenamiento para medir overfitting ===
y_train_proba = final_model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_proba > best_threshold).astype(int)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n📈 F1-score entrenamiento: {train_f1:.4f}")
print(f"📉 F1-score test:          {test_f1:.4f}")
print(f"⚠️ Overfitting (train - test): {train_f1 - test_f1:.4f}")

# === 10. Curva ROC ===
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_test_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Modelo Final")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
