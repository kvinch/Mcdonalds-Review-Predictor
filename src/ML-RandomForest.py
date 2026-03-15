import os
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance

#Lista de las direcciones de los archivos
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "McDonalds_Features.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "rf_results")
MODEL_DIR   = os.path.join(BASE_DIR, "model")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

PALETTE = {
    "good":    "#2ecc71",
    "neutral": "#f39c12",
    "bad":     "#e74c3c",
}
CUSTOM_CMAP = sns.diverging_palette(145, 10, as_cmap=True)

LABEL_ORDER = ["bad", "neutral", "good"]
COLORS_ORDER = [PALETTE[l] for l in LABEL_ORDER]

plt.rcParams.update({
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#3d3d5c",
    "axes.labelcolor":   "#e0e0ff",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#a0a0cc",
    "ytick.color":       "#a0a0cc",
    "text.color":        "#e0e0ff",
    "grid.color":        "#2a2a4a",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  "#3d3d5c",
})

#Cargamos los Datos
def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  CARGA DE DATOS")
    print(f"{'='*60}")
    df = pd.read_csv(path)
    print(f"  ✓ Shape         : {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  ✓ Columnas      : {df.columns.tolist()}")
    print(f"  ✓ Nulos totales : {df.isnull().sum().sum()}")
    return df

#Preproceso de los datos
def preprocess(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  PREPROCESAMIENTO")
    print(f"{'='*60}\n")

    # Excluimos columnas con data leakage
    #-'rating': define directamente rating_label
    #-'rating_normalized': transformación lineal de rating
    #-'rating_vs_sentiment_gap': rating_normalized - sentiment_compound
    numeric_features = [
        "rating_count", "review_time_since_days", "review_length",
        "location_cluster", "sentiment_compound", "sentiment_polarity",
        "sentiment_subjectivity", "review_word_count",
    ]

    # Encodear city y postal_code (top N + "other")
    for col in ["city", "postal_code"]:
        if col in df.columns:
            top = df[col].value_counts().nlargest(30).index
            df[col + "_enc"] = df[col].apply(lambda x: x if x in top else "other")
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col + "_enc"].astype(str))
            numeric_features.append(col + "_enc")

    # sentiment_label_vader (categorical -> binary encoding)
    if "sentiment_label_vader" in df.columns:
        sent_dummies = pd.get_dummies(df["sentiment_label_vader"], prefix="senti")
        df = pd.concat([df, sent_dummies], axis=1)
        numeric_features += list(sent_dummies.columns)

    # Target -> rating_label
    le_target   = LabelEncoder()
    y           = le_target.fit_transform(df["rating_label"])
    class_names = le_target.classes_

    X = df[numeric_features].copy().fillna(0)

    print(f"  ✓ Features seleccionadas ({len(numeric_features)}): {numeric_features}")
    print(f"  ✓ Clases (target)       : {class_names.tolist()}")
    print(f"  ✓ Shape X               : {X.shape}")
    print(f"  ✓ Distribución y        : { {c: int((y == i).sum()) for i, c in enumerate(class_names)} }")

    return X, y, class_names, numeric_features


#Entrenamos el modelo de Random Forest
def train_random_forest(X, y, class_names):
    print(f"\n{'='*60}")
    print("  ENTRENAMIENTO-RANDOM FOREST")
    print(f"{'='*60}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train : {X_train.shape[0]:,} muestras")
    print(f"Test  : {X_test.shape[0]:,} muestras")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("Modelo entrenado con n_estimators=300, class_weight='balanced'")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\nCross-Val Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return rf, X_train, X_test, y_train, y_test

#Evaluación del modelo RF
def evaluate_model(rf, X_train, X_test, y_train, y_test, class_names, feature_names):
    print(f"\n{'='*60}")
    print("EVALUACIÓN DEL MODELO")
    print(f"{'='*60}\n")

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc_train = rf.score(X_train, y_train)
    acc_test  = rf.score(X_test,  y_test)
    print(f"Accuracy TRAIN : {acc_train:.4f}")
    print(f"Accuracy TEST  : {acc_test:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── 5.1  Confusion Matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Matriz de Confusión — Random Forest", fontsize=14, fontweight="bold", pad=12)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="YlGnBu")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    _save(fig, "rf_01_confusion_matrix.png")
    print("  [✓] RF 01 – Matriz de confusión guardada.")

    # ── 5.2  ROC Curves (one-vs-rest) ────────────────────────────────────────
    from sklearn.preprocessing import label_binarize
    Y_bin = label_binarize(y_test, classes=range(len(class_names)))
    roc_colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Curvas ROC por Clase (One-vs-Rest)", fontsize=14, fontweight="bold")
    for i, (cls, col) in enumerate(zip(class_names, roc_colors)):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.tight_layout()
    _save(fig, "rf_02_roc_curves.png")
    print("  [✓] RF 02 – Curvas ROC guardadas.")

    # ── 5.3  Feature Importance (Gini) ───────────────────────────────────────
    importances_gini = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_n = 15
    top_feat = importances_gini.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Top {top_n} Features — Importancia Gini (RF)", fontsize=14, fontweight="bold")
    bars = ax.barh(top_feat.index[::-1], top_feat.values[::-1],
                   color=sns.color_palette("mako", top_n)[::-1],
                   edgecolor="#ffffff11")
    ax.set_xlabel("Importancia media (Gini)")
    ax.xaxis.grid(True)
    for bar, val in zip(bars, top_feat.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "rf_03_feature_importance_gini.png")
    print("  [✓] RF 03 – Feature importance (Gini) guardada.")
    print(f"\n  Top 5 features (Gini):\n{importances_gini.head(5).to_string()}")

    # ── 5.4  Permutation Importance ──────────────────────────────────────────
    print("\n  Calculando Permutation Importance")
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    perm = permutation_importance(rf, X_test_df, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False)

    top_perm = perm_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Top {top_n} Features — Permutation Importance", fontsize=14, fontweight="bold")
    ax.barh(top_perm["feature"][::-1], top_perm["importance"][::-1],
            xerr=top_perm["std"][::-1],
            color=sns.color_palette("flare", top_n)[::-1],
            edgecolor="#ffffff11", ecolor="#ffffff66", capsize=3)
    ax.set_xlabel("Caída de accuracy al permutar")
    ax.xaxis.grid(True)
    plt.tight_layout()
    _save(fig, "rf_04_permutation_importance.png")
    print("  [✓] RF 04 – Permutation importance guardada.")

    # ── 5.5  Class Probability Distributions ─────────────────────────────────
    fig, axes = plt.subplots(1, len(class_names), figsize=(16, 5), sharey=False)
    fig.suptitle("Distribución de Probabilidades Predichas por Clase", fontsize=14, fontweight="bold")

    for i, (cls, col, ax) in enumerate(zip(class_names, roc_colors, axes)):
        for j, (cls2, col2) in enumerate(zip(class_names, roc_colors)):
            mask = (y_test == j)
            ax.hist(y_proba[mask, i], bins=30, alpha=0.7, label=f"Real: {cls2}", color=col2,
                    edgecolor="#ffffff11")
        ax.set_title(f"P(clase = {cls})")
        ax.set_xlabel("Probabilidad")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True)

    plt.tight_layout()
    _save(fig, "rf_05_proba_distributions.png")
    print("  [✓] RF 05 – Distribuciones de probabilidad guardadas.")

    # ── 5.6  Cross-Val Scores visual ─────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, pd.DataFrame(np.vstack([X_train, X_test]), columns=feature_names),
                             np.concatenate([y_train, y_test]),
                             cv=cv, scoring="accuracy", n_jobs=-1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Accuracy por Fold — Cross Validation (5-fold)", fontsize=14, fontweight="bold")
    fold_colors = sns.color_palette("cool", 5)
    bars = ax.bar([f"Fold {i+1}" for i in range(5)], scores, color=fold_colors, edgecolor="#ffffff22")
    ax.axhline(scores.mean(), color="white", linestyle="--", lw=1.5, label=f"Media: {scores.mean():.4f}")
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(scores.min() - 0.02, 1.0)
    ax.legend()
    ax.yaxis.grid(True)
    plt.tight_layout()
    _save(fig, "rf_06_crossval_scores.png")
    print(f"  [✓] RF 06 – Cross-val scores guardadas. Media={scores.mean():.4f} ± {scores.std():.4f}")

    return importances_gini, perm_df





# Guardamos el modelo
def save_model(rf, class_names, feature_names: list) -> None:
    #Guardamos también el .pkl para poder utilizarlo con datos nuevos
    payload = {
        "model":         rf,
        "class_names":   class_names,
        "feature_names": feature_names,
    }
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    joblib.dump(payload, model_path, compress=3)

    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    print(f"\n{'='*60}")
    print(f"  MODELO GUARDADO")
    print(f"{'='*60}")
    print(f"Ruta   : {model_path}")
    print(f"Tamaño : {size_mb:.2f} MB")
    print("Clases : {class_names.tolist()}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"{'='*60}\n")


def _save(fig, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__ == "__main__":
    # 1. Cargar datos
    df = load_data(DATA_PATH)

    # 2. Preprocesamiento
    # Excluimos columnas con data leakage: rating, rating_normalized, rating_vs_sentiment_gap
    LEAKAGE_COLS = ["rating", "rating_normalized", "rating_vs_sentiment_gap"]
    df_model = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    X, y, class_names, feature_names = preprocess(df_model)

    # 3. Entrenamos
    rf, X_train, X_test, y_train, y_test = train_random_forest(X, y, class_names)

    # 4. Resultados y gráficos del modelo
    evaluate_model(
        rf, X_train, X_test, y_train, y_test, class_names, feature_names
    )

    # 5. Guardamos el modelo
    save_model(rf, class_names, feature_names)

    print(f"Gráficos:{OUTPUT_DIR}")
    print(f"Modelo:{os.path.join(MODEL_DIR, 'rf_model.pkl')}\n")
