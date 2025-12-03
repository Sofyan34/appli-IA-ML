import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils_ml import render_sidebar_nav
import math

def clean_value(x):
    """
    Convertit proprement les valeurs venant de modèles / pandas / numpy.
    - np.intXX → int
    - np.floatXX → float
    - nan → None
    """
    if x is None:
        return None

    # ints numpy -> python int
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)

    # floats numpy -> python float (nan -> None)
    if isinstance(x, (np.floating, np.float32, np.float64)):
        if math.isnan(x):
            return None
        return float(x)

    return x

def clean_dict(d):
    """Applique clean_value à tout un dictionnaire."""
    return {k: clean_value(v) for k, v in d.items()}


# -----------------------------------------------------------------------------
# CONFIG PAGE + NAV
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Modélisation", page_icon=":brain:", layout="wide")
render_sidebar_nav()
st.title("Modélisation - Recherche d'hyperparamètres & comparaison")

# -----------------------------------------------------------------------------
# 0. RÉCUP DATA NETTOYÉE (page Nettoyage)
# -----------------------------------------------------------------------------
needed_keys = ["X_train", "y_train"]
missing = [k for k in needed_keys if k not in st.session_state]

if missing:
    st.error(
        "Les données ne sont pas prêtes pour la modélisation.\n"
        "Va d'abord dans la page 'Nettoyage' pour splitter / nettoyer.\n"
        f"Clés manquantes : {missing}"
    )
    st.stop()

X_train = st.session_state.X_train.copy()
y_train_raw = st.session_state.y_train.copy()

X_test = st.session_state.get("X_test", None)
y_test_raw = st.session_state.get("y_test", None)

st.caption(
    f"Train set : X_train = {X_train.shape}, "
    f"y_train classes uniques = {pd.Series(y_train_raw).nunique()}"
)

# -----------------------------------------------------------------------------
# 1. ENCODAGE DE LA CIBLE EN ENTIER (train + test si dispo)
# -----------------------------------------------------------------------------
def encode_labels(y_series: pd.Series):
    """
    Transforme la target texte/catégorielle/numérique en entiers 0..K-1.
    Retourne y_enc (np.array), mapping code->label
    """
    y_norm = y_series.astype(str).str.strip().str.lower()
    y_cat = y_norm.astype("category")

    codes = y_cat.cat.codes.to_numpy()
    code_to_label = dict(enumerate(y_cat.cat.categories))

    return codes, code_to_label

# encode train
y_train_enc, code_to_label = encode_labels(y_train_raw)

# encode test en réutilisant le mapping du train
def encode_test_with_train_mapping(y_test_series, code_to_label_train):
    """
    On mappe les labels du test -> codes du train.
    Si une classe du test n'existe pas dans le train, on lui met -1.
    """
    label_to_code = {v: k for k, v in code_to_label_train.items()}
    y_test_norm = y_test_series.astype(str).str.strip().str.lower()
    y_test_enc = y_test_norm.map(label_to_code).fillna(-1).astype(int).to_numpy()
    return y_test_enc

if y_test_raw is not None:
    y_test_enc = encode_test_with_train_mapping(y_test_raw, code_to_label)
else:
    y_test_enc = None

st.markdown("#### Mapping classes (train)")
st.dataframe(
    pd.DataFrame(
        [{"code": c, "label": lbl} for c, lbl in code_to_label.items()]
    ),
    width='stretch'
)

# -----------------------------------------------------------------------------
# 2. PIPELINE DE PRÉPROCESSING
# -----------------------------------------------------------------------------
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -----------------------------------------------------------------------------
# 3. OUTILS CV & ÉVALUATION
# -----------------------------------------------------------------------------
def make_stratified_cv(y_enc: np.ndarray, wanted_folds: int):
    """
    Essaie de créer un StratifiedKFold robuste.
    On réduit le nombre de folds si une classe est trop petite.
    On renvoie (cv_obj, used_folds) ou (None, None) si impossible.
    """
    values, counts = np.unique(y_enc, return_counts=True)
    min_class_count = counts.min()

    # si une classe a moins de 2 exemples -> pas de stratification possible
    if min_class_count < 2:
        return None, None

    folds = min(wanted_folds, int(min_class_count))
    if folds < 2:
        return None, None

    cv_obj = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return cv_obj, folds


def eval_model_cv(model, X, y_enc, cv_obj):
    """
    Retourne (scores, mean, std).
    scores = cross_val_score(...) avec accuracy.
    """
    scores = cross_val_score(
        model,
        X,
        y_enc,
        cv=cv_obj,
        scoring="accuracy",
        n_jobs=-1
    )
    return scores, float(scores.mean()), float(scores.std())


def train_on_full_data(model, X, y_enc):
    """
    Fallback : pas de CV possible.
    On entraîne sur tout le train et on renvoie l'accuracy
    sur le train lui-même.
    """
    model.fit(X, y_enc)
    y_pred_train = model.predict(X)
    acc_train = accuracy_score(y_enc, y_pred_train)
    return float(acc_train), model


def compute_overfit_report(best_cfg, model_ctor, model_name: str):
    """
    Affiche train_acc, test_acc, et alerte overfit pour la meilleure config.
    - best_cfg est une ligne du df_* (Series ou dict)
    - model_ctor(**kwargs) doit renvoyer le bon estimateur final
      (RandomForestClassifier / KNeighborsClassifier / SVC)
    """
    st.markdown("#### Vérification de l'overfitting")

    # 1. Reconstruire le meilleur modèle complet (prep + clf)
    clf_params = {}

    if model_name == "RandomForest":
        # n_estimators
        clf_params["n_estimators"] = int(best_cfg["n_estimators"])

        # max_depth peut être None, int ou NaN
        raw_depth = best_cfg.get("max_depth", None)
        depth_is_nan = isinstance(raw_depth, float) and pd.isna(raw_depth)
        if raw_depth is None or depth_is_nan:
            final_depth = None
        else:
            final_depth = int(raw_depth)

        clf_params["max_depth"] = final_depth
        clf_params["random_state"] = 42
        clf_params["n_jobs"] = -1

    elif model_name == "KNN":
        clf_params["n_neighbors"] = int(best_cfg["k"])

    elif model_name == "SVM":
        clf_params["kernel"] = best_cfg["kernel"]
        clf_params["C"] = float(best_cfg["C"])
        clf_params["gamma"] = "scale"
        clf_params["probability"] = False

    final_model = Pipeline([
        ("prep", preprocessor),
        ("clf", model_ctor(**clf_params))
    ])

    # 2. Fit sur tout le train
    final_model.fit(X_train, y_train_enc)
    train_pred = final_model.predict(X_train)
    train_acc = accuracy_score(y_train_enc, train_pred)

    # 3. Test si dispo
    if X_test is not None and y_test_enc is not None:
        test_pred = final_model.predict(X_test)
        test_acc = accuracy_score(y_test_enc, test_pred)

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Accuracy train : {train_acc:.3f}")
        with c2:
            st.write(f"Accuracy test : {test_acc:.3f}")

        gap = train_acc - test_acc
        if gap > 0.1:
            st.error(
                f"Risque d'overfitting : écart train/test = {gap:.3f} (>0.10)"
            )
        else:
            st.success("Pas de signe clair d'overfitting (écart raisonnable).")
    else:
        st.info(f"Accuracy train (pas de test disponible) : {train_acc:.3f}")

    st.divider()


# -----------------------------------------------------------------------------
# 4. CHOIX SOUS-ONGLET
# -----------------------------------------------------------------------------
st.subheader("Sélection du modèle / analyse")
choice = st.radio(
    "Choisissez un modèle à étudier",
    ["RandomForest", "KNN", "SVM", "Comparaison globale"],
    horizontal=True
)

# on stockera la meilleure config de chaque modèle ici
if "best_models" not in st.session_state:
    st.session_state.best_models = {}

# -----------------------------------------------------------------------------
# 5. RANDOM FOREST
# -----------------------------------------------------------------------------
if choice == "RandomForest":
    st.markdown("### RandomForest - Recherche d'hyperparamètres")

    col1, col2 = st.columns(2)
    with col1:
        n_estimators_list = st.multiselect(
            "Liste des n_estimators",
            [50, 100, 200, 300, 400],
            default=[100, 300]
        )
    with col2:
        max_depth_list = st.multiselect(
            "Liste des max_depth",
            [None, 5, 10, 20],
            default=[None, 10]
        )

    folds_wanted = st.slider(
        "Nombre de folds pour la cross-validation",
        2, 10, 5, 1
    )

    if st.button("Lancer l'étude RandomForest"):
        cv_obj, used_folds = make_stratified_cv(y_train_enc, folds_wanted)

        rows = []
        for n_est in n_estimators_list:
            for depth in max_depth_list:
                rf_model = Pipeline([
                    ("prep", preprocessor),
                    ("clf", RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])

                if cv_obj is not None:
                    scores, mean_acc, std_acc = eval_model_cv(rf_model, X_train, y_train_enc, cv_obj)
                    rows.append({
                        "model": "RandomForest",
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "cv_folds": used_folds,
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                        "mode": "CV"
                    })
                else:
                    acc_train, _ = train_on_full_data(rf_model, X_train, y_train_enc)
                    rows.append({
                        "model": "RandomForest",
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "cv_folds": 1,
                        "mean_accuracy": acc_train,
                        "std_accuracy": np.nan,
                        "mode": "TRAIN_ONLY"
                    })

        df_rf = pd.DataFrame(rows)
        st.markdown("#### Résultats RandomForest")
        st.dataframe(df_rf, width='stretch')

        best_row = df_rf.loc[df_rf["mean_accuracy"].idxmax()]
        st.success(
            f"Meilleure config : n_estimators={best_row['n_estimators']}, "
            f"max_depth={best_row['max_depth']} → "
            f"accuracy={best_row['mean_accuracy']:.3f} "
            f"({best_row['mode']})"
        )

        # on sauvegarde pour comparaison globale
        st.session_state.best_models["RandomForest"] = clean_dict(dict(best_row))

        # check overfitting sur CE meilleur modèle
        compute_overfit_report(
            best_cfg=best_row,
            model_ctor=RandomForestClassifier,
            model_name="RandomForest"
        )


# -----------------------------------------------------------------------------
# 6. KNN
# -----------------------------------------------------------------------------
elif choice == "KNN":
    st.markdown("### KNN - Recherche d'hyperparamètres")

    k_list = st.multiselect(
        "Liste des k (n_neighbors)",
        list(range(1, 21)),
        default=[3, 5, 7]
    )

    folds_wanted = st.slider(
        "Nombre de folds pour la cross-validation",
        2, 10, 5, 1,
        key="cv_knn"
    )

    if st.button("Lancer l'étude KNN"):
        cv_obj, used_folds = make_stratified_cv(y_train_enc, folds_wanted)

        rows = []
        for k in k_list:
            knn_model = Pipeline([
                ("prep", preprocessor),
                ("clf", KNeighborsClassifier(n_neighbors=k))
            ])

            if cv_obj is not None:
                scores, mean_acc, std_acc = eval_model_cv(knn_model, X_train, y_train_enc, cv_obj)
                rows.append({
                    "model": "KNN",
                    "k": k,
                    "cv_folds": used_folds,
                    "mean_accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "mode": "CV"
                })
            else:
                acc_train, _ = train_on_full_data(knn_model, X_train, y_train_enc)
                rows.append({
                    "model": "KNN",
                    "k": k,
                    "cv_folds": 1,
                    "mean_accuracy": acc_train,
                    "std_accuracy": np.nan,
                    "mode": "TRAIN_ONLY"
                })

        df_knn = pd.DataFrame(rows)
        st.markdown("#### Résultats KNN")
        st.dataframe(df_knn, width='stretch')

        best_row = df_knn.loc[df_knn["mean_accuracy"].idxmax()]
        st.success(
            f"Meilleure config : k={best_row['k']} → "
            f"accuracy={best_row['mean_accuracy']:.3f} "
            f"({best_row['mode']})"
        )

        st.session_state.best_models["KNN"] = clean_dict(dict(best_row))

        # check overfitting sur CE meilleur modèle
        compute_overfit_report(
            best_cfg=best_row,
            model_ctor=KNeighborsClassifier,
            model_name="KNN"
        )

        # petit graphe barres
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(df_knn["k"], df_knn["mean_accuracy"])
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy moyenne")
        ax.set_title("KNN: accuracy moyenne par k")
        for i, v in enumerate(df_knn["mean_accuracy"]):
            ax.text(df_knn["k"].iloc[i], v + 0.01, f"{v:.2f}", ha="center")
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# 7. SVM
# -----------------------------------------------------------------------------
elif choice == "SVM":
    st.markdown("### SVM - Recherche d'hyperparamètres")

    col1, col2 = st.columns(2)
    with col1:
        C_list = st.multiselect(
            "Liste des C",
            [0.5, 1.0, 2.0, 5.0],
            default=[1.0, 2.0]
        )
    with col2:
        kernel_list = st.multiselect(
            "Liste des kernels",
            ["linear", "rbf", "poly"],
            default=["rbf", "linear"]
        )

    folds_wanted = st.slider(
        "Nombre de folds pour la cross-validation",
        2, 10, 5, 1,
        key="cv_svm"
    )

    if st.button("Lancer l'étude SVM"):
        cv_obj, used_folds = make_stratified_cv(y_train_enc, folds_wanted)

        rows = []
        for C_val in C_list:
            for kernel_val in kernel_list:
                svm_model = Pipeline([
                    ("prep", preprocessor),
                    ("clf", SVC(
                        kernel=kernel_val,
                        C=C_val,
                        gamma="scale",
                        probability=False
                    ))
                ])

                if cv_obj is not None:
                    scores, mean_acc, std_acc = eval_model_cv(svm_model, X_train, y_train_enc, cv_obj)
                    rows.append({
                        "model": "SVM",
                        "C": C_val,
                        "kernel": kernel_val,
                        "cv_folds": used_folds,
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                        "mode": "CV"
                    })
                else:
                    acc_train, _ = train_on_full_data(svm_model, X_train, y_train_enc)
                    rows.append({
                        "model": "SVM",
                        "C": C_val,
                        "kernel": kernel_val,
                        "cv_folds": 1,
                        "mean_accuracy": acc_train,
                        "std_accuracy": np.nan,
                        "mode": "TRAIN_ONLY"
                    })

        df_svm = pd.DataFrame(rows)
        st.markdown("#### Résultats SVM")
        st.dataframe(df_svm, width='stretch')

        best_row = df_svm.loc[df_svm["mean_accuracy"].idxmax()]
        st.success(
            f"Meilleure config : kernel={best_row['kernel']}, C={best_row['C']} → "
            f"accuracy={best_row['mean_accuracy']:.3f} "
            f"({best_row['mode']})"
        )

        st.session_state.best_models["SVM"] = clean_dict(dict(best_row))

        # check overfitting sur CE meilleur modèle
        compute_overfit_report(
            best_cfg=best_row,
            model_ctor=SVC,
            model_name="SVM"
        )

        # heatmap accuracies
        pivot_acc = df_svm.pivot_table(
            index="kernel",
            columns="C",
            values="mean_accuracy",
            aggfunc="max"
        )
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Accuracy moyenne SVM (kernel vs C)")
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# 8. COMPARAISON GLOBALE
# -----------------------------------------------------------------------------
elif choice == "Comparaison globale":
    st.markdown("### Comparaison globale des meilleurs modèles")

    if len(st.session_state.best_models) == 0:
        st.info(
            "Aucun meilleur modèle enregistré pour l'instant.\n"
            "Lance d'abord RandomForest / KNN / SVM pour stocker leur meilleure config."
        )
        st.stop()

    st.write("Meilleures configs enregistrées :")
    st.json(st.session_state.best_models)

    folds_wanted = st.slider(
        "Nombre de folds pour la comparaison finale (CV)",
        2, 10, 5, 1,
        key="cv_global"
    )

    cv_obj, used_folds = make_stratified_cv(y_train_enc, folds_wanted)

    compare_rows = []

    # --- RandomForest
    if "RandomForest" in st.session_state.best_models:
        brf = st.session_state.best_models["RandomForest"]

        # normalisation des params
        n_est = int(brf["n_estimators"])
        raw_depth = brf.get("max_depth", None)
        depth_is_nan = isinstance(raw_depth, float) and pd.isna(raw_depth)
        if raw_depth is None or depth_is_nan:
            depth_val = None
        else:
            depth_val = int(raw_depth)

        rf_model = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth_val,
                random_state=42,
                n_jobs=-1
            ))
        ])

        if cv_obj is not None:
            scores = cross_val_score(
                rf_model, X_train, y_train_enc,
                cv=cv_obj, scoring="accuracy", n_jobs=-1
            )
            compare_rows.append({
                "model": "RandomForest",
                "mean_acc": float(scores.mean()),
                "std_acc": float(scores.std())
            })
        else:
            rf_model.fit(X_train, y_train_enc)
            train_pred = rf_model.predict(X_train)
            train_acc = accuracy_score(y_train_enc, train_pred)
            compare_rows.append({
                "model": "RandomForest",
                "mean_acc": train_acc,
                "std_acc": np.nan
            })

    # --- KNN
    if "KNN" in st.session_state.best_models:
        bknn = st.session_state.best_models["KNN"]
        k_val = int(bknn["k"])

        knn_model = Pipeline([
            ("prep", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=k_val))
        ])

        if cv_obj is not None:
            scores = cross_val_score(
                knn_model, X_train, y_train_enc,
                cv=cv_obj, scoring="accuracy", n_jobs=-1
            )
            compare_rows.append({
                "model": "KNN",
                "mean_acc": float(scores.mean()),
                "std_acc": float(scores.std())
            })
        else:
            knn_model.fit(X_train, y_train_enc)
            train_pred = knn_model.predict(X_train)
            train_acc = accuracy_score(y_train_enc, train_pred)
            compare_rows.append({
                "model": "KNN",
                "mean_acc": train_acc,
                "std_acc": np.nan
            })

    # --- SVM
    if "SVM" in st.session_state.best_models:
        bsvm = st.session_state.best_models["SVM"]
        C_val = float(bsvm["C"])
        kernel_val = bsvm["kernel"]

        svm_model = Pipeline([
            ("prep", preprocessor),
            ("clf", SVC(
                kernel=kernel_val,
                C=C_val,
                gamma="scale",
                probability=False
            ))
        ])

        if cv_obj is not None:
            scores = cross_val_score(
                svm_model, X_train, y_train_enc,
                cv=cv_obj, scoring="accuracy", n_jobs=-1
            )
            compare_rows.append({
                "model": "SVM",
                "mean_acc": float(scores.mean()),
                "std_acc": float(scores.std())
            })
        else:
            svm_model.fit(X_train, y_train_enc)
            train_pred = svm_model.predict(X_train)
            train_acc = accuracy_score(y_train_enc, train_pred)
            compare_rows.append({
                "model": "SVM",
                "mean_acc": train_acc,
                "std_acc": np.nan
            })

    if len(compare_rows) == 0:
        st.warning("Aucun modèle n'a pu être comparé.")
        st.stop()

    df_cmp = pd.DataFrame(compare_rows)
    st.subheader("Résultats comparés (meilleures configs)")
    st.dataframe(df_cmp, width='stretch')

    best_global = df_cmp.loc[df_cmp["mean_acc"].idxmax()]

    msg = (
        f"Meilleur modèle global : {best_global['model']} → "
        f"accuracy moyenne {best_global['mean_acc']:.3f}"
    )
    if not pd.isna(best_global["std_acc"]):
        msg += f" ± {best_global['std_acc']:.3f}"
    st.success(msg)

    # Graphe comparatif
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(
        df_cmp["model"],
        df_cmp["mean_acc"],
        yerr=df_cmp["std_acc"],
        capsize=5
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy moyenne")
    ax.set_title("Comparaison des meilleurs modèles")
    for i, v in enumerate(df_cmp["mean_acc"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    st.pyplot(fig)
