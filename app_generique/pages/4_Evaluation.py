import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pickle

from utils_ml import render_sidebar_nav

# ------------------------------------------------------------------
# CONFIG PAGE
# ------------------------------------------------------------------
st.set_page_config(page_title="Évaluation", page_icon=":test_tube:", layout="wide")
render_sidebar_nav()
st.title("Évaluation finale des meilleurs modèles")
st.caption("On évalue uniquement les meilleures configs trouvées dans l’onglet Modélisation ✅")

# ------------------------------------------------------------------
# CONTRÔLES DE BASE
# ------------------------------------------------------------------
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("❗ Veuillez d’abord charger un dataset depuis la page d’accueil.")
    st.stop()

if "target" not in st.session_state or st.session_state.target not in st.session_state.df.columns:
    st.warning("❗ Définissez la colonne cible dans la page d'accueil.")
    st.stop()

if "X_train" not in st.session_state or "y_train" not in st.session_state:
    st.warning("❗ Vous devez passer par la page 'Nettoyage' pour splitter les données.")
    st.stop()

if "best_models" not in st.session_state or len(st.session_state.best_models) == 0:
    st.warning("❗ Aucun meilleur modèle enregistré. Allez dans 'Modélisation' et lancez les études RandomForest / KNN / SVM.")
    st.stop()

# ------------------------------------------------------------------
# RÉCUP DES DONNÉES PRÉPARÉES
# ------------------------------------------------------------------
if "X_test" in st.session_state:
    X_full = pd.concat(
        [st.session_state.X_train, st.session_state.X_test],
        ignore_index=True
    )
else:
    X_full = st.session_state.X_train.copy()

if "y_test" in st.session_state:
    y_full = pd.concat(
        [st.session_state.y_train, st.session_state.y_test],
        ignore_index=True
    )
else:
    y_full = st.session_state.y_train.copy()

target_name = st.session_state.target

# encodage de la target -> entiers
def encode_labels_for_eval(y_series: pd.Series):
    y_norm = y_series.astype(str).str.strip().str.lower()
    y_cat = y_norm.astype("category")
    codes = y_cat.cat.codes.to_numpy()
    code_to_label = dict(enumerate(y_cat.cat.categories))
    inv_label_to_code = {v: k for k, v in code_to_label.items()}
    return codes, code_to_label, inv_label_to_code

y_codes, code_to_label, label_to_code = encode_labels_for_eval(y_full)

# split train/test pour l'éval finale
X_train_eval, X_test_eval, y_train_eval_codes, y_test_eval_codes = train_test_split(
    X_full,
    y_codes,
    test_size=0.2,
    random_state=42,
    stratify=y_codes if len(pd.Series(y_codes).unique()) > 1 else None
)

# ------------------------------------------------------------------
# PIPELINE PREPROCESS (doit matcher Modélisation)
# ------------------------------------------------------------------
num_cols = X_full.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X_full.columns if c not in num_cols]

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
    remainder="drop",
)

# ------------------------------------------------------------------
# FONCTIONS POUR RECONSTRUIRE CHAQUE MEILLEUR MODÈLE
# ------------------------------------------------------------------
def build_best_model(model_key: str):
    """
    model_key = "RandomForest" | "KNN" | "SVM"
    On va chercher st.session_state.best_models[model_key]
    et recréer le pipeline final avec ces hyperparams.
    """
    if model_key not in st.session_state.best_models:
        return None, None  # pas dispo

    best_cfg = st.session_state.best_models[model_key]

    if model_key == "RandomForest":
        raw_depth = best_cfg.get("max_depth", None)
        depth_is_nan = isinstance(raw_depth, float) and pd.isna(raw_depth)
        if raw_depth is None or depth_is_nan:
            max_depth = None
        else:
            max_depth = int(raw_depth)

        n_est = int(best_cfg.get("n_estimators", 100))

        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )

    elif model_key == "KNN":
        k = int(best_cfg.get("k", 5))
        clf = KNeighborsClassifier(n_neighbors=k)

    elif model_key == "SVM":
        kernel = best_cfg.get("kernel", "rbf")
        C_val = float(best_cfg.get("C", 1.0))
        clf = SVC(kernel=kernel, C=C_val, gamma="scale", probability=False)

    else:
        return None, None

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])

    label = f"{model_key} | params: {best_cfg}"
    return pipe, label

# ------------------------------------------------------------------
# ONGLETs
# ------------------------------------------------------------------
tabs = st.tabs([
    "Comparaison des meilleurs modèles",
    "Analyse détaillée d’un modèle",
    "Sauvegarde d’un modèle .pkl",
    "Prédiction manuelle"
])

# ------------------------------------------------------------------
# ONGLET 1 — COMPARAISON DES MEILLEURS MODÈLES
# ------------------------------------------------------------------
with tabs[0]:
    st.subheader("⚖️ Comparaison directe des meilleurs modèles trouvés en Modélisation")

    available_best = list(st.session_state.best_models.keys())
    selected_eval = st.multiselect(
        "Sélectionne les modèles à comparer",
        available_best,
        default=available_best,
    )

    if st.button("Évaluer les modèles sélectionnés"):
        results_table = []
        cols = st.columns(len(selected_eval)) if selected_eval else []

        for i, mname in enumerate(selected_eval):
            model_pipe, label_txt = build_best_model(mname)
            if model_pipe is None:
                st.warning(f"{mname} pas dispo dans best_models.")
                continue

            model_pipe.fit(X_train_eval, y_train_eval_codes)
            y_pred_codes = model_pipe.predict(X_test_eval)

            acc = accuracy_score(y_test_eval_codes, y_pred_codes)

            # mapping vers labels lisibles
            y_true_lbl = pd.Series(y_test_eval_codes).map(code_to_label)
            y_pred_lbl = pd.Series(y_pred_codes).map(code_to_label)

            report_dict = classification_report(
                y_true_lbl,
                y_pred_lbl,
                output_dict=True,
                zero_division=0
            )

            results_table.append({
                "model": mname,
                "accuracy_test": acc,
                "params": st.session_state.best_models[mname],
            })

            with cols[i]:
                st.markdown(f"### {mname}")
                st.json(st.session_state.best_models[mname])  # hyperparams retenus
                st.metric("Accuracy test", f"{acc:.3f}")

                st.markdown("Rapport de classification")
                st.dataframe(
                    pd.DataFrame(report_dict).T.style.format(precision=3),
                    width='stretch'
                )

                st.markdown("Matrice de confusion")
                fig_cm, ax_cm = plt.subplots()
                ConfusionMatrixDisplay(
                    confusion_matrix(
                        y_true_lbl,
                        y_pred_lbl,
                        labels=sorted(code_to_label.values())
                    )
                ).plot(ax=ax_cm, cmap="Blues", colorbar=False, xticks_rotation=45)
                plt.tight_layout()
                st.pyplot(fig_cm)

        if results_table:
            st.subheader("Récapitulatif global")
            st.dataframe(pd.DataFrame(results_table), width='stretch')

            best_global = max(results_table, key=lambda r: r["accuracy_test"])
            st.success(
                f"🏆 Meilleur modèle global : {best_global['model']} "
                f"→ accuracy_test = {best_global['accuracy_test']:.3f}"
            )

# ------------------------------------------------------------------
# ONGLET 2 — ANALYSE DÉTAILLÉE
# ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Analyse détaillée d’un seul meilleur modèle")

    mname = st.selectbox(
        "Choisir un modèle parmi les meilleurs",
        list(st.session_state.best_models.keys())
    )

    if st.button("Analyser ce modèle"):
        model_pipe, label_txt = build_best_model(mname)
        if model_pipe is None:
            st.error("Modèle introuvable.")
            st.stop()

        model_pipe.fit(X_train_eval, y_train_eval_codes)
        y_pred_codes = model_pipe.predict(X_test_eval)

        y_true_lbl = pd.Series(y_test_eval_codes).map(code_to_label)
        y_pred_lbl = pd.Series(y_pred_codes).map(code_to_label)

        st.markdown("#### Hyperparamètres retenus")
        st.json(st.session_state.best_models[mname])

        st.markdown("#### Matrice de confusion")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(
                y_true_lbl,
                y_pred_lbl,
                labels=sorted(code_to_label.values())
            )
        ).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Rapport de classification")
        st.text(metrics.classification_report(y_true_lbl, y_pred_lbl))

        # dispo pour les autres onglets
        st.session_state["last_trained_model"] = model_pipe
        st.session_state["last_mapping"] = code_to_label   # code -> label
        st.session_state["last_model_name"] = mname
        st.success("Modèle prêt pour sauvegarde / prédiction.")


# ------------------------------------------------------------------
# ONGLET 3 — SAUVEGARDE EN PICKLE
# ------------------------------------------------------------------
with tabs[2]:
    st.subheader("💾 Sauvegarder le meilleur modèle entraîné (pickle .pkl)")

    if "last_trained_model" not in st.session_state:
        st.warning("Lance d'abord l'onglet 'Analyse détaillée' pour entraîner et charger un modèle.")
        st.stop()

    mname_loaded = st.session_state["last_model_name"]
    st.info(f"Modèle actuellement chargé : **{mname_loaded}**")

    filename = st.text_input(
        "Nom du fichier .pkl à créer",
        value=f"best_{mname_loaded.lower()}.pkl"
    )

    if st.button("Sauvegarder en .pkl"):
        artifact = {
            "model": st.session_state["last_trained_model"],
            "mapping_code_to_label": st.session_state["last_mapping"],  # {0:'classA',1:'classB',...}
            "features": list(X_full.columns),
            "target_col": target_name,
        }

        with open(filename, "wb") as f:
            pickle.dump(artifact, f)

        st.success(f"✅ Modèle sauvegardé : {filename}")


# ------------------------------------------------------------------
# ONGLET 4 — PRÉDICTION MANUELLE MULTI-MODÈLE
# ------------------------------------------------------------------
with tabs[3]:
    st.subheader("Prédiction manuelle indépendante")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Chargez d'abord un dataset dans la page d'accueil.")
        st.stop()

    df = st.session_state.df.copy()
    
    # Choix de la colonne cible
    target_col = st.selectbox(
        "Sélectionnez la colonne cible",
        options=[c for c in df.columns],
        index=df.columns.get_loc(st.session_state.target) 
        if "target" in st.session_state else 0
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encodage des labels
    y_codes = y.astype(str).str.strip().str.lower().astype('category').cat.codes
    code_to_label = dict(enumerate(y.astype(str).str.strip().str.lower().astype('category').cat.categories))

    # Sélection du modèle
    model_choice = st.selectbox("Choisir le modèle", ["Random Forest", "KNN", "SVM"])

    # Pipeline preprocessing
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # Choix du modèle
    if model_choice == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "KNN":
        k = st.slider("Nombre de voisins (k)", 1, 20, 5)
        clf = KNeighborsClassifier(n_neighbors=k)
    elif model_choice == "SVM":
        kernel = st.selectbox("Kernel SVM", ["linear", "rbf", "poly"])
        C = st.number_input("Paramètre C", 0.01, 10.0, 1.0, 0.01)
        clf = SVC(kernel=kernel, C=C, gamma="scale", probability=False)

    # Pipeline complet
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])

    # Entraînement sur tout le dataset
    pipe.fit(X, y_codes)

    # Collecte des valeurs utilisateur
    st.markdown("### Renseignez les valeurs des variables :")
    user_input = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            val = st.number_input(
                f"{col} :",
                min_value=float(X[col].min()),
                max_value=float(X[col].max()),
                value=float(X[col].mean()),
                key=f"manual_input_{col}"
            )
        else:
            options = sorted(X[col].dropna().astype(str).unique())
            val = st.selectbox(f"{col} :", options, key=f"manual_input_{col}")
        user_input[col] = val

    if st.button("Prédire la classe"):
        input_df = pd.DataFrame([user_input])
        pred_code = pipe.predict(input_df)[0]
        pred_label = code_to_label[pred_code]
        st.success(f"Prédiction : **{pred_label}**")