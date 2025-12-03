import streamlit as st
import pandas as pd
from pathlib import Path

from utils_ml import render_sidebar_nav

st.set_page_config(page_title="Projet ML - Classification de vins", page_icon=":microscope:", layout="wide")
render_sidebar_nav()

st.caption("Accueil - Import du dataset et définition de la colonne cible")

# --- Initialisation du state ---
if "df" not in st.session_state:
    st.session_state.df = None
if "target" not in st.session_state:
    st.session_state.target = None
if "df_src" not in st.session_state:
    st.session_state.df_src = None
if "text_cols" not in st.session_state:
    st.session_state.text_cols = []

# --- Import du dataset ---
st.header("Import du dataset")

uploaded = st.file_uploader("Charge ton fichier CSV", type=["csv"])
col1, col2 = st.columns([1, 1])

if uploaded is not None:
    file_id = f"upload:{uploaded.name}:{uploaded.size}"
    if st.session_state.df_src != file_id:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state.df_src = file_id

            if st.session_state.target not in df.columns:
                st.session_state.target = None

            st.success(f"Dataset importé : {uploaded.name}")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du CSV : {e}")

elif col1.button("Utiliser le dataset par défaut (vin.csv)"):
    path = Path("vin.csv")
    if path.exists():
        try:
            file_id = f"default:{path}:{path.stat().st_mtime}"
            if st.session_state.df_src != file_id:
                st.session_state.df = pd.read_csv(path)
                st.session_state.df_src = file_id

                if st.session_state.target not in st.session_state.df.columns:
                    st.session_state.target = None

                st.success("Dataset par défaut chargé (vin.csv).")
            else:
                st.info("Dataset par défaut déjà chargé.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture de vin.csv : {e}")
    else:
        st.error("Fichier vin.csv introuvable à la racine du projet.")

# --- Sélection de la colonne cible ---
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    st.subheader("Définition de la colonne cible")

    cols = df.columns.tolist()

    # Valeur par défaut robuste
    if st.session_state.target in cols:
        default_value = st.session_state.target
    elif "target" in cols:
        default_value = "target"
    else:
        default_value = cols[0]

    target_col = st.selectbox(
        "Sélectionne la colonne cible (variable à prédire) :",
        options=cols,
        index=cols.index(default_value),
    )

    # Sauvegarde du choix
    st.session_state.target = target_col
    st.info(f"Colonne cible sélectionnée : **{target_col}**")

    # Infos
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Infos dataset")
        st.write(f"**Shape** : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    with right:
        st.subheader("Infos cible")
        st.write(
            f"**Cible** : `{target_col}` — dtype: `{df[target_col].dtype}` / "
            f"uniques: {df[target_col].nunique(dropna=True)}"
        )

    # Aperçu
    st.markdown("---")
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(15), use_container_width=True)

    # Types
    st.markdown("### Types des colonnes")
    dtypes_df = pd.DataFrame({"Colonne": df.columns, "Type": df.dtypes.astype(str)})
    st.dataframe(dtypes_df, use_container_width=True)

    # Statistiques
    st.markdown("### Statistiques descriptives")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    # Valeurs manquantes
    st.markdown("### Valeurs manquantes par colonne")
    missing_df = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "Colonne", 0: "Valeurs nulles"})
        .sort_values("Valeurs nulles", ascending=False)
    )
    missing_df["% manquantes"] = (missing_df["Valeurs nulles"] / len(df) * 100).round(2)
    st.dataframe(missing_df, use_container_width=True)

    st.success(
        "Le dataset est prêt. Utilisez le menu 'Navigation' à gauche pour accéder au Nettoyage, à la Visualisation, la Modélisation ou l'Évaluation."
    )
else:
    st.info(
        "Importez un CSV ou cliquez sur 'Utiliser le dataset par défaut' pour activer les étapes suivantes."
    )
