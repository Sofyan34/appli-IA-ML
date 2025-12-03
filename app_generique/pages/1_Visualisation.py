import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter
import altair as alt
from pandas.api.types import CategoricalDtype

from utils_ml import render_sidebar_nav, is_categorical_target, target_for_hue

st.set_page_config(page_title="Visualisation", layout="wide")
render_sidebar_nav()

st.title("Visualisation des données")

# Vérifications

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Chargez d’abord un dataset depuis la page d’accueil.")
    st.stop()

if "target" not in st.session_state or st.session_state.target is None:
    st.warning("Définissez la colonne cible dans la page d’accueil avant de continuer.")
    st.stop()

# Chargement du dataset
df = st.session_state.df.copy()
target_col = st.session_state.target

# Traitement des inf
df = df.replace([np.inf, -np.inf], np.nan)

# Rappel
st.caption(
    f"Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes — Cible : `{target_col}`"
)

# Colonnes à exclure

st.markdown("### Colonnes à exclure de la visualisation")

all_cols = df.columns.tolist()
default_auto_hide = [
    c for c in all_cols if c.lower() in {"id", "index"} or c.lower().endswith("id")
]
prev_hidden = st.session_state.get("viz_excluded_cols", default_auto_hide)

excluded_cols = st.multiselect(
    "Retirer des colonnes (ne seront pas proposées dans les graphiques) :",
    options=all_cols,
    default=[c for c in prev_hidden if c in all_cols],
)

if target_col in excluded_cols:
    st.warning(
        f"La colonne cible `{target_col}` ne peut pas être exclue. Elle a été retirée."
    )
    excluded_cols = [c for c in excluded_cols if c != target_col]

st.session_state.viz_excluded_cols = excluded_cols
dfv = df.drop(columns=excluded_cols, errors="ignore")

st.caption(
    "Colonnes exclues : " + (", ".join(excluded_cols) if excluded_cols else "aucune")
)

# Gestion des NA

st.markdown("### Gestion des valeurs manquantes")
na_mode = st.radio(
    "Que faire des NA pour les graphiques ?",
    [
        "Ignorer (supprimer les lignes manquantes)",
        "Garder et étiqueter « (NA) » (catégorielles)",
    ],
    horizontal=True,
    index=0,
)

MISSING_TOKENS = {"", "na", "n/a", "none", "null", "nan", "-", "?", "undefined"}


def normalize_missing(s: pd.Series) -> pd.Series:
    s2 = s.copy()

    def _is_missing(v):
        if pd.isna(v):
            return True
        if isinstance(v, str):
            return v.strip().lower() in MISSING_TOKENS
        return False

    return s2.mask(s2.map(_is_missing), np.nan)


def prep_cat(s: pd.Series) -> pd.Series:
    s2 = normalize_missing(s)
    return s2.fillna("(NA)") if "Garder" in na_mode else s2


def num_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


# Colonnes texte

st.markdown("---")
st.subheader("Colonnes texte")

# Multiselect + persistance
text_options = [c for c in dfv.columns if c != target_col]
selected_text_cols = st.multiselect(
    "Sélectionne les colonnes à considérer comme **textuelles** :",
    options=text_options,
    default=[c for c in st.session_state.get("text_cols", []) if c in text_options],
)
st.session_state.text_cols = selected_text_cols

if not selected_text_cols:
    st.info("Aucune colonne texte sélectionnée.")
else:
    selected_text_col = st.selectbox(
        "Choisis une colonne texte à explorer :", options=selected_text_cols
    )

    text_series = dfv[selected_text_col].dropna().astype(str)
    full_text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s']", " ", " ".join(text_series).lower())

    extra_stopwords = {
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "un",
        "une",
        "et",
        "en",
        "à",
        "au",
        "aux",
        "dans",
        "pour",
        "sur",
        "par",
        "ce",
        "cet",
        "cette",
        "ces",
        "est",
        "qui",
        "que",
        "ou",
        "où",
        "avec",
        "sans",
        "ne",
        "pas",
        "plus",
        "moins",
    }
    stopwords = STOPWORDS.union(extra_stopwords)

    if full_text.strip():
        wc = WordCloud(
            stopwords=stopwords,
            background_color="white",
            max_words=100,
            width=800,
            height=400,
            colormap="viridis",
        ).generate(full_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        st.caption(f"Nuage de mots pour : `{selected_text_col}`")

        top_n = st.slider(
            "Nombre de mots à afficher (Top N)",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )

        words = [w for w in full_text.split() if w not in stopwords and len(w) > 2]
        common_words = Counter(words).most_common(top_n)

        if common_words:
            df_words = pd.DataFrame(common_words, columns=["Mot", "Fréquence"])
            chart = (
                alt.Chart(df_words)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Mot:N", sort=alt.SortField("Fréquence", order="descending")
                    ),
                    y=alt.Y("Fréquence:Q"),
                    tooltip=["Mot", "Fréquence"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Pas de mots fréquents après filtrage des stopwords.")
    else:
        st.info("Pas de texte exploitable dans cette colonne.")


# Distribution de la cible

st.markdown("---")
st.subheader(f"Distribution de la variable cible : `{target_col}`")
fig, ax = plt.subplots()
if is_categorical_target(df[target_col]):
    s = prep_cat(df[target_col])
    if "Ignorer" in na_mode:
        s = s.dropna()
    s.astype(str).value_counts(normalize=True, dropna=False).plot(kind="bar", ax=ax)
    ax.set_title("Répartition des classes")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Proportion")
else:
    x = num_clean(df[target_col])
    sns.histplot(x, kde=True, ax=ax)
    ax.set_title("Distribution (cible numérique)")
    ax.set_xlabel(target_col)
st.pyplot(fig)

# Variables catégorielles

st.markdown("---")
st.subheader("Variables catégorielles")

non_num_cols = [
    c for c in dfv.select_dtypes(exclude=[np.number]).columns if c != target_col
]
non_num_cols = [c for c in non_num_cols if c not in selected_text_cols]

explicit_cat = [
    c
    for c in dfv.columns
    if isinstance(dfv[c].dtype, CategoricalDtype) and c != target_col
]
cat_cols_all = list(dict.fromkeys([*non_num_cols, *explicit_cat]))

if not cat_cols_all:
    st.info("Aucune variable catégorielle détectée (hors texte/cible).")
else:
    cats_to_show = st.multiselect(
        "Choisis les variables catégorielles à afficher",
        options=cat_cols_all,
        default=cat_cols_all[:5],
    )
    top_k = st.slider("Nombre max de modalités à afficher (Top K)", 3, 30, 10, 1)

    if cats_to_show:
        n, ncols = len(cats_to_show), 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows)
        )
        axes = np.atleast_1d(axes).ravel()

        for i, col in enumerate(cats_to_show):
            ax = axes[i]
            try:
                s = prep_cat(dfv[col])
                s = s.dropna() if "Ignorer" in na_mode else s
                vc = (
                    s.astype(str).value_counts(normalize=True, dropna=False).head(top_k)
                )
                vc.plot(kind="bar", ax=ax)
                ax.set_title(col)
                ax.set_xlabel("")
                ax.set_ylabel("Proportion")
            except Exception as e:
                ax.text(0.5, 0.5, f"Erreur pour {col}\n{e}", ha="center", va="center")
                ax.set_axis_off()

        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)
    else:
        st.info(
            "Sélectionne au moins une variable catégorielle pour afficher les barres."
        )

    # Comparer une variable catégorielle avec la cible

    st.markdown("### Comparer une variable catégorielle avec la cible")

    # Cible NUMÉRIQUE
    if not is_categorical_target(df[target_col]):
        if cat_cols_all:
            comp_col = st.selectbox(
                "Catégorielle pour la comparaison (couleurs) :",
                options=cat_cols_all,
                index=0,
            )
            bins_comp = st.slider("Nombre de barres (comparaison)", 10, 80, 30, 5)
            kde_comp = st.checkbox("Afficher la courbe KDE (comparaison)", value=False)

            try:
                sub = dfv[[target_col, comp_col]].copy()
                sub[target_col] = pd.to_numeric(
                    sub[target_col], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)
                sub[comp_col] = prep_cat(sub[comp_col])
                sub = sub.dropna(subset=[target_col])
                if "Ignorer" in na_mode:
                    sub = sub.dropna(subset=[comp_col])

                if sub.empty:
                    st.info("Données insuffisantes pour la comparaison.")
                else:
                    fig_cmp, ax_cmp = plt.subplots(figsize=(7, 4))
                    sns.histplot(
                        data=sub,
                        x=target_col,
                        hue=comp_col,
                        bins=bins_comp,
                        kde=kde_comp,
                        element="step",
                        ax=ax_cmp,
                    )
                    ax_cmp.set_title(f"{target_col} par {comp_col}")
                    ax_cmp.set_xlabel(target_col)
                    ax_cmp.set_ylabel("Effectif")
                    st.pyplot(fig_cmp)
            except Exception as e:
                st.warning(f"Comparaison indisponible : {e}")
        else:
            st.info("Aucune variable catégorielle disponible pour la comparaison.")

    # Cible CATÉGORIELLE
    else:
        candidates = [c for c in cat_cols_all if c != target_col]
        if candidates:
            comp_col = st.selectbox(
                "Variable catégorielle à comparer avec la cible :",
                options=candidates,
                index=0,
            )

            top_k_rows = st.slider("Top K modalités (axe lignes)", 2, 30, 10, 1)
            top_k_cols = st.slider("Top K modalités (axe colonnes)", 2, 30, 10, 1)

            try:
                tmp = dfv[[comp_col, target_col]].copy()
                tmp[comp_col] = prep_cat(tmp[comp_col])
                tmp[target_col] = prep_cat(tmp[target_col])
                if "Ignorer" in na_mode:
                    tmp = tmp.dropna(subset=[comp_col, target_col])

                vc_rows = tmp[comp_col].astype(str).value_counts()
                vc_cols = tmp[target_col].astype(str).value_counts()
                top_rows = vc_rows.head(top_k_rows).index
                top_cols = vc_cols.head(top_k_cols).index
                sub_df = tmp[
                    tmp[comp_col].astype(str).isin(top_rows)
                    & tmp[target_col].astype(str).isin(top_cols)
                ]

                if sub_df.empty:
                    st.info("Données insuffisantes pour la comparaison.")
                else:
                    fig_cmp, ax_cmp = plt.subplots(figsize=(7, 4))
                    ct = (
                        pd.crosstab(
                            sub_df[comp_col].astype(str),
                            sub_df[target_col].astype(str),
                            normalize="index",
                        )
                        * 100
                    )
                    ct.plot(kind="bar", ax=ax_cmp)
                    ax_cmp.set_title(f"{comp_col} par {target_col}")
                    ax_cmp.set_xlabel(comp_col)
                    ax_cmp.set_ylabel("Proportion")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig_cmp)
            except Exception as e:
                st.warning(f"Comparaison catégorielle indisponible : {e}")
        else:
            st.info("Aucune autre variable catégorielle à comparer avec la cible.")

# Boxplots (numériques vs cible)
if is_categorical_target(df[target_col]):
    st.markdown("### Boxplots (numériques vs cible)")
    num_cols_box = [
        c for c in dfv.select_dtypes(include=[np.number]).columns if c != target_col
    ]
    if num_cols_box:
        sel = st.multiselect(
            "Variables numériques pour boxplots", num_cols_box, default=num_cols_box[:6]
        )
        if sel:
            melted = dfv[sel + [target_col]].copy()
            melted[target_col] = prep_cat(melted[target_col])
            if "Ignorer" in na_mode:
                melted = melted.dropna(subset=[target_col])
            for c in sel:
                melted[c] = pd.to_numeric(melted[c], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
            melted = melted.melt(
                id_vars=target_col,
                value_vars=sel,
                var_name="Variable",
                value_name="Valeur",
            )
            melted = melted.dropna(subset=["Valeur"])

            if melted.empty:
                st.info("Aucune valeur numérique non nulle pour les boxplots.")
            else:
                fig, ax = plt.subplots(figsize=(1.2 * len(sel) + 5, 5))
                sns.boxplot(
                    data=melted, x="Variable", y="Valeur", hue=target_col, ax=ax
                )
                ax.set_title(f"Distribution des variables numériques par {target_col}")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)
    else:
        st.info("Aucune variable numérique disponible pour les boxplots.")

# Histogrammes : variables numériques

st.markdown("---")
st.subheader("Histogrammes des variables numériques")

num_df = dfv.select_dtypes(include=["number"])
num_cols_all = [c for c in num_df.columns if c != target_col]

if not num_cols_all:
    st.info("Aucune variable numérique (hors cible) disponible pour les histogrammes.")
else:
    cols_to_show = st.multiselect(
        "Choisis les variables numériques à afficher",
        options=num_cols_all,
        default=num_cols_all[:10],
    )

    col_controls1, col_controls2, col_controls3 = st.columns([1, 1, 1])
    with col_controls1:
        bins = st.slider(
            "Nombre de barres", min_value=10, max_value=80, value=30, step=5
        )
    with col_controls2:
        show_kde = st.checkbox("Afficher la courbe KDE", value=True)
    with col_controls3:
        color_by_target = st.checkbox("Colorer par la cible", value=True)

    if cols_to_show:
        n, ncols = len(cols_to_show), 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows)
        )
        axes = np.atleast_1d(axes).ravel()

        for i, col in enumerate(cols_to_show):
            ax = axes[i]
            try:
                plot_df = dfv[[col]].copy().rename(columns={col: "__x__"})
                plot_df["__x__"] = pd.to_numeric(
                    plot_df["__x__"], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

                if color_by_target:
                    plot_df["__hue__"] = target_for_hue(df, target_col)
                    if "Garder" in na_mode:
                        plot_df["__hue__"] = normalize_missing(
                            plot_df["__hue__"]
                        ).fillna("(NA)")
                        plot_df = plot_df.dropna(subset=["__x__"])
                    else:
                        plot_df["__hue__"] = normalize_missing(plot_df["__hue__"])
                        plot_df = plot_df.dropna(subset=["__x__", "__hue__"])

                    sns.histplot(
                        data=plot_df,
                        x="__x__",
                        hue="__hue__",
                        bins=bins,
                        kde=show_kde,
                        element="step",
                        ax=ax,
                    )
                else:
                    plot_df = plot_df.dropna(subset=["__x__"])
                    sns.histplot(
                        data=plot_df,
                        x="__x__",
                        bins=bins,
                        kde=show_kde,
                        element="step",
                        ax=ax,
                    )

                ax.set_title(col)
                ax.set_xlabel("")
                ax.set_ylabel("")
            except Exception as e:
                ax.text(0.5, 0.5, f"Erreur pour {col}\n{e}", ha="center", va="center")
                ax.set_axis_off()

        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)
    else:
        st.info(
            "Sélectionne au moins une variable numérique pour voir les histogrammes."
        )

# Matrice de corrélation

st.subheader("Matrice de corrélation (features numériques)")
num_df = dfv.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
num_df = num_df.loc[:, num_df.std(numeric_only=True) > 0]

if num_df.shape[1] >= 2:
    corr = num_df.corr(numeric_only=True).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        annot_kws={"size": 8},
    )
    st.pyplot(fig)
else:
    st.info("Pas assez de colonnes numériques non constantes pour la corrélation.")
