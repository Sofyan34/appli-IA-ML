import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def encode_target(data, target_col: str = None):
    """
    Transforme la cible en entiers (0,1,2,...) + retourne le mapping inverse.

    - Si `data` est une Series: on encode directement cette Series.
    - Si `data` est un DataFrame: on encode la colonne `target_col`.

    Le texte est normalisé (minuscules, trim, espaces multiples -> 1 espace),
    et on corrige certaines variantes (vin sucre -> vin sucré, etc.).
    """

    # 1. Récupérer la série cible
    if isinstance(data, pd.Series):
        s = data.copy()
    elif isinstance(data, pd.DataFrame):
        if target_col is None:
            raise ValueError(
                "encode_target: target_col doit être fourni si `data` est un DataFrame"
            )
        if target_col not in data.columns:
            raise ValueError(
                f"encode_target: '{target_col}' n'existe pas dans le DataFrame"
            )
        s = data[target_col].copy()
    else:
        raise TypeError(
            "encode_target attend soit une pandas Series, soit un DataFrame + target_col"
        )

    # 2. Normalisation texte pour les cas 'vin'
    #    (on garde cette étape car elle ne gêne pas si ce n'est pas du vin)
    s = (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
         .replace({
             "vin equilibre": "vin équilibré",
             "vin éuilibré": "vin équilibré",
             "vin sucre": "vin sucré"
         })
    )

    # 3. Conversion en catégories
    y_cat = s.astype("category")
    y_codes = y_cat.cat.codes.to_numpy()

    # 4. Mapping code -> label d'origine normalisé
    mapping = dict(enumerate(y_cat.cat.categories))

    return y_codes, mapping



from sklearn.preprocessing import OneHotEncoder

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    return preprocessor



def make_models(preprocessor: ColumnTransformer,
                rf_n_estimators=300, rf_max_depth=None,
                knn_k=5, svm_C=2.0, svm_gamma="scale"):
    models = {
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                n_jobs=-1,
                random_state=42
            ))
        ]),
        f"KNN (k={knn_k})": Pipeline([
            ("prep", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=knn_k))
        ]),
        f"SVM (RBF, C={svm_C})": Pipeline([
            ("prep", preprocessor),
            ("model", SVC(kernel="rbf", C=svm_C, gamma=svm_gamma, probability=True))
        ]),
    }
    return models


# --- Navigation latérale personnalisée (affiche les numéros) ---
def render_sidebar_nav():
    import streamlit as st

    # Masquer la nav multipage par défaut
    st.markdown(
        """
        <style>
        div[data-testid="stSidebarNav"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Navigation")
    # NOTE: chemins = noms EXACTS de tes fichiers dans /pages
    st.sidebar.page_link("app.py", label="0. Accueil")
    st.sidebar.page_link("pages/1_Visualisation.py", label="1. Visualisation")
    st.sidebar.page_link("pages/2_Nettoyage.py", label="2. Nettoyage")  # Ajouter cette ligne pour la page 2. Nettoyage
    st.sidebar.page_link("pages/3_Modelisation.py", label="2. Modelisation")
    st.sidebar.page_link("pages/4_Evaluation.py", label="3. Evaluation")


    # --- Utilitaire numerique/categoriel ---
def is_categorical_target(series: pd.Series, max_classes: int = 20) -> bool:
    """
    Considère la cible comme catégorielle si :
    - booléenne
    - dtype Categorical
    - object/string
    - OU si peu de modalités uniques (<= max_classes)
    """
    dtype = series.dtype
    # bool
    if is_bool_dtype(dtype):
        return True
    # Categorical
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    # chaînes / objets
    if is_object_dtype(dtype) or is_string_dtype(dtype):
        return True
    # règle de secours : faible cardinalité
    nunique = series.nunique(dropna=True)
    return nunique <= max_classes


def target_for_hue(df: pd.DataFrame, target: str):
    """Retourne une série pour 'hue' :
    - si la cible est catégorielle -> telle quelle (str)
    - si la cible est numérique -> binnings par quantiles (4 par défaut) sinon bins égaux (5)
    """
    s = df[target]
    if is_categorical_target(s):
        return s.astype(str)
    s_num = pd.to_numeric(s, errors="coerce")
    try:
        if s_num.nunique(dropna=True) <= 10:
            return s_num.astype(str)
        binned = pd.qcut(s_num, q=4, duplicates="drop")
        return binned.astype(str)
    except Exception:
        binned = pd.cut(s_num, bins=5, include_lowest=True)
        return binned.astype(str)

