import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# --- Configuration Streamlit ---
st.set_page_config(
    page_title="Projet ML",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Réinitialisation du cache Streamlit ---
st.cache_data.clear()

# Obtenir le chemin absolu du dossier contenant ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le CSV
csv_path = os.path.join(BASE_DIR, "vin.csv")

# --- Chargement du fichier CSV ---
df = pd.read_csv(csv_path)

st.write("Valeurs uniques dans target (avant nettoyage) :", df["target"].unique())

# --- Suppression de la première colonne inutile (ex: Unnamed ou index) ---
if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
    df = df.drop(df.columns[0], axis=1)

# --- Nettoyage et correction des libellés de la cible ---
df["target"] = (
    df["target"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "vin éuilibré": "vin équilibré",
        "vin euilibré": "vin équilibré",
        "vin equilibré": "vin équilibré",
        "vin equilibre": "vin équilibré"
    })
)

# --- Remise au bon format (majuscules + accents corrects) ---
df["target"] = df["target"].replace({
    "vin amer": "Vin amer",
    "vin sucré": "Vin sucré",
    "vin équilibré": "Vin équilibré"
})

# --- Vérification du résultat ---
st.write("Valeurs uniques corrigées :", df["target"].unique())

# --- Aperçu du dataset ---
st.subheader("Aperçu du dataset")
st.dataframe(df.head())

# --- Séparation features / target ---
X = df.drop("target", axis=1)
y = df["target"]

# --- Encodage des labels textuels (après nettoyage complet) ---
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Division du dataset ---
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# --- Tabs Streamlit ---
tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs([
    "Traitement des données", 
    "Visualisations", 
    "Modélisation", 
    "Évaluation"
])

with tabs_1:
    st.header("Traitement des données")
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())

    st.write("**Dimensions :**", df.shape)
    st.write("**Colonnes :**", list(df.columns))
    st.write("**Valeurs manquantes :**", df.isna().sum().sum())
    st.write("**Valeurs uniques dans Target :**", df["target"].unique())

    st.success("Les données ont été nettoyées et divisées en train/test (80/20).")

with tabs_2:
    st.header("Visualisation des données")

    numeric_cols = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_cols.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)


    st.subheader("Distribution d'une variable")
    feature = st.selectbox("Choisis une caractéristique :", X.columns)
    fig, ax = plt.subplots()
    sns.histplot(df, x=feature, hue="target", kde=True, ax=ax)
    st.pyplot(fig)

with tabs_3:
    st.header("Modélisation")

    model_choice = st.selectbox("Choisis ton modèle :", ["Decision Tree", "Réseau de neurones"])
    
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)

    pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    st.success("Modèle entraîné avec succès !")
    st.write("**Accuracy (train) :**", pipe.score(X_train, y_train))
    st.write("**Accuracy (test) :**", pipe.score(X_test, y_test))
    
with tabs_4:
    st.header("Évaluation")

    y_pred = pipe.predict(X_test)

    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Rapport de classification")
    st.text(metrics.classification_report(y_test, y_pred))

    st.subheader("Faire une prédiction")
    user_input = []
    for col in X.columns:
        val = st.number_input(f"{col} :", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        user_input.append(val)
    
    if st.button("Prédire la catégorie du vin"):
        prediction = pipe.predict([user_input])[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"La catégorie prédite du vin est : **{predicted_label}**")
