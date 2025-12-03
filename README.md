# Préparation de l'environnement virtuel :
python -m venv .venv
.\.venv\Scripts\Activate

# Une fois dans le .venv, installer les prérequis : 
python.exe -m pip install --upgrade pip

pip install -r requirements.txt

pip install streamlit pandas numpy matplotlib seaborn scikit-learn

pip freeze > requirements.txt

# 1. app_vin : Développement d'une IA pouvant prédire la catégorie du vin en fonction de ses caractéristiques
streamlit run app_vin/app_vin.py

# 2. app_generique : Développement d'une IA généraliste pouvant lire des CSV et utiliser leur contenu
streamlit run app_generique/app.py
