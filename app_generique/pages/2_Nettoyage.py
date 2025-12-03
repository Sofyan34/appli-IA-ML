import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils_ml import render_sidebar_nav
import numpy as np

# Fonctions ------------------------------------------------------------------
# Fonction pour afficher le DataFrame dans une box
def afficher_dataframe():
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df.round(2)
        st.dataframe(df)
    else :
        st.info('Veuillez charger la donnée')

# Affiche la description
def description(cible):
    st.dataframe(cible.describe())
    if cible.dtype == 'object':
        st.dataframe(cible.value_counts(normalize=True, sort=False, dropna=False))



# Gestion des nan ------------------------------------------------------------
# 1 --- Num et cat ---
# Remplace les nan dans les features : OK
def replace_nan(list_nan, nan_replacer) :
    if pd.api.types.is_numeric_dtype(st.session_state.X_test[list_nan]) : 
        st.session_state.X_test[list_nan] = st.session_state.X_test[list_nan].fillna(nan_replacer)
        st.session_state.X_train[list_nan] = st.session_state.X_train[list_nan].fillna(nan_replacer)
    else :
        st.session_state.X_test[list_nan] = st.session_state.X_test[list_nan].fillna(str(nan_replacer))
        st.session_state.X_train[list_nan] = st.session_state.X_train[list_nan].fillna(str(nan_replacer))
    

def delete_nan(colonne) :
    # jeu de test
    st.session_state.X_test = st.session_state.X_test.dropna(subset=[colonne])
    test_filtre_index = st.session_state.X_test.index
    st.session_state.y_test = st.session_state.y_test.loc[test_filtre_index]

    # jeu de train
    st.session_state.X_train = st.session_state.X_train.dropna(subset=[colonne])
    train_filtre_index = st.session_state.X_train.index
    st.session_state.y_train = st.session_state.y_train.loc[train_filtre_index]
    

# 2 --- Numérique uniquement ---
# Remplace les nan en moyenne
def nan_moyenne(list_nan) :
    # Vérification du format
    st.session_state.X_train[list_nan].fillna(st.session_state.X_train[list_nan].mean(), inplace=True)
    st.session_state.X_test[list_nan].fillna(st.session_state.X_train[list_nan].mean(), inplace=True)
    


# Remplace les nan en médianne
def nan_median(list_nan):
    # Vérification du format
    st.session_state.X_train[list_nan].fillna(st.session_state.X_train[list_nan].median(), inplace=True)
    st.session_state.X_test[list_nan].fillna(st.session_state.X_train[list_nan].median(), inplace=True)
    

# 3 --- Cat uniquement ---
# Remplacer en Mode des valeurs présentent dans le jeu de donner.
def nan_cat_mode(list_nan) :
    _value = st.session_state.X_train[list_nan].mode()[0]

    # Remplacer les NaN
    st.session_state.X_test[list_nan] = st.session_state.X_test[list_nan].fillna(_value)
    st.session_state.X_train[list_nan] = st.session_state.X_train[list_nan].fillna(_value)




# La page commence ici ----------------------------

st.set_page_config(page_title="Modélisation", layout="wide")
render_sidebar_nav()

st.title("Nettoyage et préparation des données")

# Session state ---------------
if "df" not in st.session_state:
    st.session_state.df = None
if "split" not in st.session_state :
    st.session_state.split = False
if 'features' not in st.session_state : 
    st.session_state.features = None
if 'target' not in st.session_state :
    st.session_state.target = None
if 'missing_targets' not in st.session_state :
    st.session_state.missing_targets = None


if 'y_train' not in st.session_state :
    st.session_state.y_train = None
if 'y_test' not in st.session_state :
    st.session_state.y_test = None
if 'X_test' not in st.session_state :
    st.session_state.X_test = None
if 'X_train' not in st.session_state :
    st.session_state.X_train = None


st.markdown('#### Données :')
# Affichage du DataFrame
afficher_dataframe()


if "df" in st.session_state and st.session_state.df is not None:
    # Suppression de colonne ---------------------------------------------------
    st.divider()
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="bottom")
    with col1 :
        st.markdown('#### Suppression de colonne :')

    with col2 :
        colonnes = st.multiselect(
            "Sélectionnez les colonnes à supprimer :",
            options=st.session_state.df.columns,
            key="colonnes_a_supprimer")
    
    with col3 :
        # Bouton pour exécuter la suppression
        if st.button("Exécuter la suppression"):
            if colonnes:
                st.session_state.df = st.session_state.df.drop(columns=colonnes)
                st.success("Colonnes supprimées avec succès !")
                st.rerun()
            else:
                st.warning("Veuillez sélectionner au moins une colonne à supprimer.")

    with col4 :
        doublons = st.button(label="Supprimer les doublons")
        

        if doublons :
            st.session_state.df = st.session_state.df.drop_duplicates(ignore_index=True)
            st.rerun()

            
    # Renommer une colonne ---------------------------------------------------
    st.divider()
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="bottom")
    with col1 :
        st.markdown('#### Renommer une colonne :')


    with col2 :
        colonnes2 = st.selectbox(
            "Sélectionnez les colonnes à renommer :",
            options=st.session_state.df.columns,
            key="colonnes_a_renommer")
    
    with col3 :
        new_name = st.text_input(label='Nom_de_colonne', placeholder='Nom de la nouvelle colonne')

    with col4 :
        # Bouton pour Renommer la colonne
        if st.button("Renommer la colonne"):
            if colonnes2 :
                st.session_state.df = st.session_state.df.rename(columns={colonnes2 : new_name})
                st.success("Colonne(s) renommée(s) avec succès !")
                st.rerun()

            else:
                st.warning("Veuillez sélectionner au moins une colonne à renommer.")


    # Division de la donnée ---------------------------------------------------
    st.divider()
    
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="top")

    with col1 :
        st.markdown('#### Division de la donnée :')

    with col2 :
        # Afficher le nb de lignes
        st.markdown(f'#### {len(st.session_state.df)} lignes à séparer')
        # Boutton choix de target
        target = st.selectbox("Sélectionnez la target :",
            options=st.session_state.df.columns,
            key="Colonne target")
        
        # Vérifie s'il y a des valeurs manquantes dans la colonne target
        missing_targets = st.session_state.df[target].isna().sum()
        if missing_targets > 0 :
            st.session_state.missing_targets = int(missing_targets)
            st.session_state.df = st.session_state.df.loc[st.session_state.df[target].notna()]


        if st.session_state.missing_targets and st.session_state.missing_targets > 0 :
            st.error(f"{st.session_state.missing_targets} lignes avec valeurs manquantes, elles serons supprimées", icon="🚨")

        split_button = st.button('Diviser la donnée')

        if split_button :
            st.session_state.split = True
            st.session_state.target = target
            st.rerun()

        
    with col3:
        split_num = st.slider(min_value=1, max_value=99, value=20, label='Taille du test')
        test_coeff = split_num/100
        train_coeff = (100-split_num)/100
        st.markdown(f'Taille du jeu de test +/- {int(test_coeff*(len(st.session_state.df)))} lignes')
        st.markdown(f'Taille du jeu de train +/- {int(train_coeff*len(st.session_state.df))} lignes')


    if st.session_state.split == True :
        st.session_state.features = st.session_state.df.columns
        st.session_state.features = st.session_state.features.drop(st.session_state.target)
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(st.session_state.df[st.session_state.features], 
                                                                                                                                st.session_state.df[st.session_state.target], 
                                                                                                                                test_size=test_coeff, 
                                                                                                                                random_state=42)
        st.session_state.split = False

    # Description de la Target  ---------------------------------------------------
    st.divider()
    if (st.session_state.X_train is not None 
        and st.session_state.X_test is not None 
        and st.session_state.y_train is not None 
        and st.session_state.y_test is not None) :
        
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="top")
        with col1 :
            st.markdown('#### Description de la Target :')

        with col3 :
            st.markdown('Target (y) train:')
            description(st.session_state.y_train)

        with col4 :
            st.markdown('Target (y) test:')
            description(st.session_state.y_test)

        # Sous-Partie target mapping--------------------
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="top")

        with col2:
            st.markdown('##### Target mapping Value :')

        with col3:
            # Récupérer toutes les valeurs uniques de y_train et y_test
            target_value_list = list(st.session_state.y_train.unique())
            for ele in st.session_state.y_test.unique():
                if ele not in target_value_list:
                    target_value_list.append(ele)

            target_value_list.sort()
            # Selection de la colonne à modifier (box)
            old_target_value = st.selectbox(
                "Valeur à changer :",
                options=target_value_list,
                key="map_target"
            )

        with col4 :
            # Text box (nex value)
            new_target_value = st.text_input(label='Nouvelle valeur')

        with col5:
            # Trigger button
            change_target_value = st.button("Changer la valeur de la target")

            # Trigger button action
            if change_target_value and old_target_value and new_target_value :
                
                # Appliquer le mapping à y_train et y_test
                st.session_state.y_train = st.session_state.y_train.replace({old_target_value : new_target_value})
                st.session_state.y_test = st.session_state.y_test.replace({old_target_value : new_target_value})
                st.rerun()


        # Description des features---------------------------------------------------
        st.divider()
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="top")
        with col1 :
            st.markdown("#### Description des features:")

        with col2 :
            colonnes3 = st.selectbox(
            "Sélectionnez les colonnes à explorer:",
            options=st.session_state.features,
            key="choix de la feature")

        with col3 :
            if colonnes3 :
                st.markdown('Features (X) train:')
                description(st.session_state.X_train[colonnes3])


        with col4 :
            if colonnes3 :
                st.markdown('Features (X) test:')
                description(st.session_state.X_test[colonnes3])


        # Gestion des nan---------------------------------------------------
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="top")
        with col2 :
            st.markdown("##### Gestion des valeurs manquantes :")

            # liste des colonnes avec des nan.

            features_nan_list = []

            for ele in (st.session_state.X_train.isna().sum()[st.session_state.X_train.isna().sum() > 0].index.tolist() 
                        + st.session_state.X_test.isna().sum()[st.session_state.X_test.isna().sum() > 0].index.tolist() ):
                if ele not in features_nan_list :
                    features_nan_list.append(ele)


            # Si des nan sont trouvés
            if len(features_nan_list)> 0 :
                st.error(f'Veuillez régler les valeurs manquantes dans les colonnes :')

                with col3 :
                    # Affiche les colonnes avec nan
                    list_nan = st.selectbox("Colonne à modifier",
                                    options=features_nan_list,
                                    key="list_nan")
                    
                    nan_type = st.session_state.X_train[list_nan].dtype

                    st.info(f'{nan_type}')
                    all_format = st.checkbox(f"Appliquer le changement pour tous les formats {nan_type}")



                with col4 :
                    # Choix de la modif
                    choix_nan = st.selectbox("modification",
                                    options=["Médiane", "Moyenne", "Remplacement", "Mode", "Suppression"],
                                    key="choix_nan")


                    # Si remplacement -> input valeur
                    if choix_nan == "Remplacement" :
                        nan_replacer = st.text_input(label="valeur de remplacement")
                        
                    # WARNING
                    # trigger button : on le cache si mauvaise conditions
                    if (choix_nan == "Médiane" or choix_nan == "Moyenne")  and (not pd.api.types.is_numeric_dtype(st.session_state.X_test[list_nan])) :
                        st.error(f"{list_nan} n'est pas au format numérique !!")
                        modif_nan = None # évitre les erreurs


                    elif (choix_nan == "Mode") and (pd.api.types.is_numeric_dtype(st.session_state.X_test[list_nan])):
                        st.error(f"{list_nan} n'est pas au format catégorielle !!")
                        modif_nan = None # évitre les erreurs

                    else :
                        modif_nan = st.button("Appliquer la modification")

            


                    # Action du trigger boutton
                    # Remplacement
                    if modif_nan and (choix_nan == "Remplacement") :
                        if all_format :
                            for col in features_nan_list :
                               replace_nan(col, nan_replacer) 
                        elif not all_format :
                            replace_nan(list_nan, nan_replacer)
                        st.rerun()

                    # Moyenne
                    elif modif_nan and (choix_nan == "Moyenne") :
                        if all_format :
                            for col in features_nan_list :
                                nan_moyenne(col)

                        elif not all_format :
                            nan_moyenne(list_nan)
                    
                        st.rerun()

                    # Médiane
                    elif modif_nan and (choix_nan == "Médiane") :
                        if all_format :
                            for col in features_nan_list :
                                if pd.api.types.is_numeric_dtype(st.session_state.X_test[col]) :
                                    nan_median(col)

                        elif not all_format :
                            nan_median(list_nan)
                        st.rerun()

                    # Suppression
                    elif modif_nan and (choix_nan == "Suppression") :
                        if all_format :
                            for col in features_nan_list :
                               delete_nan(col) 
                        elif not all_format :
                            delete_nan(list_nan)
                        st.rerun()

                    # Mode
                    elif modif_nan and (choix_nan == "Mode") :
                        if all_format :
                            for col in features_nan_list :
                                if not pd.api.types.is_numeric_dtype(st.session_state.X_test[col]) :
                                    nan_cat_mode(col) 
                        elif not all_format :
                            nan_cat_mode(list_nan)
                        st.rerun()
        


            else :
                st.success(f"Il n'y a pas de valeurs manquantes.")