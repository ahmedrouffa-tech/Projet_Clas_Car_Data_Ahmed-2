import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de Gamme de Prix - Véhicules d'Occasion",
    page_icon="🚗",
    layout="centered"
)

# Titre principal
st.title("🚗 Prédicteur de Gamme de Prix de Vente")
st.write("Cette application utilise un modèle de Machine Learning (Gradient Boosting) pour classifier le prix de revente d'un véhicule d'occasion en **Prix Standard** ou **Prix Élevé**.")

# 1. Chargement des artefacts sauvegardés avec gestion robuste des erreurs de version
@st.cache_resource
def load_artifacts():
    try:
        # Charger le dictionnaire global regroupant tous les éléments
        artifacts = joblib.load('list_items.joblib')
        return artifacts
    except ModuleNotFoundError as e:
        st.error(f"❌ **Erreur d'incompatibilité de version Scikit-Learn**.\n\n"
                 f"Le modèle a été entraîné avec une version différente de celle installée actuellement sur le serveur.\n"
                 f"* Version actuelle du serveur : `{sklearn.__version__}`\n"
                 f"* Détails de l'erreur : `{e}`\n\n"
                 f"👉 **Solution** : Vérifiez que votre fichier `requirements.txt` contient bien la ligne `scikit-learn=={sklearn.__version__}` (ou la version identique à celle de votre notebook d'entraînement) et redéployez sur Streamlit Cloud.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des artefacts : {e}")
        return None

artifacts = load_artifacts()

if artifacts:
    encoders = artifacts['encoders']
    uniques = artifacts['uniques']
    scaler = artifacts['scaler']
    model = artifacts['model']

    st.sidebar.header("🔧 Caractéristiques du Véhicule")

    # Saisie des variables quantitatives
    kms_driven = st.sidebar.number_input("Kilométrage (Kms Driven)", min_value=0, max_value=1000000, value=30000, step=1000)
    present_price = st.sidebar.number_input("Prix d'origine (en Lakhs / Prix Neuf)", min_value=0.1, max_value=100.0, value=6.0, step=0.5)
    age = st.sidebar.slider("Âge du véhicule (Années)", min_value=0, max_value=30, value=5)

    # Saisie des variables catégorielles à partir des valeurs uniques d'origine
    fuel_type = st.sidebar.selectbox("Type de Carburant", uniques['Fuel_Type'])
    seller_type = st.sidebar.selectbox("Type de Vendeur", uniques['Seller_Type'])
    transmission = st.sidebar.selectbox("Type de Transmission", uniques['Transmission'])

    # Bouton de prédiction
    if st.button("🔮 Prédire la gamme de prix"):
        # Construction du DataFrame d'entrée
        input_data = pd.DataFrame({
            'Kms_Driven': [kms_driven],
            'Present_Price': [present_price],
            'Fuel_Type': [fuel_type],
            'Seller_Type': [seller_type],
            'Transmission': [transmission],
            'Age': [age]
        })

        # Encodage des variables catégorielles à l'aide des encodeurs sauvegardés
        for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
            input_data[col] = encoders[col].transform(input_data[col])

        # Normalisation des données
        input_scaled = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Affichage du résultat
        st.subheader("Résultat de la prédiction :")
        if prediction == 1:
            st.error("💰 Gamme de prix estimée : **Prix Élevé** (supérieur ou égal à la médiane de 8.88 Lakhs)")
            st.write(f"Confiance du modèle : **{prediction_proba[1]*100:.2f}%**")
        else:
            st.success("🟢 Gamme de prix estimée : **Prix Standard** (inférieur à la médiane de 8.88 Lakhs)")
            st.write(f"Confiance du modèle : **{prediction_proba[0]*100:.2f}%**")
else:
    st.warning("Le fichier 'list_items.joblib' et des versions de bibliothèques compatibles sont requis pour faire fonctionner cette application.")
