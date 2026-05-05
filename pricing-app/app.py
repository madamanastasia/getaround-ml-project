import streamlit as st
import pandas as pd
import requests
from pathlib import Path

st.set_page_config(
    page_title="Getaround — Estimation du prix",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1565C0, #0288D1);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
    }
    .result-price {
        font-size: 3rem;
        font-weight: 700;
    }
    .result-label {
        font-size: 1rem;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "https://anabeldg-getaround-pricing-api.hf.space/predict"

FEATURE_ORDER = [
    "model_key", "mileage", "engine_power", "fuel",
    "paint_color", "car_type", "private_parking_available",
    "has_gps", "has_air_conditioning", "automatic_car",
    "has_getaround_connect", "has_speed_regulator", "winter_tires"
]

st.markdown('<div class="main-title">🚗 Getaround — Estimation du prix</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Renseignez les caractéristiques du véhicule pour obtenir une estimation du prix journalier.</div>', unsafe_allow_html=True)

st.divider()

st.subheader("Caractéristiques du véhicule")

col1, col2 = st.columns(2)

with col1:
    model_key = st.selectbox("Marque", sorted([
        "Citroën", "Peugeot", "Renault", "BMW", "Mercedes",
        "Audi", "Volkswagen", "Toyota", "Ford", "Opel",
        "Mitsubishi", "Ferrari", "Porsche", "SEAT", "Nissan",
        "Honda", "Suzuki", "KIA Motors", "Alfa Romeo", "Fiat",
        "Lexus", "Lamborghini", "Maserati", "Subaru", "Dacia", "Volvo"
    ]))
    mileage = st.number_input("Kilométrage (km)", min_value=0, max_value=500000, value=50000, step=1000)
    engine_power = st.number_input("Puissance moteur (ch)", min_value=50, max_value=600, value=120, step=10)
    fuel = st.selectbox("Carburant", ["diesel", "petrol", "hybrid_petrol", "electro"])
    paint_color = st.selectbox("Couleur", ["black", "white", "grey", "blue", "red", "beige", "brown", "silver"])
    car_type = st.selectbox("Type de carrosserie", [
        "sedan", "hatchback", "suv", "van", "estate", "convertible", "coupe", "subcompact"
    ])

with col2:
    st.markdown("**Équipements**")
    private_parking_available = st.checkbox("🅿️ Parking privé disponible", value=True)
    has_gps = st.checkbox("🗺️ GPS", value=True)
    has_air_conditioning = st.checkbox("❄️ Climatisation", value=True)
    automatic_car = st.checkbox("⚙️ Boîte automatique", value=False)
    has_getaround_connect = st.checkbox("🔑 Boîtier Connect (sans clé)", value=False)
    has_speed_regulator = st.checkbox("🚀 Régulateur de vitesse", value=True)
    winter_tires = st.checkbox("🌨️ Pneus hiver", value=False)

st.divider()

if st.button("💶 Estimer le prix journalier", type="primary", use_container_width=True):

    input_row = [
        model_key,
        mileage,
        engine_power,
        fuel,
        paint_color,
        car_type,
        int(private_parking_available),
        int(has_gps),
        int(has_air_conditioning),
        int(automatic_car),
        int(has_getaround_connect),
        int(has_speed_regulator),
        int(winter_tires),
    ]

    payload = {"input": [input_row]}

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        predicted_price = response.json()["prediction"][0]

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Prix journalier estimé</div>
            <div class="result-price">{predicted_price:.0f} €</div>
            <div class="result-label">par jour</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.caption("Estimation via API FastAPI · Modèle Random Forest (R²=0.73, MAE=10.8€) · 4 843 véhicules Getaround.")

    except requests.exceptions.Timeout:
        st.error("⏱️ L'API met trop de temps à répondre. Réessayez dans quelques secondes.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Impossible de contacter l'API. Vérifiez que le Space FastAPI est actif.")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")