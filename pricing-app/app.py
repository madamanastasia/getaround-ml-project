import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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

APP_DIR = Path(__file__).resolve().parent
PRICING_PATH = APP_DIR / "get_around_pricing_project.csv"


@st.cache_data
def load_pricing():
    dfp = pd.read_csv(PRICING_PATH)
    dfp = dfp.loc[:, ~dfp.columns.str.match(r"^Unnamed")]
    return dfp


@st.cache_resource
def train_model(df):
    TARGET = "rental_price_per_day"
    df = df.dropna()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    cat_cols = ["model_key", "fuel", "paint_color", "car_type"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X, y)
    return pipe


pricing_df = load_pricing()
pipe = train_model(pricing_df)

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

    input_data = pd.DataFrame([{
        "model_key": model_key,
        "mileage": mileage,
        "engine_power": engine_power,
        "fuel": fuel,
        "paint_color": paint_color,
        "car_type": car_type,
        "private_parking_available": int(private_parking_available),
        "has_gps": int(has_gps),
        "has_air_conditioning": int(has_air_conditioning),
        "automatic_car": int(automatic_car),
        "has_getaround_connect": int(has_getaround_connect),
        "has_speed_regulator": int(has_speed_regulator),
        "winter_tires": int(winter_tires),
    }])

    predicted_price = pipe.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Prix journalier estimé</div>
        <div class="result-price">{predicted_price:.0f} €</div>
        <div class="result-label">par jour</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    med = float(pricing_df["rental_price_per_day"].median())
    diff = predicted_price - med
    if diff > 0:
        st.info(f"📈 Ce prix est **{diff:.0f} €** au-dessus du prix médian du marché ({med:.0f} €/jour).")
    else:
        st.info(f"📉 Ce prix est **{abs(diff):.0f} €** en-dessous du prix médian du marché ({med:.0f} €/jour).")

    st.caption("Estimation basée sur un modèle Random Forest (R²=0.73, MAE=10.8€) entraîné sur 4 843 véhicules Getaround.")