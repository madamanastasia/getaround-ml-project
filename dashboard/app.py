import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Getaround — Analyse du délai de retour", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "get_around_delay_analysis.csv"
PRICING_PATH = APP_DIR / "get_around_pricing_project.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


@st.cache_data
def load_pricing():
    dfp = pd.read_csv(PRICING_PATH)
    dfp = dfp.loc[:, ~dfp.columns.str.match(r"^Unnamed")]
    return dfp


df = load_data()
pricing_df = load_pricing()

MEDIAN_PRICE = float(pricing_df["rental_price_per_day"].median())
MEAN_PRICE = float(pricing_df["rental_price_per_day"].mean())

st.title("Getaround — Analyse du tampon de retard (2017)")

st.markdown(
    """
Ce tableau de bord explore le compromis lié à l'introduction d'un **délai minimum** entre deux locations consécutives.
Un tampon réduit les frictions causées par les retours tardifs, mais peut diminuer l'utilisation de la plateforme.
"""
)

st.subheader("Statistiques générales des retards")

total_with_delay = df[df["state"] == "ended"]["delay_at_checkout_in_minutes"].dropna()
late = total_with_delay[total_with_delay > 0]
pct_late = len(late) / len(total_with_delay) * 100
median_late = late.median()
has_prev_pct = df["previous_ended_rental_id"].notna().sum() / len(df) * 100

col_a, col_b, col_c = st.columns(3)
col_a.metric(
    "Locations rendues en retard",
    f"{pct_late:.1f}%",
    f"{len(late):,} sur {len(total_with_delay):,}"
)
col_b.metric(
    "Retard médian (retardataires)",
    f"{median_late:.0f} min",
    "parmi les locations en retard"
)
col_c.metric(
    "Locations avec une précédente",
    f"{has_prev_pct:.1f}%",
    f"{df['previous_ended_rental_id'].notna().sum():,} locations"
)

with st.sidebar:
    st.header("Paramètres de la politique")
    scope = st.selectbox(
        "Périmètre",
        ["Tous les véhicules", "Connect uniquement", "Mobile uniquement"],
        index=0
    )
    threshold = st.slider(
        "Délai minimum (minutes)",
        min_value=0,
        max_value=720,
        value=120,
        step=5
    )

    st.header("Visualisation")
    clip_mode = st.selectbox(
        "Écrêtage des retards",
        ["Aucun", "Percentiles (1–99)", "Plage fixe (±24h)"],
        index=1
    )
    bins = st.slider("Nombre de classes (histogramme)", 20, 200, 60, step=10)

    st.header("Filtres")
    include_canceled = st.checkbox("Inclure les locations annulées", value=False)

work = df.copy()

if not include_canceled:
    work = work[work["state"] == "ended"].copy()

if scope == "Connect uniquement":
    work = work[work["checkin_type"] == "connect"].copy()
elif scope == "Mobile uniquement":
    work = work[work["checkin_type"] == "mobile"].copy()

ended = df[df["state"] == "ended"][["rental_id", "delay_at_checkout_in_minutes"]].copy()
ended["delay_at_checkout_in_minutes"] = ended["delay_at_checkout_in_minutes"].fillna(0)

prev_delay_map = dict(zip(ended["rental_id"].astype(float), ended["delay_at_checkout_in_minutes"]))
work["previous_delay_min"] = work["previous_ended_rental_id"].map(prev_delay_map).fillna(0)

work["gap_min"] = work["time_delta_with_previous_rental_in_minutes"].fillna(np.inf)
work["impact_on_next_driver_min"] = np.maximum(0, work["previous_delay_min"] - work["gap_min"])

work["affected_by_policy"] = work["gap_min"] < threshold
work["problematic"] = work["impact_on_next_driver_min"] > 0
work["solved_by_policy"] = work["problematic"] & work["affected_by_policy"]

eligible = np.isfinite(work["gap_min"])
gap_pos = work["gap_min"].where(eligible, 0).clip(lower=0)
work["blocked_minutes"] = np.where(eligible, np.maximum(0, threshold - gap_pos), 0)

total_gap_minutes = float(gap_pos.sum())
total_blocked_minutes = float(work["blocked_minutes"].sum())

revenue_at_risk_pct = (100 * total_blocked_minutes / total_gap_minutes) if total_gap_minutes > 0 else 0.0
blocked_days = total_blocked_minutes / 1440
estimated_revenue_loss_eur = blocked_days * MEDIAN_PRICE

total_rentals = len(work)
affected = int(work["affected_by_policy"].sum())
problematic = int(work["problematic"].sum())
solved = int(work["solved_by_policy"].sum())

pct = lambda a, b: (100 * a / b) if b else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Locations terminées (périmètre)", f"{total_rentals:,}")
col2.metric("Locations affectées par la politique", f"{affected:,}", f"{pct(affected, total_rentals):.1f}%")
col3.metric("Cas problématiques (attente > 0)", f"{problematic:,}", f"{pct(problematic, total_rentals):.1f}%")
col4.metric(
    "Cas problématiques résolus",
    f"{solved:,}",
    f"{pct(solved, problematic):.1f}% des cas problématiques" if problematic else "0%"
)
col5.metric("Revenu à risque (estimation)", f"{revenue_at_risk_pct:.1f}%", f"≈ €{estimated_revenue_loss_eur:,.0f} est.")

st.caption(
    f"Estimation en € basée sur le prix journalier médian "
    f"(médiane = €{MEDIAN_PRICE:.0f}, moyenne = €{MEAN_PRICE:.0f}). "
    "Le revenu à risque est calculé à partir du temps de créneau bloqué "
    "entre deux locations consécutives. "
)
st.caption(
    "ℹ️ Ce calcul est basé sur les 1 841 locations consécutives du même véhicule "
    "(seuls cas où le buffer a un effet réel sur la disponibilité). "
    "Les locations sans précédent immédiat ne sont pas concernées par la politique de buffer."
)

st.subheader("Distribution des retards au check-out (minutes)")

delays = df["delay_at_checkout_in_minutes"].dropna().astype(float)

if clip_mode == "Percentiles (1–99)":
    lo, hi = delays.quantile([0.01, 0.99])
    delays_plot = delays.clip(lo, hi)
    st.caption(f"Écrêté aux percentiles 1–99 : [{lo:.0f} min, {hi:.0f} min]")
elif clip_mode == "Plage fixe (±24h)":
    lo, hi = -1440, 1440
    delays_plot = delays.clip(lo, hi)
    st.caption("Écrêté à ±24 heures : [-1440 min, 1440 min]")
else:
    delays_plot = delays
    st.caption("Aucun écrêtage (valeurs brutes)")

hist_df = pd.DataFrame({"delay_min": delays_plot})

chart = (
    alt.Chart(hist_df)
    .mark_bar()
    .encode(
        x=alt.X("delay_min:Q", bin=alt.Bin(maxbins=bins), title="Retard au check-out (minutes)"),
        y=alt.Y("count():Q", title="Nombre de locations"),
    )
    .properties(height=280)
)

st.altair_chart(chart, use_container_width=True)

st.divider()

st.subheader("Sensibilité au seuil (courbe rapide)")

thresholds = np.arange(0, 721, 15)

def compute_curve(th):
    affected_mask = (work["gap_min"] < th)
    solved_mask = work["problematic"] & affected_mask

    eligible = np.isfinite(work["gap_min"])
    gap_pos = work["gap_min"].where(eligible, 0).clip(lower=0)
    blocked = np.where(eligible, np.maximum(0, th - gap_pos), 0)

    total_gap = float(gap_pos.sum())
    total_blocked = float(blocked.sum())
    revenue_risk_share = (total_blocked / total_gap) if total_gap > 0 else 0.0

    return float(affected_mask.mean()), int(solved_mask.sum()), float(revenue_risk_share)

affected_share = []
solved_counts = []
revenue_risk_share = []

for th in thresholds:
    a, s, r = compute_curve(th)
    affected_share.append(a)
    solved_counts.append(s)
    revenue_risk_share.append(r)

curve_df = pd.DataFrame({
    "threshold_min": thresholds,
    "affected_share": affected_share,
    "solved_problematic_cases": solved_counts,
    "revenue_at_risk_share": revenue_risk_share,
})
curve_df["revenue_at_risk_pct"] = 100 * curve_df["revenue_at_risk_share"]
curve_df["affected_pct"] = 100 * curve_df["affected_share"]

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.caption("Part des locations affectées (masquées dans la recherche)")
    affected_chart = (
        alt.Chart(curve_df)
        .mark_line(color="#4C78A8")
        .encode(
            x=alt.X("threshold_min:Q", title="Seuil (minutes)"),
            y=alt.Y("affected_pct:Q", title="Locations affectées (%)"),
        )
        .properties(height=260)
    )
    st.altair_chart(affected_chart, use_container_width=True)

with c2:
    st.caption("Nombre de cas problématiques résolus")
    solved_chart = (
        alt.Chart(curve_df)
        .mark_line(color="#F58518")
        .encode(
            x=alt.X("threshold_min:Q", title="Seuil (minutes)"),
            y=alt.Y("solved_problematic_cases:Q", title="Cas résolus (nombre)"),
        )
        .properties(height=260)
    )
    st.altair_chart(solved_chart, use_container_width=True)

with c3:
    st.caption("Revenu à risque (estimation) — part du créneau bloqué")
    revenue_chart = (
        alt.Chart(curve_df)
        .mark_line(color="#E45756")
        .encode(
            x=alt.X("threshold_min:Q", title="Seuil (minutes)"),
            y=alt.Y("revenue_at_risk_pct:Q", title="Revenu à risque (%)"),
        )
        .properties(height=260)
    )
    st.altair_chart(revenue_chart, use_container_width=True)

st.subheader("Vue en coude : frictions résolues vs revenu à risque")
scatter = (
    alt.Chart(curve_df)
    .mark_circle(size=70)
    .encode(
        x=alt.X("revenue_at_risk_pct:Q", title="Revenu à risque (%)"),
        y=alt.Y("solved_problematic_cases:Q", title="Cas problématiques résolus (nombre)"),
        tooltip=[
            alt.Tooltip("threshold_min:Q", title="Seuil (min)"),
            alt.Tooltip("revenue_at_risk_pct:Q", title="Revenu à risque (%)", format=".2f"),
            alt.Tooltip("solved_problematic_cases:Q", title="Cas résolus"),
            alt.Tooltip("affected_pct:Q", title="Locations affectées (%)", format=".1f"),
        ],
    )
    .properties(height=320)
)
st.altair_chart(scatter, use_container_width=True)

st.divider()


st.subheader("Facteurs influençant le prix journalier")

@st.cache_data
def compute_feature_importance():
    df = pricing_df.copy()
    df = df.dropna()
    
    target = "rental_price_per_day"
    X = df.drop(columns=[target])
    y = df[target]
    
    cat_cols = ["model_key", "fuel", "paint_color", "car_type"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X, y)
    
    ohe = pipe["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_names = cat_names + num_cols
    
    feat_df = pd.DataFrame({
        "feature": all_names,
        "importance": pipe["model"].feature_importances_
    }).sort_values("importance", ascending=False).head(12)
    
    return feat_df

feat_df = compute_feature_importance()

chart = (
    alt.Chart(feat_df)
    .mark_bar()
    .encode(
        x=alt.X("importance:Q", title="Importance (score)"),
        y=alt.Y("feature:N", sort="-x", title="Caractéristique"),
        color=alt.Color("importance:Q", scale=alt.Scale(scheme="blues"), legend=None),
    )
    .properties(height=380)
)

feat_df["feature"] = feat_df["feature"].replace({
    "engine_power": "Puissance moteur",
    "mileage": "Kilométrage",
    "has_gps": "GPS",
    "has_getaround_connect": "Connect (sans clé)",
    "automatic_car": "Boîte automatique",
    "private_parking_available": "Parking privé",
    "has_air_conditioning": "Climatisation",
    "has_speed_regulator": "Régulateur de vitesse",
    "winter_tires": "Pneus hiver",
    "car_type_suv": "Type: SUV",
    "car_type_estate": "Type: Break",
    "car_type_sedan": "Type: Berline",
    "car_type_hatchback": "Type: Citadine",
    "model_key_BMW": "Marque: BMW",
    "model_key_Citroën": "Marque: Citroën",
    "model_key_Mitsubishi": "Marque: Mitsubishi",
    "paint_color_blue": "Couleur: Bleu",
    "paint_color_black": "Couleur: Noir",
    "paint_color_white": "Couleur: Blanc",
})

st.altair_chart(chart, use_container_width=True)
st.caption("Importance calculée via Random Forest sur le jeu de données tarifaire complet.")