# 🚗 Getaround – Analyse des Retards & API de Pricing

## 📌 Présentation du projet

Ce projet combine une analyse métier des retards de location et le déploiement d’un modèle de Machine Learning via une API FastAPI. Il reproduit une architecture proche d’un environnement de production avec validation stricte des entrées, contrôle explicite du schéma des features et déploiement conteneurisé via Docker.

Le projet se compose de deux parties complémentaires : un dashboard interactif d’analyse des retards et une API de prédiction du prix journalier des véhicules. L’objectif est double : analyser l’impact des retards sur le fonctionnement opérationnel et automatiser la stratégie de pricing à l’aide d’un modèle supervisé.

---

## 📊 Dashboard – Analyse des Retards

Dashboard développé avec Streamlit.

Version en ligne :  
https://huggingface.co/spaces/Anabeldg/getaround-delay-dashboard

### 🎯 Objectif métier

L’introduction d’un buffer minimum entre deux locations permet de réduire les conflits liés aux retards et les frictions entre utilisateurs. Cependant, augmenter ce buffer peut également diminuer le taux d’utilisation des véhicules. Le dashboard permet d’explorer la distribution des retards, de simuler différents seuils (60 / 120 / 180 minutes) et d’analyser le trade-off entre fiabilité opérationnelle et volume d’activité.

Technologies utilisées : pandas, Altair, Streamlit.

---

## 🚀 API de Prédiction des Prix

API développée avec FastAPI et déployée sur Hugging Face Spaces (Docker).

API en ligne :  
https://huggingface.co/spaces/Anabeldg/getaround-pricing-api

---

## 📡 Documentation de l’API

La documentation complète de l’API est disponible ici :

https://anabeldg-getaround-pricing-api.hf.space/docs


---

## 📡 Endpoints disponibles

### GET /

Page d’accueil HTML listant les endpoints disponibles.

### GET /health

Endpoint de vérification de l’état du service.

Réponse :

```json
{
  "status": "ok"
}
```

Permet de vérifier que le service est actif et que le modèle est correctement chargé.

### POST /predict

Prédit le rental_price_per_day. L’API accepte deux formats d’entrée.

#### Format principal (requis pour l’évaluation)

Format matriciel :

```json
{
  "input": [
    ["Citroën", 100000, 110, "diesel", "black", "sedan", true, true, true, false, true, true, false]
  ]
}
```

Contraintes :
- Chaque ligne doit contenir exactement 13 features.
- L’ordre doit correspondre strictement à FEATURE_ORDER.
- Toute erreur de dimension entraîne une réponse HTTP 422.

#### 📬 Exemple avec curl

curl -X POST "https://anabeldg-getaround-pricing-api.hf.space/predict" \
-H "Content-Type: application/json" \
-d '{"input":[["Citroën",140411,100,"diesel","black","convertible",1,1,0,0,1,1,1]]}'


#### 🐍 Exemple avec Python

```import requests

payload = {
  "input": [
    ["Citroën", 140411, 100, "diesel", "black", "convertible", 1, 1, 0, 0, 1, 1, 1]
  ]
}

response = requests.post(
    "https://anabeldg-getaround-pricing-api.hf.space/predict",
    json=payload
)

print(response.json())
```

#### 📤 Format de réponse

```{
  "prediction": [97.15]
}
```

Une valeur est retournée pour chaque ligne fournie dans "input".

#### Format secondaire (compatibilité)

Format dictionnaire :

```json
{
  "input": [
    {
      "model_key": "Citroën",
      "mileage": 100000,
      "engine_power": 110,
      "fuel": "diesel",
      "paint_color": "black",
      "car_type": "sedan",
      "private_parking_available": true,
      "has_gps": true,
      "has_air_conditioning": true,
      "automatic_car": false,
      "has_getaround_connect": true,
      "has_speed_regulator": true,
      "winter_tires": false
    }
  ]
}
```

Contraintes :
- Toutes les colonnes requises doivent être présentes.
- Les colonnes sont automatiquement réordonnées selon FEATURE_ORDER.

#### Réponse de l’API

```json
{
  "prediction": [62.4]
}
```

Une valeur prédite par ligne d’entrée (type float).

---

## 🧠 Modèle de Machine Learning

Algorithme : Random Forest Regressor  
Prétraitement : ColumnTransformer + OneHotEncoder  
Sérialisation : joblib  
Chargement du modèle au démarrage du service afin de minimiser la latence.

---

## 🏗 Structure du projet

```
getaround-project/
│
├── dashboard/
│   ├── app.py
│   ├── get_around_delay_analysis.csv
│   ├── requirements.txt
│   └── Dockerfile
│
├── pricing_api/
│   ├── main.py
│   ├── pricing_model.joblib
│   ├── feature_order.json
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

---

## 💻 Lancement en local

### Lancer l’API

```bash
cd pricing_api
pip install -r requirements.txt
uvicorn main:app --reload
```

API disponible sur :  
http://127.0.0.1:8000

Documentation personnalisée :  
http://127.0.0.1:8000/docs

### Lancer le dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🐳 Déploiement

Les deux composants sont déployés via Docker sur Hugging Face Spaces. Cette architecture garantit la reproductibilité, l’isolation des dépendances, la cohérence entre environnement local et production, ainsi qu’une séparation claire entre analyse et service ML.

---

## 📈 Compétences mobilisées

Analyse de données, modélisation Machine Learning, validation de schéma, conception d’API REST, conteneurisation Docker et déploiement cloud.

## 👤 Auteur

**Anastasiia Belosludtseva**  
Projet Machine Learning & Data Science
