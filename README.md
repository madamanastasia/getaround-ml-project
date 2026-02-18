# 🚗 Getaround – Projet d’Analyse des Retards & API de Pricing

## 📌 Présentation du projet

Ce projet combine analyse métier et déploiement d’un modèle de machine learning dans une architecture proche d’un environnement de production.

Il se compose de deux parties principales :

- 📊 **Dashboard interactif** – analyse des retards de location et de leur impact business  
- 🚀 **API de prédiction de prix** – estimation du prix journalier d’un véhicule à partir de ses caractéristiques  

L’objectif est d’aider à la prise de décision opérationnelle et d’automatiser la stratégie de pricing via un modèle déployé.

---

## 📊 Dashboard en ligne

Dashboard interactif développé avec **Streamlit**.

🔗 **Version en ligne :**  
https://huggingface.co/spaces/Anabeldg/getaround-delay-dashboard

### ✨ Fonctionnalités

- Exploration des distributions de retards  
- Analyse de différents seuils de retard (60 / 120 / 180 minutes)  
- Interprétation de l’impact business  
- Visualisations interactives avec **pandas** et **Altair**

---

## 🚀 API de prédiction des prix

API développée avec **FastAPI** et déployée sur **Hugging Face Spaces**.

🔗 **API en ligne :**  
https://huggingface.co/spaces/Anabeldg/getaround-pricing-api

### 📡 Endpoints disponibles

#### `GET /`
Page d’accueil présentant l’API.

#### `GET /health`
Endpoint de vérification de l’état du service.

Exemple de réponse :

```json
{
  "status": "ok"
}
```

#### `POST /predict`
Prédiction du prix journalier d’un véhicule.

### 📥 Exemple de requête

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

### 📤 Exemple de réponse

```json
{
  "prediction": [62.4]
}
```

---

## 🧠 Modèle

- Algorithme : **Random Forest Regressor**  
- Prétraitement : **ColumnTransformer + OneHotEncoder**  
- Sauvegarde du modèle : `joblib`  
- Version de `scikit-learn` fixée pour assurer la reproductibilité  

---

## 🏗 Structure du projet

```text
getaround-project/
│
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── pricing_api/
│   ├── main.py
│   ├── pricing_model.joblib
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

---

## 💻 Lancement en local

### 1️⃣ Cloner le repository

```bash
git clone https://github.com/madamanastasia/getaround-project.git
cd getaround-project
```

### 2️⃣ Lancer l’API localement

```bash
cd pricing_api
pip install -r requirements.txt
uvicorn main:app --reload
```

API accessible à :  
http://127.0.0.1:8000

### 3️⃣ Lancer le dashboard localement

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Dashboard accessible à :  
http://localhost:8501

---

## ⚙️ Déploiement

Les deux composants sont déployés via **Docker sur Hugging Face Spaces** :

- 📊 Dashboard → Space Docker Streamlit  
- 🚀 API → Space Docker FastAPI + Uvicorn  

Cette architecture garantit :

- Reproductibilité  
- Gestion des dépendances  
- Isolation de l’environnement  
- Approche proche d’un contexte de production  

---

## 📈 Contexte métier

Le projet répond à deux problématiques principales :

1. Déterminer un seuil optimal de retard pour minimiser les conflits opérationnels.  
2. Automatiser la stratégie de pricing à l’aide d’un modèle de machine learning.  

Compétences mobilisées :

- Analyse de données  
- Modélisation machine learning  
- Conception d’API  
- Conteneurisation (Docker)  
- Déploiement cloud  

---

## 👤 Auteur

**Anastasiia Belosludtseva**  
Projet Machine Learning & Data Science
