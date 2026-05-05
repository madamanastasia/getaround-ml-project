# 🚗 Getaround – Analyse des Retards & API de Pricing

## 📌 À propos du projet

Ce projet combine une analyse métier des retards de location et le déploiement d'un modèle de Machine Learning via une API FastAPI. Il reproduit une architecture proche d'un environnement de production avec validation stricte des entrées et déploiement conteneurisé via Docker.

Le projet se compose de **trois parties complémentaires** :
- Un **dashboard interactif** d'analyse des retards
- Une **API de prédiction** du prix journalier des véhicules
- Une **application web interactive** de pricing connectée à l'API

---

## 📊 Dashboard – Analyse des Retards

🔗 [Voir le dashboard en ligne](https://huggingface.co/spaces/Anabeldg/getaround-delay-dashboard)

Développé avec **Streamlit**, ce dashboard permet d'explorer la distribution des retards, de simuler différents seuils de buffer (60 / 120 / 180 minutes) et d'analyser le compromis entre fiabilité opérationnelle et revenus.

**Résultats clés :**
- 57,5% des locations sont rendues en retard — médiane : 53 minutes
- Seuil optimal : **120 minutes sur les véhicules Connect**
- 92% des conflits résolus pour seulement €1,830 de revenu à risque

**Technologies :** pandas · Altair · Streamlit

---

## 🚀 API de Prédiction des Prix

🔗 [API en ligne](https://huggingface.co/spaces/Anabeldg/getaround-pricing-api)  
🔗 [Documentation](https://anabeldg-getaround-pricing-api.hf.space/docs)

Développée avec **FastAPI**, déployée via **Docker** sur Hugging Face Spaces.

### Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Page d'accueil |
| GET | `/health` | Statut du service |
| POST | `/predict` | Prédiction du prix journalier |
| GET | `/docs` | Documentation HTML |

### Format d'entrée — matriciel (requis)

```json
{
  "input": [
    ["Citroën", 85000, 82, "petrol", "white", "hatchback", 1, 0, 1, 0, 0, 1, 0]
  ]
}
```

### Exemple curl

```bash
curl -X POST "https://anabeldg-getaround-pricing-api.hf.space/predict" \
-H "Content-Type: application/json" \
-d '{"input":[["Citroën",85000,82,"petrol","white","hatchback",1,0,1,0,0,1,0]]}'
```

### Exemple Python

```python
import requests

payload = {
    "input": [
        ["Citroën", 85000, 82, "petrol", "white", "hatchback", 1, 0, 1, 0, 0, 1, 0]
    ]
}

response = requests.post(
    "https://anabeldg-getaround-pricing-api.hf.space/predict",
    json=payload
)
print(response.json())
# {"prediction": [96.4]}
```

---

## 🖥 Application Web – Estimation du Prix

🔗 [Application en ligne](https://huggingface.co/spaces/Anabeldg/getaround-pricing-app)

Interface interactive développée avec **Streamlit**. L'utilisateur saisit les caractéristiques d'un véhicule et obtient une estimation du prix journalier via l'API FastAPI.

**Technologies :** Streamlit · requests

---

## 🧠 Modèle de Machine Learning

| Paramètre | Valeur |
|---|---|
| Algorithme | Random Forest Regressor |
| Estimateurs | 400 |
| R² | 0.73 |
| MAE | 10.8 €/jour |
| Prétraitement | ColumnTransformer + OneHotEncoder |
| Sérialisation | joblib |
| Suivi expériences | MLflow |

**Facteurs principaux influençant le prix :** puissance moteur · kilométrage · GPS · boîtier Connect

---

## 💻 Lancement en local

### API FastAPI
```bash
cd pricing_api
pip install -r requirements.txt
uvicorn main:app --reload
# http://127.0.0.1:8000
```

### Dashboard Streamlit
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Application Pricing
```bash
cd pricing_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🐳 Déploiement

Les trois composants sont déployés sur **Hugging Face Spaces**. L'API FastAPI utilise **Docker** pour garantir la reproductibilité et l'isolation des dépendances. L'application de pricing appelle directement l'API via `requests.post()`.

---

## 📈 Compétences mobilisées

`Machine Learning` · `FastAPI` · `Docker` · `MLflow` · `Streamlit` · `API REST` · `Data Analysis` · `scikit-learn`

---

## 👤 Auteur

**Anastasiia Belosludtseva**  
Certification CDSD – Bloc 5  
🔗 [Hugging Face](https://huggingface.co/Anabeldg) · [GitHub](https://github.com/madamanastasia)
