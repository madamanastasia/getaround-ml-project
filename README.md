# Getaround – Pricing & Delay Analysis Project

## Project Overview

This project combines business analysis and machine learning deployment into a production-ready architecture.

It consists of two main components:

- Interactive Dashboard – analysis of rental delays and business impact.
- Machine Learning Pricing API – prediction of daily rental price based on car characteristics.

The objective is to support operational decision-making and automate pricing using a deployed ML model.

---

## Online Dashboard

Interactive dashboard built with Streamlit.

Live version:  
https://huggingface.co/spaces/Anabeldg/getaround-delay-dashboard

Features:

- Exploration of rental delay distributions  
- Analysis of delay thresholds (60 / 120 / 180 minutes)  
- Business impact reasoning  
- Interactive visualizations using pandas and Altair  

---

## Pricing Prediction API

Production-ready API built with FastAPI and deployed on Hugging Face Spaces.

Live API:  
https://huggingface.co/spaces/Anabeldg/getaround-pricing-api

### Available Endpoints

**GET /**  
Landing page with project information.

**GET /health**  
Healthcheck endpoint.

Example response:

{
  "status": "ok"
}

**POST /predict**  
Predict daily rental price.

Example request:

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

Example response:

{
  "prediction": [62.4]
}

---

## Model

- Algorithm: Random Forest Regressor  
- Preprocessing: ColumnTransformer + OneHotEncoder  
- Model persistence: joblib  
- scikit-learn version pinned for reproducibility  

---

## Project Structure

getaround-project/

dashboard/
- app.py
- requirements.txt
- Dockerfile

pricing_api/
- main.py
- pricing_model.joblib
- requirements.txt
- Dockerfile

README.md

---

## Local Setup

### Clone repository

git clone https://github.com/YOUR_USERNAME/getaround-project.git  
cd getaround-project

### Run the API locally

cd pricing_api  
pip install -r requirements.txt  
uvicorn main:app --reload  

API will be available at:  
http://127.0.0.1:8000  

### Run the Dashboard locally

cd dashboard  
pip install -r requirements.txt  
streamlit run app.py  

Dashboard will be available at:  
http://localhost:8501  

---

## Deployment

Both components are deployed using Docker on Hugging Face Spaces:

- Dashboard → Streamlit Docker Space  
- API → FastAPI + Uvicorn Docker Space  

This setup ensures:

- Reproducibility  
- Version control  
- Dependency isolation  
- Production-like environment  

---

## Business Context

The project addresses two key business questions:

1. What is the optimal delay threshold to minimize operational conflicts?
2. How can pricing be automated using a machine learning model?

The final architecture demonstrates:

- Data analysis  
- Model training  
- API design  
- Containerization  
- Cloud deployment  

---

## Author

Anabel DG  
Machine Learning & Data Science Project
