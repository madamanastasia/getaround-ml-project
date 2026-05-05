import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field



APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "pricing_model.joblib"
FEATURE_ORDER_PATH = APP_DIR / "feature_order.json"
DOCS_HTML_PATH = APP_DIR / "docs.html"


def _load_model(path: Path):
    if not path.exists():
        raise RuntimeError(
            f"Model artifact not found: {path}. "
            f"Make sure pricing_model.joblib is committed to the Space repository."
        )
    return joblib.load(path)


def _load_feature_order(path: Path) -> List[str]:
    if not path.exists():
        raise RuntimeError(
            f"feature_order.json not found: {path}. "
            f"Commit feature_order.json next to main.py in the Space repository."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"feature_order.json is not valid JSON: {e}") from e

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise RuntimeError("feature_order.json must be a JSON list of strings.")

    return data


model = _load_model(MODEL_PATH)
FEATURE_ORDER: List[str] = _load_feature_order(FEATURE_ORDER_PATH)
N_FEATURES = len(FEATURE_ORDER)



app = FastAPI(
    title="Getaround Pricing API",
    version="1.0.0",
    docs_url=None,   
    redoc_url=None,
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>🚗 Getaround Pricing API</h1>
    <p>Endpoints:</p>
    <ul>
      <li><a href="/health">GET /health</a></li>
      <li><a href="/predict">POST /predict</a></li>
      <li><a href="/docs">GET /docs</a> - API Documentation</li>
    </ul>
    """



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/docs", response_class=HTMLResponse)
def docs():
    if DOCS_HTML_PATH.exists():
        return DOCS_HTML_PATH.read_text(encoding="utf-8")

    return """
    <h1>Documentation</h1>
    <p>Custom API documentation file not found.</p>
    """




Scalar = Union[str, int, float, bool]


class PredictMatrix(BaseModel):
    """
    Required by evaluation:
    {"input": [[...], [...]]}
    """
    input: List[List[Scalar]] = Field(
        ...,
        description=f"Matrix of rows. Each row must have exactly {N_FEATURES} values in FEATURE_ORDER.",
    )


class PredictDicts(BaseModel):
    """
    Optional/backward compatible:
    {"input": [{"feature": value, ...}, ...]}
    """
    input: List[Dict[str, Scalar]] = Field(
        ...,
        description="List of dicts with feature names as keys (backward compatible).",
    )


PredictPayload = Union[PredictMatrix, PredictDicts]


def _matrix_to_df(rows: List[List[Scalar]]) -> pd.DataFrame:
    
    bad = [i for i, row in enumerate(rows) if len(row) != N_FEATURES]
    if bad:
        example_bad = {i: len(rows[i]) for i in bad[:10]}
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid input shape.",
                "expected_n_features": N_FEATURES,
                "bad_rows_index_to_length": example_bad,
                "expected_feature_order": FEATURE_ORDER,
            },
        )
    return pd.DataFrame(rows, columns=FEATURE_ORDER)


def _dicts_to_df(items: List[Dict[str, Scalar]]) -> pd.DataFrame:
    X = pd.DataFrame(items)
    missing = [c for c in FEATURE_ORDER if c not in X.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required columns in dict payload.",
                "missing_columns": missing,
                "expected_feature_order": FEATURE_ORDER,
            },
        )
    return X[FEATURE_ORDER]


def _payload_to_df(payload: PredictPayload) -> pd.DataFrame:
    rows = payload.input
    if not rows:
     
        return pd.DataFrame(columns=FEATURE_ORDER)

    first = rows[0]
    if isinstance(first, list):
       
        return _matrix_to_df(rows)  

    
    return _dicts_to_df(rows)  



@app.post("/predict")
def predict(payload: PredictPayload):
    X = _payload_to_df(payload)

    try:
        preds = model.predict(X)
    except Exception as e:
        
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {type(e).__name__}: {e}",
        ) from e

    return {"prediction": preds.tolist()}
