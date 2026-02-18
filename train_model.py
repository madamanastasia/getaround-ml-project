import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "get_around_pricing_project.csv"


OUT_MODEL_PATH = APP_DIR / "pricing_model.joblib"
OUT_FEATURE_ORDER_PATH = APP_DIR / "feature_order.json"

TARGET = "rental_price_per_day"


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Robustly drop any accidental index columns from CSV exports
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = _clean_dataframe(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

  
    feature_order = list(X.columns)

   
    cat_cols = ["model_key", "fuel", "paint_color", "car_type"]
    missing_cats = [c for c in cat_cols if c not in X.columns]
    if missing_cats:
        raise ValueError(f"Missing expected categorical columns: {missing_cats}")

    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    mlflow.set_experiment("getaround-pricing")
    with mlflow.start_run(run_name="rf_baseline"):
        
        pipe.fit(X_train, y_train)

        # Evaluate
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "test_size": 0.2,
                "random_state": 42,
                "categorical_cols": ",".join(cat_cols),
                "n_features_raw": X.shape[1],
            }
        )
        mlflow.log_metrics({"mae": float(mae), "r2": float(r2)})

        
        joblib.dump(pipe, OUT_MODEL_PATH)
        OUT_FEATURE_ORDER_PATH.write_text(
            json.dumps(feature_order, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(OUT_MODEL_PATH))
        mlflow.log_artifact(str(OUT_FEATURE_ORDER_PATH))

        
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"Saved model to: {OUT_MODEL_PATH}")
        print(f"Saved feature order to: {OUT_FEATURE_ORDER_PATH}")
        print(f"MAE: {mae:.2f}")
        print(f"R2:  {r2:.3f}")


if __name__ == "__main__":
    main()
