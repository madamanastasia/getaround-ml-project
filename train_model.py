import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "get_around_pricing_project.csv"
OUT_PATH = APP_DIR / "pricing_model.joblib"

TARGET = "rental_price_per_day"

def main():
    df = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"])
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(pipe, OUT_PATH)
    print(f"Saved model to: {OUT_PATH}")
    print(f"MAE: {mae:.2f}")
    print(f"R2:  {r2:.3f}")

if __name__ == "__main__":
    main()
