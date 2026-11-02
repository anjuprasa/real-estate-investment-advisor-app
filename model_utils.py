"""Model code shared by the Streamlit app (no Streamlit imports, so it is easy to test)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_YEAR = 2026
FEATURES = ["Area_sqft", "Bedrooms", "Bathrooms", "Floors", "Property_Age", "Location_Score",
            "School_Score", "Current_Price", "Price_per_Sqft", "Infra_Score", "Total_Rooms",
            "Area_per_Bedroom"]


def build_features(area, bedrooms, bathrooms, floors, year_built, location_score, school_score, price):
    """Turn the form inputs into the exact feature row the models were trained on."""
    return pd.DataFrame([{
        "Area_sqft": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Floors": floors,
        "Property_Age": CURRENT_YEAR - year_built, "Location_Score": location_score,
        "School_Score": school_score, "Current_Price": price,
        "Price_per_Sqft": price / area, "Infra_Score": (location_score + school_score) / 2,
        "Total_Rooms": bedrooms + bathrooms, "Area_per_Bedroom": area / bedrooms,
    }])[FEATURES]


def train_models(df, seed=42):
    X = df[FEATURES]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, df["Good_Investment"], df["Price_After_5_Years"],
        test_size=0.2, random_state=seed, stratify=df["Good_Investment"])

    clf = Pipeline([("sc", StandardScaler()), ("m", GradientBoostingClassifier(random_state=seed))])
    reg = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    clf.fit(X_tr, yc_tr)
    reg.fit(X_tr, yr_tr)

    pc, pr = clf.predict(X_te), reg.predict(X_te)
    metrics = {
        "clf": {"Accuracy": accuracy_score(yc_te, pc), "Precision": precision_score(yc_te, pc),
                "Recall": recall_score(yc_te, pc), "F1": f1_score(yc_te, pc)},
        "reg": {"MAE": mean_absolute_error(yr_te, pr),
                "RMSE": float(np.sqrt(mean_squared_error(yr_te, pr))), "R2": r2_score(yr_te, pr)},
    }
    # Final models use all data once evaluation is done
    clf.fit(X, df["Good_Investment"])
    reg.fit(X, df["Price_After_5_Years"])
    importance = pd.Series(clf.named_steps["m"].feature_importances_, index=FEATURES).sort_values()
    return clf, reg, metrics, importance
