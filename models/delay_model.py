import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/delivery_logistics_clean.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

# IMPORTANT:
# We are NOT using:
# delayed
# delivery_status
# expected_time_hours
# delayed_binary
# time_difference
#
# These variables directly reveal whether a delivery
# was delayed and would cause data leakage.

features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "delivery_rating",
    "delivery_cost"
]

target = "delayed_binary"


X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Identify categorical features
# --------------------------------------------------

categorical_features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition"
]


# --------------------------------------------------
# 4. Identify numerical features
# --------------------------------------------------

numerical_features = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "delivery_rating",
    "delivery_cost"
]


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 6. Create Random Forest classifier
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# --------------------------------------------------
# 7. Create pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

print("\nTraining Random Forest delay model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 10. Predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 11. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n----- DELAY MODEL PERFORMANCE -----")

print("Accuracy:", round(accuracy, 4))
print("ROC-AUC :", round(auc, 4))


print("\n----- CLASSIFICATION REPORT -----")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["On Time", "Delayed"]
    )
)


print("\n----- CONFUSION MATRIX -----")

print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 12. Save model
# --------------------------------------------------

joblib.dump(
    pipeline,
    "models/delay_model.pkl"
)

print("\nDelay model saved successfully!")
print("File: models/delay_model.pkl")