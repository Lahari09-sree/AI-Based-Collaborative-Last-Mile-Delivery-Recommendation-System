import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv("data/delivery_ml.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# 2. Define input features and target
X = df.drop("delivery_cost", axis=1)
y = df["delivery_cost"]


# 3. Categorical columns
categorical_features = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition"
]


# 4. Numerical columns
numerical_features = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "delivery_rating"
]


# 5. Preprocessing
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


# 6. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# 7. Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 8. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 9. Train the model
print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# 10. Make predictions
y_pred = pipeline.predict(X_test)


# 11. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n----- MODEL PERFORMANCE -----")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# 12. Save the trained model
joblib.dump(
    pipeline,
    "models/cost_model.pkl"
)

print("\nModel saved successfully!")
print("File: models/cost_model.pkl")