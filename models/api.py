from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from models.recommendation import (
    recommend_delivery,
    find_collaborative_deliveries,
    create_delivery_groups,
    calculate_delivery_savings,
    calculate_compatibility,
    optimize_delivery_route,
    calculate_route_summary,
    orders_df,
)


# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="SyncDrop API",
    description="AI-Based Collaborative Last-Mile Delivery API",
    version="1.0.0",
)


# ==========================================================
# REQUEST MODELS
# ==========================================================

class RecommendationRequest(BaseModel):
    package_type: str
    region: str
    weather_condition: str
    distance_km: float
    package_weight_kg: float
    priority: str = "balanced"


# ==========================================================
# BASIC HEALTH CHECK
# ==========================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "app": "SyncDrop",
        "message": "SyncDrop backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==========================================================
# ORDERS
# ==========================================================

@app.get("/orders")
def get_orders():
    try:
        orders = orders_df.fillna("").to_dict(orient="records")

        return {
            "success": True,
            "count": len(orders),
            "orders": orders
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# SINGLE ORDER
# ==========================================================

@app.get("/orders/{order_id}")
def get_order(order_id: str):

    try:
        result = orders_df[
            orders_df["order_id"].astype(str) == str(order_id)
        ]

        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Order {order_id} not found"
            )

        order = result.iloc[0].fillna("").to_dict()

        return {
            "success": True,
            "order": order
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# AI DELIVERY PARTNER RECOMMENDATION
# ==========================================================

@app.post("/recommend")
def get_recommendation(request: RecommendationRequest):

    try:

        result = recommend_delivery(
            package_type=request.package_type,
            region=request.region,
            weather_condition=request.weather_condition,
            distance_km=request.distance_km,
            package_weight_kg=request.package_weight_kg,
            priority=request.priority,
        )

        # Convert DataFrame results to JSON if necessary
        if hasattr(result, "to_dict"):
            result = result.to_dict(orient="records")

        return {
            "success": True,
            "recommendation": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )