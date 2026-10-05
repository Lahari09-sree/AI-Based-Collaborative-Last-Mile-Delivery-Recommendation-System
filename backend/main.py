import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

ORDERS_FILE = DATA_DIR / "orders.csv"


# ============================================================
# PROJECT ROOT
# ============================================================

os.chdir(BASE_DIR)

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))


# ============================================================
# IMPORT REAL ML / RECOMMENDATION SYSTEM
# ============================================================

from recommendation import (
    recommend_delivery,
    calculate_compatibility,
    find_collaborative_deliveries,
    create_delivery_groups,
    calculate_delivery_savings,
    optimize_delivery_route,
    calculate_route_summary,
    orders_df,
    df as ml_df,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Last-Mile Delivery API",
    description="Backend API for AI-based collaborative last-mile delivery",
    version="1.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class RecommendationRequest(BaseModel):
    package_type: str
    region: str
    weather_condition: str
    distance_km: float
    package_weight_kg: float
    priority: str = "balanced"


class OrderRecommendationRequest(BaseModel):
    package_type: str
    region: str
    weather_condition: str
    priority: str = "balanced"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def dataframe_to_records(dataframe):
    """
    Convert DataFrame into JSON-friendly records.
    """

    if dataframe is None or len(dataframe) == 0:
        return []

    result = dataframe.copy()

    result = result.where(pd.notnull(result), None)

    return result.to_dict(orient="records")


def get_order(order_id: str):
    """
    Find an order by order ID.
    """

    matches = orders_df[
        orders_df["order_id"].astype(str) == str(order_id)
    ]

    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Order {order_id} not found",
        )

    return matches.iloc[0]


def order_to_dict(order):
    """
    Convert pandas Series to JSON-safe dictionary.
    """

    return {
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in order.to_dict().items()
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Last-Mile Delivery API is running",
        "status": "success",
        "version": "1.1.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Backend and ML recommendation system are running",
        "orders_loaded": len(orders_df),
        "ml_records_loaded": len(ml_df),
    }


# ============================================================
# GET ALL ORDERS
# ============================================================

@app.get("/orders")
def get_orders():

    return {
        "success": True,
        "count": len(orders_df),
        "orders": dataframe_to_records(orders_df),
    }


# ============================================================
# GET SINGLE ORDER
# ============================================================

@app.get("/orders/{order_id}")
def get_single_order(order_id: str):

    order = get_order(order_id)

    return {
        "success": True,
        "order": order_to_dict(order),
    }


# ============================================================
# ML INPUT OPTIONS
# ============================================================

@app.get("/recommend/options")
def recommendation_options():

    package_types = sorted(
        ml_df["package_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    regions = sorted(
        ml_df["region"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    weather_conditions = sorted(
        ml_df["weather_condition"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return {
        "success": True,
        "package_types": package_types,
        "regions": regions,
        "weather_conditions": weather_conditions,
        "priorities": [
            "balanced",
            "fast",
            "economy",
        ],
    }


# ============================================================
# GENERAL ML RECOMMENDATION
# ============================================================

@app.post("/recommend")
def get_recommendation(request: RecommendationRequest):

    try:

        if request.priority not in [
            "balanced",
            "fast",
            "economy",
        ]:
            raise HTTPException(
                status_code=400,
                detail="Priority must be balanced, fast, or economy",
            )

        recommendations = recommend_delivery(
            package_type=request.package_type,
            region=request.region,
            weather_condition=request.weather_condition,
            distance_km=request.distance_km,
            package_weight_kg=request.package_weight_kg,
            priority=request.priority,
        )

        records = dataframe_to_records(recommendations)

        return {
            "success": True,
            "priority": request.priority,
            "count": len(records),
            "recommendations": records,
            "best_recommendation": (
                records[0]
                if len(records) > 0
                else None
            ),
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation error: {str(e)}",
        )


# ============================================================
# AI RECOMMENDATION FOR A SPECIFIC ORDER
# ============================================================

@app.post("/orders/{order_id}/recommend")
def recommend_for_order(
    order_id: str,
    request: OrderRecommendationRequest,
):

    order = get_order(order_id)

    try:

        distance = float(order["distance_km"])
        weight = float(order["package_weight_kg"])

        recommendations = recommend_delivery(
            package_type=request.package_type,
            region=request.region,
            weather_condition=request.weather_condition,
            distance_km=distance,
            package_weight_kg=weight,
            priority=request.priority,
        )

        records = dataframe_to_records(recommendations)

        return {
            "success": True,
            "order_id": order_id,
            "distance_km": distance,
            "package_weight_kg": weight,
            "priority": request.priority,
            "count": len(records),
            "recommendations": records,
            "best_recommendation": (
                records[0]
                if len(records) > 0
                else None
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Order recommendation error: {str(e)}",
        )


# ============================================================
# CHECK COMPATIBILITY BETWEEN TWO ORDERS
# ============================================================

@app.get("/compatibility/{order1_id}/{order2_id}")
def check_compatibility(
    order1_id: str,
    order2_id: str,
):

    order1 = get_order(order1_id)
    order2 = get_order(order2_id)

    result = calculate_compatibility(
        order1,
        order2,
    )

    return {
        "success": True,
        "order_1": order1_id,
        "order_2": order2_id,
        "compatibility": result,
    }


# ============================================================
# ALL COLLABORATIVE DELIVERY PAIRS
# ============================================================

@app.get("/collaborative")
def collaborative_deliveries():

    try:

        recommendations = find_collaborative_deliveries(
            orders_df
        )

        records = dataframe_to_records(
            recommendations
        )

        return {
            "success": True,
            "count": len(records),
            "collaborative_deliveries": records,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Collaborative delivery error: {str(e)}",
        )


# ============================================================
# COLLABORATIVE DELIVERIES FOR ONE SELECTED ORDER
#
# NEW ENDPOINT
#
# Example:
# GET /orders/O101/collaborative
#
# This returns only compatible delivery pairs involving O101.
# ============================================================

@app.get("/orders/{order_id}/collaborative")
def collaborative_for_order(order_id: str):

    # Make sure selected order exists
    get_order(order_id)

    try:

        all_recommendations = find_collaborative_deliveries(
            orders_df
        )

        if (
            all_recommendations is None
            or len(all_recommendations) == 0
        ):
            return {
                "success": True,
                "order_id": order_id,
                "count": 0,
                "collaborative_deliveries": [],
            }

        selected = all_recommendations[
            (
                all_recommendations["order_1"].astype(str)
                == str(order_id)
            )
            |
            (
                all_recommendations["order_2"].astype(str)
                == str(order_id)
            )
        ].copy()

        records = dataframe_to_records(selected)

        return {
            "success": True,
            "order_id": order_id,
            "count": len(records),
            "collaborative_deliveries": records,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Order collaborative delivery error: {str(e)}",
        )


# ============================================================
# CREATE DELIVERY GROUPS
# ============================================================

@app.get("/groups")
def delivery_groups():

    try:

        groups = create_delivery_groups(
            orders_df,
            max_package_weight=10,
        )

        savings = calculate_delivery_savings(
            orders_df,
            groups,
        )

        return {
            "success": True,
            "savings": savings,
            "group_count": len(groups),
            "groups": dataframe_to_records(groups),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Delivery group error: {str(e)}",
        )


# ============================================================
# GET ONE DELIVERY GROUP
# ============================================================

@app.get("/groups/{group_id}")
def get_delivery_group(group_id: str):

    groups = create_delivery_groups(
        orders_df,
        max_package_weight=10,
    )

    if len(groups) == 0:

        raise HTTPException(
            status_code=404,
            detail="No delivery groups found",
        )

    matches = groups[
        groups["group_id"].astype(str)
        == str(group_id)
    ]

    if len(matches) == 0:

        raise HTTPException(
            status_code=404,
            detail=f"Delivery group {group_id} not found",
        )

    group = matches.iloc[0]

    order_ids = [
        order_id.strip()
        for order_id in str(group["orders"]).split(",")
    ]

    group_orders = orders_df[
        orders_df["order_id"].astype(str).isin(order_ids)
    ].copy()

    return {
        "success": True,
        "group": order_to_dict(group),
        "orders": dataframe_to_records(group_orders),
    }


# ============================================================
# OPTIMIZE ROUTE FOR DELIVERY GROUP
# ============================================================

@app.get("/groups/{group_id}/route")
def optimize_group_route(group_id: str):

    groups = create_delivery_groups(
        orders_df,
        max_package_weight=10,
    )

    if len(groups) == 0:

        raise HTTPException(
            status_code=404,
            detail="No delivery groups found",
        )

    matches = groups[
        groups["group_id"].astype(str)
        == str(group_id)
    ]

    if len(matches) == 0:

        raise HTTPException(
            status_code=404,
            detail=f"Delivery group {group_id} not found",
        )

    group = matches.iloc[0]

    order_ids = [
        order_id.strip()
        for order_id in str(group["orders"]).split(",")
    ]

    group_orders = orders_df[
        orders_df["order_id"].astype(str).isin(order_ids)
    ].copy()

    optimized_route = optimize_delivery_route(
        group_orders
    )

    summary = calculate_route_summary(
        optimized_route
    )

    return {
        "success": True,
        "group_id": group_id,
        "delivery_person": group["delivery_person"],
        "route_summary": summary,
        "route": dataframe_to_records(
            optimized_route
        ),
    }
# ============================================================
# OPTIMIZE ROUTE FOR ONE ORDER
# ============================================================

@app.get("/orders/{order_id}/route")
def optimize_order_route(order_id: str):

    # Make sure the order exists
    order = get_order(order_id)

    try:
        # Get the selected order as a one-row DataFrame
        order_df = orders_df[
            orders_df["order_id"].astype(str)
            == str(order_id)
        ].copy()

        if len(order_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Order {order_id} not found",
            )

        # Optimize route for this individual delivery
        optimized_route = optimize_delivery_route(
            order_df
        )

        # Calculate the same route summary used by group routes
        summary = calculate_route_summary(
            optimized_route
        )

        return {
            "success": True,
            "order_id": order_id,
            "delivery_person": "Assigned Agent",
            "route_summary": summary,
            "route": dataframe_to_records(
                optimized_route
            ),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Individual route optimization error: {str(e)}",
        )