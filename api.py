from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from models.recommendation import (
    orders_df,
    recommend_delivery,
    find_collaborative_deliveries,
    create_delivery_groups,
    calculate_compatibility,
    optimize_delivery_route,
    calculate_route_summary,
)


# ==========================================================
# APP
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


class OrderRecommendationRequest(BaseModel):
    package_type: str = "Standard"
    region: str = "South"
    weather_condition: str = "Clear"
    priority: str = "balanced"


# ==========================================================
# HELPERS
# ==========================================================

def clean_value(value):
    """
    Convert pandas/numpy values into JSON-safe values.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(dataframe):
    """
    Convert DataFrame to JSON-safe list of dictionaries.
    """

    if dataframe is None:
        return []

    if dataframe.empty:
        return []

    records = dataframe.to_dict(orient="records")

    cleaned = []

    for record in records:
        cleaned_record = {}

        for key, value in record.items():
            cleaned_record[key] = clean_value(value)

        cleaned.append(cleaned_record)

    return cleaned


def get_order(order_id):
    """
    Find an order by order_id.
    """

    result = orders_df[
        orders_df["order_id"].astype(str)
        ==
        str(order_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Order {order_id} not found"
        )

    return result.iloc[0]


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "app": "SyncDrop",
        "version": "1.0.0",
        "message": "SyncDrop backend is running"
    }


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "SyncDrop API"
    }


# ==========================================================
# ORDERS
# ==========================================================

@app.get("/orders")
def get_orders():

    try:

        return {
            "success": True,
            "count": len(orders_df),
            "orders": dataframe_to_records(orders_df)
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
def get_single_order(order_id: str):

    try:

        order = get_order(order_id)

        return {
            "success": True,
            "order": {
                key: clean_value(value)
                for key, value in order.to_dict().items()
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# RECOMMENDATION OPTIONS
# ==========================================================

@app.get("/recommend/options")
def recommendation_options():

    try:

        return {
            "success": True,

            "package_types": sorted(
                orders_df[
                    "package_type"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if "package_type" in orders_df.columns
            else [],

            "regions": sorted(
                orders_df[
                    "region"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if "region" in orders_df.columns
            else [],

            "weather_conditions": sorted(
                orders_df[
                    "weather_condition"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if "weather_condition" in orders_df.columns
            else [],

            "priorities": [
                "fast",
                "balanced",
                "economy"
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# GENERAL AI RECOMMENDATION
# ==========================================================

@app.post("/recommend")
def general_recommendation(
    request: RecommendationRequest
):

    try:

        result = recommend_delivery(
            package_type=request.package_type,
            region=request.region,
            weather_condition=request.weather_condition,
            distance_km=request.distance_km,
            package_weight_kg=request.package_weight_kg,
            priority=request.priority
        )

        return {
            "success": True,
            "recommendations":
                dataframe_to_records(result)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# RECOMMENDATION FOR EXISTING ORDER
# ==========================================================

@app.post("/orders/{order_id}/recommend")
def order_recommendation(
    order_id: str,
    request: OrderRecommendationRequest
):

    try:

        order = get_order(order_id)

        package_type = (
            request.package_type
            or str(order.get("package_type", "Standard"))
        )

        region = (
            request.region
            or str(order.get("region", "South"))
        )

        weather = (
            request.weather_condition
            or str(
                order.get(
                    "weather_condition",
                    "Clear"
                )
            )
        )

        distance = float(
            order.get(
                "distance_km",
                0
            )
        )

        weight = float(
            order.get(
                "package_weight_kg",
                0
            )
        )

        result = recommend_delivery(
            package_type=package_type,
            region=region,
            weather_condition=weather,
            distance_km=distance,
            package_weight_kg=weight,
            priority=request.priority
        )

        return {
            "success": True,
            "order_id": order_id,
            "recommendations":
                dataframe_to_records(result)
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# ALL COLLABORATIVE DELIVERIES
# ==========================================================

@app.get("/collaborative")
def collaborative_deliveries():

    try:

        result = find_collaborative_deliveries(
            orders_df
        )

        return {
            "success": True,
            "count": len(result),
            "collaborative_deliveries":
                dataframe_to_records(result)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# COLLABORATIVE DELIVERIES FOR ORDER
# ==========================================================

@app.get("/orders/{order_id}/collaborative")
def order_collaborative_deliveries(
    order_id: str
):

    try:

        order = get_order(order_id)

        result = find_collaborative_deliveries(
            orders_df
        )

        if result.empty:

            return {
                "success": True,
                "order_id": order_id,
                "collaborative_deliveries": []
            }

        filtered = result[
            (
                result["order_1"].astype(str)
                ==
                str(order_id)
            )
            |
            (
                result["order_2"].astype(str)
                ==
                str(order_id)
            )
        ]

        return {
            "success": True,
            "order_id": order_id,
            "collaborative_deliveries":
                dataframe_to_records(filtered)
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# DELIVERY GROUPS
# ==========================================================

@app.get("/groups")
def delivery_groups():

    try:

        groups = create_delivery_groups(
            orders_df
        )

        return {
            "success": True,
            "count": len(groups),
            "groups":
                dataframe_to_records(groups)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# SINGLE DELIVERY GROUP
# ==========================================================

@app.get("/groups/{group_id}")
def single_delivery_group(
    group_id: str
):

    try:

        groups = create_delivery_groups(
            orders_df
        )

        if groups.empty:

            raise HTTPException(
                status_code=404,
                detail="No delivery groups found"
            )

        result = groups[
            groups["group_id"].astype(str)
            ==
            str(group_id)
        ]

        if result.empty:

            raise HTTPException(
                status_code=404,
                detail=f"Group {group_id} not found"
            )

        return {
            "success": True,
            "group":
                dataframe_to_records(result)[0]
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# GROUP ROUTE
# ==========================================================

@app.get("/groups/{group_id}/route")
def group_route(group_id: str):

    try:

        groups = create_delivery_groups(
            orders_df
        )

        if groups.empty:

            raise HTTPException(
                status_code=404,
                detail="No delivery groups found"
            )

        group_result = groups[
            groups["group_id"].astype(str)
            ==
            str(group_id)
        ]

        if group_result.empty:

            raise HTTPException(
                status_code=404,
                detail=f"Group {group_id} not found"
            )

        group = group_result.iloc[0]

        order_ids = [
            x.strip()
            for x in str(
                group["orders"]
            ).split(",")
            if x.strip()
        ]

        group_orders = orders_df[
            orders_df["order_id"]
            .astype(str)
            .isin(order_ids)
        ].copy()

        route = optimize_delivery_route(
            group_orders
        )

        summary = calculate_route_summary(
            route
        )

        return {
            "success": True,
            "group_id": group_id,
            "route":
                dataframe_to_records(route),
            "summary": summary
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# INDIVIDUAL ORDER ROUTE
# ==========================================================

@app.get("/orders/{order_id}/route")
def individual_order_route(
    order_id: str
):

    try:

        order = get_order(order_id)

        route = pd.DataFrame([
            order
        ])

        route = optimize_delivery_route(
            route
        )

        summary = calculate_route_summary(
            route
        )

        return {
            "success": True,
            "order_id": order_id,
            "route":
                dataframe_to_records(route),
            "summary": summary
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# COMPATIBILITY
# ==========================================================

@app.get(
    "/compatibility/{order1_id}/{order2_id}"
)
def compatibility(
    order1_id: str,
    order2_id: str
):

    try:

        order1 = get_order(
            order1_id
        )

        order2 = get_order(
            order2_id
        )

        result = calculate_compatibility(
            order1,
            order2
        )

        return {
            "success": True,
            "order_1": order1_id,
            "order_2": order2_id,
            **result
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )