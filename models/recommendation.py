import pandas as pd
import joblib


# ==========================================================
# LOAD DATA AND MODELS
# ==========================================================

DATA_PATH = "data/delivery_ml.csv"
ORDERS_PATH = "data/orders.csv"

cost_model = joblib.load("models/cost_model.pkl")
delay_model = joblib.load("models/delay_model.pkl")

df = pd.read_csv(DATA_PATH)
orders_df = pd.read_csv(ORDERS_PATH)

print("Models loaded successfully!")
print("Dataset shape:", df.shape)
print("Orders dataset shape:", orders_df.shape)


# ==========================================================
# PART 1
# AI DELIVERY PARTNER RECOMMENDATION
# ==========================================================

def recommend_delivery(
    package_type,
    region,
    weather_condition,
    distance_km,
    package_weight_kg,
    priority="balanced"
):

    options = df[
        [
            "delivery_partner",
            "vehicle_type",
            "delivery_mode"
        ]
    ].drop_duplicates().copy()

    options["package_type"] = package_type
    options["region"] = region
    options["weather_condition"] = weather_condition
    options["distance_km"] = distance_km
    options["package_weight_kg"] = package_weight_kg

    partner_rating = (
        df.groupby("delivery_partner")["delivery_rating"]
        .mean()
    )

    options["delivery_rating"] = options[
        "delivery_partner"
    ].map(partner_rating)

    vehicle_speed = (
        df.groupby("vehicle_type")
        .apply(
            lambda x:
            x["distance_km"].sum()
            / x["delivery_time_hours"].sum(),
            include_groups=False
        )
    )

    options["estimated_speed"] = options[
        "vehicle_type"
    ].map(vehicle_speed)

    options["delivery_time_hours"] = (
        options["distance_km"]
        / options["estimated_speed"]
    )

    mode_limit = {
        "express": 4,
        "same day": 8,
        "two day": 16,
        "standard": 24
    }

    options["expected_time_hours"] = options[
        "delivery_mode"
    ].map(mode_limit)

    cost_features = [
        "delivery_partner",
        "package_type",
        "vehicle_type",
        "delivery_mode",
        "region",
        "weather_condition",
        "distance_km",
        "package_weight_kg",
        "delivery_time_hours",
        "delivery_rating"
    ]

    cost_input = options[cost_features]

    options["predicted_cost"] = (
        cost_model.predict(cost_input)
    )

    delay_features = [
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

    delay_input = options.copy()

    delay_input["delivery_cost"] = (
        options["predicted_cost"]
    )

    delay_input = delay_input[
        delay_features
    ]

    options["delay_prediction"] = (
        delay_model.predict(delay_input)
    )

    options["delay_status"] = (
        options["delay_prediction"]
        .map({
            0: "On Time",
            1: "Delayed"
        })
    )

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return series * 0

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    options["cost_score"] = normalize(
        options["predicted_cost"]
    )

    options["time_score"] = normalize(
        options["delivery_time_hours"]
    )

    options["rating_score"] = (
        1 - normalize(
            options["delivery_rating"]
        )
    )

    if priority == "fast":

        cost_weight = 0.20
        time_weight = 0.50
        rating_weight = 0.20
        delay_weight = 0.10

    elif priority == "economy":

        cost_weight = 0.55
        time_weight = 0.15
        rating_weight = 0.20
        delay_weight = 0.10

    else:

        cost_weight = 0.35
        time_weight = 0.30
        rating_weight = 0.20
        delay_weight = 0.15

    options["recommendation_score"] = (

        cost_weight * options["cost_score"]

        + time_weight * options["time_score"]

        + rating_weight * options["rating_score"]

        + delay_weight * options["delay_prediction"]
    )

    options = options[
        options["delivery_time_hours"]
        <=
        options["expected_time_hours"] * 1.5
    ]

    recommendations = (
        options
        .sort_values("recommendation_score")
        .head(10)
    )

    return recommendations[
        [
            "delivery_partner",
            "vehicle_type",
            "delivery_mode",
            "predicted_cost",
            "delivery_time_hours",
            "expected_time_hours",
            "delay_status",
            "delivery_rating",
            "recommendation_score"
        ]
    ]


# ==========================================================
# PART 2
# TIME COMPATIBILITY
# ==========================================================

def time_to_minutes(time_string):

    hours, minutes = map(
        int,
        time_string.split(":")
    )

    return hours * 60 + minutes


def calculate_time_compatibility(
    order1,
    order2
):

    start1 = time_to_minutes(
        order1["delivery_window_start"]
    )

    end1 = time_to_minutes(
        order1["delivery_window_end"]
    )

    start2 = time_to_minutes(
        order2["delivery_window_start"]
    )

    end2 = time_to_minutes(
        order2["delivery_window_end"]
    )

    overlap_start = max(
        start1,
        start2
    )

    overlap_end = min(
        end1,
        end2
    )

    return int(
        overlap_start <= overlap_end
    )


# ==========================================================
# PART 3
# CROSS-PLATFORM ORDER COMPATIBILITY
# ==========================================================

def calculate_compatibility(
    order1,
    order2
):
    """
    Determine whether two orders can be consolidated
    into one cross-platform delivery.

    Main project rule:

    SAME CUSTOMER
          +
    DIFFERENT PLATFORM
          +
    SAME LOCATION
          +
    COMPATIBLE TIME
          +
    CAPACITY
    """

    score = 0
    reasons = []

    same_customer = (
        order1["customer_id"]
        ==
        order2["customer_id"]
    )

    same_location = (
        order1["delivery_location"]
        ==
        order2["delivery_location"]
    )

    different_platform = (
        order1["platform"]
        !=
        order2["platform"]
    )

    time_compatible = (
        calculate_time_compatibility(
            order1,
            order2
        )
    )

    distance_difference = abs(
        order1["distance_km"]
        -
        order2["distance_km"]
    )

    similar_distance = (
        distance_difference <= 3
    )

    # ------------------------------------------------------
    # PRIMARY REQUIREMENT
    # ------------------------------------------------------

    if same_customer:

        score += 40

        reasons.append(
            "Same customer"
        )

    else:

        return {
            "compatibility_score": 0,
            "recommendation": "SEPARATE DELIVERY",
            "reasons": [
                "Different customer"
            ]
        }

    # ------------------------------------------------------
    # CROSS-PLATFORM REQUIREMENT
    # ------------------------------------------------------

    if different_platform:

        score += 20

        reasons.append(
            f"Cross-platform: "
            f"{order1['platform']} + "
            f"{order2['platform']}"
        )

    else:

        return {
            "compatibility_score": score,
            "recommendation": "SEPARATE DELIVERY",
            "reasons": [
                "Orders are from the same platform"
            ]
        }

    # ------------------------------------------------------
    # SAME DELIVERY LOCATION
    # ------------------------------------------------------

    if same_location:

        score += 20

        reasons.append(
            "Same delivery location"
        )

    else:

        return {
            "compatibility_score": score,
            "recommendation": "SEPARATE DELIVERY",
            "reasons": reasons + [
                "Different delivery locations"
            ]
        }

    # ------------------------------------------------------
    # TIME WINDOW
    # ------------------------------------------------------

    if time_compatible:

        score += 15

        reasons.append(
            "Compatible delivery window"
        )

    else:

        reasons.append(
            "Incompatible delivery window"
        )

    # ------------------------------------------------------
    # DISTANCE
    # ------------------------------------------------------

    if similar_distance:

        score += 5

        reasons.append(
            "Similar delivery distance"
        )

    # ------------------------------------------------------
    # FINAL DECISION
    # ------------------------------------------------------

    # Maximum = 100
    #
    # Same customer       = 40
    # Different platform  = 20
    # Same location       = 20
    # Time compatibility  = 15
    # Similar distance     = 5
    #
    # Require at least:
    # customer + platform + location + time
    # = 95 points

    if (
        same_customer
        and
        different_platform
        and
        same_location
        and
        time_compatible
    ):

        recommendation = (
            "DELIVER TOGETHER"
        )

    else:

        recommendation = (
            "SEPARATE DELIVERY"
        )

    return {

        "compatibility_score":
            score,

        "recommendation":
            recommendation,

        "reasons":
            reasons
    }


# ==========================================================
# PART 4
# FIND CROSS-PLATFORM DELIVERY PAIRS
# ==========================================================

def find_collaborative_deliveries(
    orders,
    max_package_weight=10
):

    recommendations = []

    for i in range(len(orders)):

        for j in range(i + 1, len(orders)):

            order1 = orders.iloc[i]
            order2 = orders.iloc[j]

            # --------------------------------------------------
            # DIFFERENT PLATFORM
            # --------------------------------------------------

            if (
                order1["platform"]
                ==
                order2["platform"]
            ):

                continue

            # --------------------------------------------------
            # SAME CUSTOMER
            # --------------------------------------------------

            if (
                order1["customer_id"]
                !=
                order2["customer_id"]
            ):

                continue

            # --------------------------------------------------
            # SAME LOCATION
            # --------------------------------------------------

            if (
                order1["delivery_location"]
                !=
                order2["delivery_location"]
            ):

                continue

            # --------------------------------------------------
            # PACKAGE CAPACITY
            # --------------------------------------------------

            total_weight = (
                order1["package_weight_kg"]
                +
                order2["package_weight_kg"]
            )

            if (
                total_weight
                >
                max_package_weight
            ):

                continue

            # --------------------------------------------------
            # COMPATIBILITY
            # --------------------------------------------------

            result = calculate_compatibility(
                order1,
                order2
            )

            if (
                result["recommendation"]
                ==
                "DELIVER TOGETHER"
            ):

                recommendations.append({

                    "order_1":
                        order1["order_id"],

                    "order_2":
                        order2["order_id"],

                    "customer_1":
                        order1["customer_id"],

                    "customer_2":
                        order2["customer_id"],

                    "platform_1":
                        order1["platform"],

                    "platform_2":
                        order2["platform"],

                    "location_1":
                        order1["delivery_location"],

                    "location_2":
                        order2["delivery_location"],

                    "compatibility_score":
                        result[
                            "compatibility_score"
                        ],

                    "total_weight":
                        round(
                            total_weight,
                            2
                        ),

                    "reason":
                        ", ".join(
                            result["reasons"]
                        )
                })

    recommendations_df = pd.DataFrame(
        recommendations
    )

    if len(recommendations_df) > 0:

        recommendations_df = (
            recommendations_df
            .sort_values(
                "compatibility_score",
                ascending=False
            )
        )

    return recommendations_df


# ==========================================================
# PART 5
# MULTI-ORDER CROSS-PLATFORM GROUPING
# ==========================================================

def create_delivery_groups(
    orders,
    max_package_weight=10
):
    """
    Create consolidated delivery groups.

    A valid collaborative group must contain:

    1. Same customer
    2. Same delivery location
    3. Different platforms
    4. Compatible delivery windows
    5. Total weight within capacity

    Each group is assigned to ONE delivery person.
    """

    orders = orders.copy()

    groups = []

    processed_orders = set()

    delivery_person_number = 1

    # ------------------------------------------------------
    # PROCESS EACH CUSTOMER
    # ------------------------------------------------------

    for customer_id, customer_orders in (
        orders.groupby("customer_id")
    ):

        customer_orders = (
            customer_orders
            .copy()
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # PROCESS EACH LOCATION
        # --------------------------------------------------

        for location, location_orders in (
            customer_orders
            .groupby("delivery_location")
        ):

            location_orders = (
                location_orders
                .copy()
                .reset_index(drop=True)
            )

            # ------------------------------------------------
            # GREEDY GROUP CREATION
            # ------------------------------------------------

            for i in range(
                len(location_orders)
            ):

                first_order = (
                    location_orders.iloc[i]
                )

                first_id = (
                    first_order["order_id"]
                )

                if first_id in processed_orders:
                    continue

                current_group = [
                    first_order
                ]

                current_weight = (
                    first_order[
                        "package_weight_kg"
                    ]
                )

                used_platforms = {
                    first_order["platform"]
                }

                # --------------------------------------------
                # FIND ADDITIONAL ORDERS
                # --------------------------------------------

                for j in range(
                    i + 1,
                    len(location_orders)
                ):

                    candidate = (
                        location_orders.iloc[j]
                    )

                    candidate_id = (
                        candidate["order_id"]
                    )

                    if candidate_id in processed_orders:
                        continue

                    # ----------------------------------------
                    # PLATFORM MUST BE DIFFERENT
                    # ----------------------------------------

                    if (
                        candidate["platform"]
                        in
                        used_platforms
                    ):

                        continue

                    # ----------------------------------------
                    # CAPACITY
                    # ----------------------------------------

                    new_weight = (
                        current_weight
                        +
                        candidate[
                            "package_weight_kg"
                        ]
                    )

                    if (
                        new_weight
                        >
                        max_package_weight
                    ):

                        continue

                    # ----------------------------------------
                    # CHECK AGAINST EVERY ORDER
                    # ----------------------------------------

                    compatible = True

                    for existing_order in (
                        current_group
                    ):

                        result = (
                            calculate_compatibility(
                                existing_order,
                                candidate
                            )
                        )

                        if (
                            result[
                                "recommendation"
                            ]
                            !=
                            "DELIVER TOGETHER"
                        ):

                            compatible = False
                            break

                    # ----------------------------------------
                    # ADD TO GROUP
                    # ----------------------------------------

                    if compatible:

                        current_group.append(
                            candidate
                        )

                        current_weight = (
                            new_weight
                        )

                        used_platforms.add(
                            candidate["platform"]
                        )

                # ------------------------------------------------
                # ONLY CREATE COLLABORATIVE GROUP
                # IF AT LEAST TWO ORDERS EXIST
                # ------------------------------------------------

                if len(current_group) < 2:
                    continue

                # ------------------------------------------------
                # GROUP INFORMATION
                # ------------------------------------------------

                group_id = (
                    f"GROUP-{delivery_person_number:03d}"
                )

                delivery_person = (
                    f"DP-{delivery_person_number:03d}"
                )

                order_ids = [

                    order["order_id"]

                    for order in current_group
                ]

                customers = list(
                    dict.fromkeys([

                        order["customer_id"]

                        for order in current_group
                    ])
                )

                platforms = list(
                    dict.fromkeys([

                        order["platform"]

                        for order in current_group
                    ])
                )

                locations = list(
                    dict.fromkeys([

                        order["delivery_location"]

                        for order in current_group
                    ])
                )

                # ------------------------------------------------
                # GROUP COMPATIBILITY SCORE
                # ------------------------------------------------

                pair_scores = []

                for x in range(
                    len(current_group)
                ):

                    for y in range(
                        x + 1,
                        len(current_group)
                    ):

                        pair_result = (
                            calculate_compatibility(
                                current_group[x],
                                current_group[y]
                            )
                        )

                        pair_scores.append(
                            pair_result[
                                "compatibility_score"
                            ]
                        )

                average_score = (
                    sum(pair_scores)
                    /
                    len(pair_scores)
                    if pair_scores
                    else 0
                )

                # ------------------------------------------------
                # SAVE GROUP
                # ------------------------------------------------

                groups.append({

                    "group_id":
                        group_id,

                    "delivery_person":
                        delivery_person,

                    "orders":
                        ", ".join(
                            order_ids
                        ),

                    "number_of_orders":
                        len(current_group),

                    "customers":
                        ", ".join(
                            customers
                        ),

                    "platforms":
                        ", ".join(
                            platforms
                        ),

                    "locations":
                        ", ".join(
                            locations
                        ),

                    "total_weight":
                        round(
                            current_weight,
                            2
                        ),

                    "compatibility_score":
                        round(
                            average_score,
                            2
                        ),

                    "recommendation":
                        "DELIVER TOGETHER"
                })

                # ------------------------------------------------
                # MARK ORDERS AS PROCESSED
                # ------------------------------------------------

                for order in current_group:

                    processed_orders.add(
                        order["order_id"]
                    )

                delivery_person_number += 1

    return pd.DataFrame(groups)


# ==========================================================
# PART 6
# DELIVERY EFFICIENCY
# ==========================================================

def calculate_delivery_savings(
    orders,
    groups
):

    total_orders = len(
        orders
    )

    collaborative_groups = len(
        groups
    )

    # ------------------------------------------------------
    # ORDERS ACTUALLY CONSOLIDATED
    # ------------------------------------------------------

    if (
        collaborative_groups > 0
    ):

        consolidated_orders = int(
            groups[
                "number_of_orders"
            ].sum()
        )

    else:

        consolidated_orders = 0

    # ------------------------------------------------------
    # TRIPS SAVED
    # ------------------------------------------------------

    trips_saved = (
        consolidated_orders
        -
        collaborative_groups
    )

    # ------------------------------------------------------
    # EFFICIENCY
    # ------------------------------------------------------

    if total_orders > 0:

        efficiency = (
            trips_saved
            /
            total_orders
            *
            100
        )

    else:

        efficiency = 0

    return {

        "total_orders":
            total_orders,

        "delivery_groups":
            collaborative_groups,

        "consolidated_orders":
            consolidated_orders,

        "trips_saved":
            trips_saved,

        "efficiency_percentage":
            round(
                efficiency,
                2
            )
    }
# ==========================================================
# PART 7
# ROUTE OPTIMIZATION
# ==========================================================

def optimize_delivery_route(
    group_orders,
    distance_matrix=None
):
    """
    Optimize the order sequence for a consolidated delivery.

    Current prototype:
    Orders are sequenced using their recorded distance_km.

    Later this can be upgraded to real road/GIS routing.
    """

    if group_orders is None or len(group_orders) == 0:
        return group_orders.copy()

    route = group_orders.copy()

    # Make sure distance is numeric
    route["distance_km"] = pd.to_numeric(
        route["distance_km"],
        errors="coerce"
    ).fillna(0)

    # Start with the nearest recorded delivery distance
    route = (
        route
        .sort_values("distance_km", ascending=True)
        .reset_index(drop=True)
    )

    return route


# ==========================================================
# PART 8
# ROUTE SUMMARY
# ==========================================================

def calculate_route_summary(route):
    """
    Calculate consolidated route statistics.

    Returns the key names expected by app.py.
    """

    if route is None or len(route) == 0:

        return {
            "number_of_orders": 0,
            "total_distance_km": 0.0,
            "estimated_time_hours": 0.0
        }

    route = route.copy()

    route["distance_km"] = pd.to_numeric(
        route["distance_km"],
        errors="coerce"
    ).fillna(0)

    # Combined recorded distance of all orders
    total_distance = float(
        route["distance_km"].sum()
    )

    # Average delivery speed
    average_speed = 40.0

    estimated_time = (
        total_distance / average_speed
        if average_speed > 0
        else 0.0
    )

    return {
        "number_of_orders": len(route),

        "total_distance_km": round(
            total_distance,
            2
        ),

        "estimated_time_hours": round(
            estimated_time,
            2
        )
    }