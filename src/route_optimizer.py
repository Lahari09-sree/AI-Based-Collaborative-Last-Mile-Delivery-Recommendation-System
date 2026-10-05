import pandas as pd


def time_to_minutes(time_str):
    """Convert HH:MM time into minutes."""

    hours, minutes = map(int, str(time_str).split(":"))

    return hours * 60 + minutes


def optimize_route(orders_df):
    """
    Optimize delivery stops based on:
    1. Delivery time windows
    2. Distance
    3. Consolidation of orders going to the same location
    """

    if orders_df.empty:
        return pd.DataFrame()

    df = orders_df.copy()

    # Convert delivery times to minutes
    df["start_minutes"] = df["delivery_window_start"].apply(
        time_to_minutes
    )

    df["end_minutes"] = df["delivery_window_end"].apply(
        time_to_minutes
    )

    # Consolidate orders going to the same location
    route = (
        df.groupby("delivery_location")
        .agg(
            total_orders=("order_id", "count"),
            order_ids=("order_id", lambda x: list(x)),
            platforms=("platform", lambda x: list(x.unique())),
            avg_distance_km=("distance_km", "mean"),
            total_weight_kg=("package_weight_kg", "sum"),
            earliest_window=("start_minutes", "min"),
            latest_window=("end_minutes", "max")
        )
        .reset_index()
    )

    # Prioritize earlier delivery windows, then shorter distance
    route = route.sort_values(
        by=["earliest_window", "avg_distance_km"]
    ).reset_index(drop=True)

    # Add stop numbers
    route["stop_number"] = range(1, len(route) + 1)

    # Format delivery window
    route["delivery_window"] = (
        route["earliest_window"].apply(
            lambda x: f"{x // 60:02d}:{x % 60:02d}"
        )
        + " - "
        + route["latest_window"].apply(
            lambda x: f"{x // 60:02d}:{x % 60:02d}"
        )
    )

    # Select final columns
    route = route[
        [
            "stop_number",
            "delivery_location",
            "total_orders",
            "order_ids",
            "platforms",
            "avg_distance_km",
            "total_weight_kg",
            "delivery_window"
        ]
    ]

    return route


def get_route_summary(route_df):
    """Generate route statistics."""

    if route_df.empty:
        return {
            "total_stops": 0,
            "total_orders": 0,
            "estimated_distance_km": 0
        }

    return {
        "total_stops": len(route_df),
        "total_orders": int(route_df["total_orders"].sum()),
        "estimated_distance_km": round(
            route_df["avg_distance_km"].sum(),
            2
        )
    }


# Test the route optimizer
if __name__ == "__main__":

    orders = pd.read_csv("data/orders.csv")

    optimized_route = optimize_route(orders)

    print("\nOPTIMIZED DELIVERY ROUTE\n")

    print(optimized_route.to_string(index=False))

    print("\nROUTE SUMMARY")

    print(get_route_summary(optimized_route))