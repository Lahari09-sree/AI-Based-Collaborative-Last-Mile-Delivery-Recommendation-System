import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
# ==========================================================
# LOCATION COORDINATES
# ==========================================================

LOCATION_COORDINATES = {
    "Location_A": (12.9716, 77.5946),
    "Location_B": (12.9352, 77.6245),
    "Location_C": (12.9850, 77.5600),
    "Location_D": (13.0067, 77.5810),
    "Location_E": (12.9279, 77.6271),
}
from models.recommendation import (
    find_collaborative_deliveries,
    create_delivery_groups,
    calculate_delivery_savings,
    calculate_compatibility,
    optimize_delivery_route,
    calculate_route_summary
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Last-Mile Delivery",
    page_icon="🚚",
    layout="wide"
)


# ==========================================================
# LOAD DATA AND MODELS
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/delivery_ml.csv")
    orders_df = pd.read_csv("data/orders.csv")

    return df, orders_df


@st.cache_resource
def load_models():

    cost_model = joblib.load("models/cost_model.pkl")
    delay_model = joblib.load("models/delay_model.pkl")

    return cost_model, delay_model


df, orders_df = load_data()

cost_model, delay_model = load_models()


# ==========================================================
# PRE-CALCULATE COLLABORATIVE DELIVERY DATA
# ==========================================================

collaborative_results = find_collaborative_deliveries(
    orders_df,
    max_package_weight=10
)

delivery_groups = create_delivery_groups(
    orders_df,
    max_package_weight=10
)

savings = calculate_delivery_savings(
    orders_df,
    delivery_groups
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🚚 AI Last-Mile Delivery</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Cross-Platform Order Consolidation & Intelligent Delivery Recommendation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    """
    🌐 **Project Core Idea**

    Orders placed by the **same customer on different platforms**
    such as **Myntra, Flipkart and Amazon** can be identified and
    consolidated into a **single last-mile delivery**.

    The system checks customer, location, delivery window,
    distance and package capacity before assigning the
    compatible orders to **one delivery person**.
    """
)
# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("📦 Delivery Information")


package_type = st.sidebar.selectbox(
    "Package Type",
    sorted(df["package_type"].unique()),
    key="package_type_input"
)


region = st.sidebar.selectbox(
    "Region",
    sorted(df["region"].unique()),
    key="region_input"
)


weather = st.sidebar.selectbox(
    "Weather Condition",
    sorted(df["weather_condition"].unique()),
    key="weather_input"
)


distance = st.sidebar.number_input(
    "Distance (km)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    key="distance_input"
)


weight = st.sidebar.number_input(
    "Package Weight (kg)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    key="weight_input"
)


rating = st.sidebar.slider(
    "Expected Customer Rating",
    min_value=1,
    max_value=5,
    value=4,
    key="rating_input"
)


st.sidebar.divider()

st.sidebar.write(
    "Click below to generate AI delivery recommendations."
)


predict_button = st.sidebar.button(
    "🚀 Predict Delivery",
    key="predict_delivery_button"
)


# ==========================================================
# PART 1 — AI DELIVERY PARTNER RECOMMENDATION
# ==========================================================

if predict_button:

    st.header("🤖 AI Delivery Partner Recommendation")

    st.write(
        """
        Based on the delivery information provided, the system
        evaluates available delivery partners, vehicles and
        delivery modes using machine learning to recommend
        the most suitable option.
        """
    )

    partners = df["delivery_partner"].unique()
    vehicles = df["vehicle_type"].unique()
    modes = df["delivery_mode"].unique()

    results = []

    # ======================================================
    # GENERATE ALL POSSIBLE DELIVERY OPTIONS
    # ======================================================

    for partner in partners:

        for vehicle in vehicles:

            for mode in modes:

                # ------------------------------------------
                # EXPECTED DELIVERY TIME
                # ------------------------------------------

                if mode == "express":
                    expected_time = 4

                elif mode == "same day":
                    expected_time = 8

                elif mode == "two day":
                    expected_time = 16

                else:
                    expected_time = 24

                # ------------------------------------------
                # ESTIMATE COST
                # ------------------------------------------

                similar = df[
                    (df["delivery_partner"] == partner)
                    &
                    (df["vehicle_type"] == vehicle)
                    &
                    (df["delivery_mode"] == mode)
                ]

                if len(similar) > 0:

                    estimated_cost = similar[
                        "delivery_cost"
                    ].median()

                else:

                    estimated_cost = df[
                        "delivery_cost"
                    ].median()

                # ------------------------------------------
                # DELIVERY TIME
                # ------------------------------------------

                estimated_delivery_time = distance / 70

                if estimated_delivery_time <= 0:
                    estimated_delivery_time = 1

                # ------------------------------------------
                # MODEL INPUT
                # ------------------------------------------

                model_input = pd.DataFrame({

                    "delivery_partner": [partner],

                    "package_type": [package_type],

                    "vehicle_type": [vehicle],

                    "delivery_mode": [mode],

                    "region": [region],

                    "weather_condition": [weather],

                    "distance_km": [distance],

                    "package_weight_kg": [weight],

                    "delivery_time_hours": [
                        estimated_delivery_time
                    ],

                    "delivery_rating": [rating],

                    "delivery_cost": [
                        estimated_cost
                    ]

                })

                # ------------------------------------------
                # COST PREDICTION
                # ------------------------------------------

                try:

                    predicted_cost = cost_model.predict(
                        model_input
                    )[0]

                except Exception:

                    predicted_cost = estimated_cost

                # ------------------------------------------
                # DELAY PREDICTION
                # ------------------------------------------

                try:

                    delay_prediction = delay_model.predict(
                        model_input
                    )[0]

                    try:

                        delay_probability = (
                            delay_model.predict_proba(
                                model_input
                            )[0][1]
                        )

                    except Exception:

                        delay_probability = 0.0

                except Exception:

                    delay_prediction = 0

                    delay_probability = 0.0

                # ------------------------------------------
                # DELAY STATUS
                # ------------------------------------------

                if delay_prediction == 1:

                    delay_status = "Delayed"

                else:

                    delay_status = "On Time"

                # ------------------------------------------
                # RECOMMENDATION SCORE
                # ------------------------------------------

                score = (

                    predicted_cost / 1000

                    +

                    delay_probability * 10

                    -

                    rating

                )

                # ------------------------------------------
                # SAVE RESULT
                # ------------------------------------------

                results.append({

                    "delivery_partner":
                        partner,

                    "vehicle_type":
                        vehicle,

                    "delivery_mode":
                        mode,

                    "predicted_cost":
                        predicted_cost,

                    "delivery_time_hours":
                        estimated_delivery_time,

                    "expected_time_hours":
                        expected_time,

                    "delay_probability":
                        delay_probability,

                    "delay_status":
                        delay_status,

                    "delivery_rating":
                        rating,

                    "recommendation_score":
                        score

                })

    # ======================================================
    # CREATE RESULTS DATAFRAME
    #
    # IMPORTANT:
    # THIS IS OUTSIDE ALL THREE LOOPS
    # ======================================================

    recommendations = pd.DataFrame(results)

    recommendations = recommendations.sort_values(
        "recommendation_score"
    )

    top10 = recommendations.head(10)

    best = top10.iloc[0]

    # ======================================================
    # BEST RECOMMENDATION
    # ======================================================

    st.subheader(
        "🏆 Best Delivery Recommendation"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Delivery Partner",
            best["delivery_partner"]
        )

    with col2:

        st.metric(
            "Vehicle",
            best["vehicle_type"]
        )

    with col3:

        st.metric(
            "Delivery Mode",
            best["delivery_mode"]
        )

    with col4:

        st.metric(
            "Predicted Cost",
            f"₹{best['predicted_cost']:,.2f}"
        )

    # ======================================================
    # ADDITIONAL INFORMATION
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Delivery Time",
            f"{best['delivery_time_hours']:.2f} hrs"
        )

    with col2:

        st.metric(
            "Delay Probability",
            f"{best['delay_probability'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Rating",
            f"{best['delivery_rating']:.1f}/5"
        )

    # ======================================================
    # DELAY STATUS
    # ======================================================

    if best["delay_status"] == "On Time":

        st.success(
            "✅ Recommended option is predicted to be ON TIME."
        )

    else:

        st.warning(
            "⚠️ Recommended option has a delay risk."
        )

    # ======================================================
    # TOP 10 RECOMMENDATIONS
    # ======================================================

    st.subheader(
        "🥇 Top 10 Recommended Delivery Options"
    )

    display_df = top10.copy()

    display_df["predicted_cost"] = (
        display_df["predicted_cost"].round(2)
    )

    display_df["delivery_time_hours"] = (
        display_df["delivery_time_hours"].round(2)
    )

    display_df["delay_probability"] = (
        display_df["delay_probability"] * 100
    ).round(2)

    display_df["recommendation_score"] = (
        display_df["recommendation_score"].round(4)
    )

    display_df = display_df.rename(
        columns={

            "delivery_partner":
                "Delivery Partner",

            "vehicle_type":
                "Vehicle",

            "delivery_mode":
                "Delivery Mode",

            "predicted_cost":
                "Predicted Cost (₹)",

            "delivery_time_hours":
                "Delivery Time (hrs)",

            "expected_time_hours":
                "Expected Time (hrs)",

            "delay_probability":
                "Delay Probability (%)",

            "delay_status":
                "Delay Status",

            "delivery_rating":
                "Rating",

            "recommendation_score":
                "Recommendation Score"

        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PART 2 — DATASET OVERVIEW
# ==========================================================

st.divider()

st.subheader("📊 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Deliveries",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Delivery Partners",
        df["delivery_partner"].nunique()
    )


with col3:

    st.metric(
        "Vehicle Types",
        df["vehicle_type"].nunique()
    )


with col4:

    st.metric(
        "Delivery Modes",
        df["delivery_mode"].nunique()
    )


# ==========================================================
# HISTORICAL DATA
# ==========================================================

with st.expander(
    "🔎 View Historical Dataset"
):

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PART 3 — DELIVERY ANALYTICS
# ==========================================================

st.divider()

st.subheader("📈 Delivery Analytics")


col1, col2 = st.columns(2)


with col1:

    st.write(
        "### Average Cost by Delivery Mode"
    )

    mode_cost = (
        df.groupby("delivery_mode")[
            "delivery_cost"
        ]
        .mean()
        .sort_values()
    )

    st.bar_chart(mode_cost)


with col2:

    st.write(
        "### Average Rating by Delivery Partner"
    )

    partner_rating = (
        df.groupby("delivery_partner")[
            "delivery_rating"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(partner_rating)


# ==========================================================
# PART 4 — DELAY ANALYSIS
# ==========================================================

st.divider()

st.subheader("⏱️ Delay Analysis")


delay_data = df.copy()


delay_data["expected_time"] = (
    delay_data["delivery_mode"].map({

        "express": 4,

        "same day": 8,

        "two day": 16,

        "standard": 24

    })
)


delay_data["delay_status"] = (
    delay_data["delivery_time_hours"]
    >
    delay_data["expected_time"]
)


delay_counts = (
    delay_data["delay_status"]
    .value_counts()
)


delay_labels = pd.Series({

    "On Time":
        delay_counts.get(False, 0),

    "Delayed":
        delay_counts.get(True, 0)

})


st.bar_chart(delay_labels)


total_deliveries = len(
    delay_data
)


delayed_deliveries = (
    delay_counts.get(True, 0)
)


if total_deliveries > 0:

    delay_percentage = (
        delayed_deliveries
        /
        total_deliveries
        *
        100
    )

else:

    delay_percentage = 0


st.metric(
    "Overall Delay Rate",
    f"{delay_percentage:.2f}%"
)


# ==========================================================
# PART 5 — COLLABORATIVE LAST-MILE DELIVERY
# ==========================================================

st.divider()

st.subheader(
    "🤝 Collaborative Last-Mile Delivery"
)


st.write(
    """
    The system identifies orders from different platforms
    that can potentially be delivered together by the same
    delivery person.
    
    Compatibility is calculated using:
    
    • Same customer
    • Same delivery location
    • Compatible delivery window
    • Similar delivery distance
    • Package capacity
    """
)



# ==========================================================
# PART 6 — DELIVERY GROUPING
# ==========================================================

st.divider()

st.subheader(
    "🚚 Recommended Multi-Order Delivery Groups"
)


st.write(
    """
    Orders that are compatible are grouped together.
    One delivery person can handle multiple orders,
    reducing unnecessary delivery trips.
    """
)


# ==========================================================
# KPI CARDS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📦 Total Orders",
        savings["total_orders"]
    )


with col2:

    st.metric(
        "🚚 Delivery Groups",
        savings["delivery_groups"]
    )


with col3:

    st.metric(
        "💰 Potential Trips Saved",
        savings["trips_saved"]
    )


with col4:

    st.metric(
        "📈 Delivery Efficiency",
        f"{savings['efficiency_percentage']}%"
    )


# ==========================================================
# DELIVERY GROUP TABLE
# ==========================================================

if len(delivery_groups) > 0:

    display_groups = (
        delivery_groups.copy()
    )


    display_groups = (
        display_groups.rename(
            columns={

                "group_id":
                    "Group ID",

                "delivery_person":
                    "Delivery Person",

                "orders":
                    "Orders",

                "number_of_orders":
                    "Number of Orders",

                "customers":
                    "Customers",

                "platforms":
                    "Platforms",

                "locations":
                    "Locations",

                "total_weight":
                    "Total Weight (kg)",

                "recommendation":
                    "Recommendation"

            }
        )
    )


    st.dataframe(
        display_groups,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No compatible delivery groups found."
    )
# ==========================================================
# STEP 1 — ORDERS SCREEN
# ==========================================================

st.divider()

st.header("📦 Orders")

st.write(
    """
    Select an order to view its details and use AI to find
    the best delivery option.
    """
)

# ----------------------------------------------------------
# ORDERS TABLE
# ----------------------------------------------------------

orders_display = orders_df.copy()

orders_display = orders_display.rename(
    columns={
        "order_id": "Order ID",
        "customer_id": "Customer",
        "platform": "Platform",
        "delivery_location": "Location",
        "delivery_window_start": "Window Start",
        "delivery_window_end": "Window End",
        "distance_km": "Distance (km)",
        "package_weight_kg": "Weight (kg)"
    }
)

st.dataframe(
    orders_display,
    use_container_width=True,
    hide_index=True
)


# ----------------------------------------------------------
# SELECT ORDER
# ----------------------------------------------------------

st.subheader("📋 Select an Order")

order_ids = orders_df["order_id"].tolist()

selected_order = st.selectbox(
    "Choose an order",
    order_ids,
    key="orders_screen_selection"
)

selected_data = orders_df[
    orders_df["order_id"] == selected_order
].iloc[0]


# ----------------------------------------------------------
# ORDER DETAILS
# ----------------------------------------------------------

st.subheader("📋 Order Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Order ID",
        selected_data["order_id"]
    )

with col2:
    st.metric(
        "Customer",
        selected_data["customer_id"]
    )

with col3:
    st.metric(
        "Platform",
        selected_data["platform"]
    )

with col4:
    st.metric(
        "Location",
        selected_data["delivery_location"]
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Distance",
        f"{selected_data['distance_km']} km"
    )

with col2:
    st.metric(
        "Package Weight",
        f"{selected_data['package_weight_kg']} kg"
    )

with col3:

    delivery_window = (
        f"{selected_data['delivery_window_start']} - "
        f"{selected_data['delivery_window_end']}"
    )

    st.metric(
        "Delivery Window",
        delivery_window
    )

with col4:

    current_status = st.session_state.get(
        "delivery_status",
        "Pending"
    )

    status_display = {
        "Pending": "🟡 Pending",
        "Assigned": "🔵 Assigned",
        "Out for Delivery": "🟠 Out for Delivery",
        "Delivered": "🟢 Delivered"
    }

    st.metric(
        "Status",
        status_display.get(
            current_status,
            "🟡 Pending"
        )
    )
# ==========================================================
# STEP 2 — AI DELIVERY RECOMMENDATION
# ==========================================================

st.divider()

st.subheader("🤖 AI Delivery Recommendation")

st.write(
    f"""
    The AI system analyzes **{selected_order}** and searches
    for compatible orders that can be delivered together.
    """
)


# ==========================================================
# FIND BEST DELIVERY BUTTON
# ==========================================================

find_best_delivery = st.button(
    "🚀 FIND BEST DELIVERY",
    key="find_best_delivery_v2",
    type="primary",
    use_container_width=True
)


if find_best_delivery:

    compatible_orders_ai = []

    for _, candidate in orders_df.iterrows():

        # Do not compare order with itself
        if candidate["order_id"] == selected_order:
            continue

        result = calculate_compatibility(
            selected_data,
            candidate
        )

        if result["recommendation"] == "DELIVER TOGETHER":

            compatible_orders_ai.append(
                {
                    "order_id":
                        candidate["order_id"],

                    "customer":
                        candidate["customer_id"],

                    "platform":
                        candidate["platform"],

                    "location":
                        candidate["delivery_location"],

                    "distance":
                        candidate["distance_km"],

                    "weight":
                        candidate["package_weight_kg"],

                    "score":
                        result["compatibility_score"],

                    "reason":
                        ", ".join(
                            result["reasons"]
                        )
                }
            )


    # ======================================================
    # BEST MATCH FOUND
    # ======================================================

    if compatible_orders_ai:

        compatible_orders_ai.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        best_match = compatible_orders_ai[0]


        # --------------------------------------------------
        # SAVE IMPORTANT VALUES IN SESSION STATE
        # --------------------------------------------------

        st.session_state["best_delivery"] = best_match

        st.session_state["assigned_order"] = (
            selected_order
        )

        st.session_state["recommended_order"] = (
            best_match["order_id"]
        )

        st.session_state["delivery_status"] = (
            "Pending"
        )


        # --------------------------------------------------
        # SHOW RESULT
        # --------------------------------------------------

        st.success(
            "🎯 Best compatible delivery found!"
        )

        st.markdown(
            "### 🎯 AI Recommended Order"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Recommended Order",
                best_match["order_id"]
            )


        with col2:

            st.metric(
                "Compatibility",
                f"{best_match['score']:.1f}%"
            )


        with col3:

            st.metric(
                "Platform",
                best_match["platform"]
            )


        st.write(
            f"📍 **Location:** "
            f"{best_match['location']}"
        )

        st.write(
            f"📦 **Package Weight:** "
            f"{best_match['weight']} kg"
        )

        st.write(
            f"🚚 **Distance:** "
            f"{best_match['distance']} km"
        )


        st.info(
            f"""
            💡 **Why these orders can be delivered together:**

            {best_match['reason']}
            """
        )


        st.markdown(
            """
            ### 🔗 Collaborative Delivery

            The AI recommends delivering these orders
            together because they satisfy the delivery
            compatibility constraints.
            """
        )


        st.write(
            f"**{selected_order}** → "
            f"**{best_match['order_id']}**"
        )


    else:

        # No match
        st.session_state.pop(
            "best_delivery",
            None
        )

        st.session_state.pop(
            "recommended_order",
            None
        )

        st.session_state.pop(
            "delivery_agent",
            None
        )

        st.warning(
            "⚠️ No compatible delivery order was found."
        )


# ==========================================================
# STEP 3 — DELIVERY PERSON ASSIGNMENT
# ==========================================================

if (
    "best_delivery" in st.session_state
    and "assigned_order" in st.session_state
):

    st.divider()

    st.subheader(
        "👤 Select Delivery Person"
    )


    # ------------------------------------------------------
    # GET ORDERS FROM SESSION STATE
    # ------------------------------------------------------

    assigned_order = st.session_state[
        "assigned_order"
    ]

    recommended_order = st.session_state[
        "recommended_order"
    ]

    best = st.session_state[
        "best_delivery"
    ]


    st.write(
        f"""
        Orders **{assigned_order}** and
        **{recommended_order}** are ready to be
        assigned to one delivery person.
        """
    )


    # ------------------------------------------------------
    # AVAILABLE DELIVERY AGENTS
    # ------------------------------------------------------

    if (
        "delivery_person"
        in delivery_groups.columns
    ):

        available_agents = (
            delivery_groups[
                "delivery_person"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        available_agents = [
            "DP-001",
            "DP-002",
            "DP-003",
            "DP-004"
        ]


    if not available_agents:

        available_agents = [
            "DP-001",
            "DP-002",
            "DP-003",
            "DP-004"
        ]


    # ------------------------------------------------------
    # SELECT DELIVERY PERSON
    # ------------------------------------------------------

    selected_agent = st.selectbox(

        "Choose a delivery person",

        available_agents,

        key="selected_delivery_person"

    )


    # ------------------------------------------------------
    # CONFIRM ASSIGNMENT
    # ------------------------------------------------------

    confirm_assignment = st.button(

        "✅ CONFIRM ASSIGNMENT",

        key="confirm_assignment",

        type="primary",

        use_container_width=True

    )


    if confirm_assignment:

        st.session_state[
            "delivery_agent"
        ] = selected_agent

        st.session_state[
            "delivery_status"
        ] = "Assigned"


        st.success(

            f"""
            ✅ Orders **{assigned_order}** and
            **{recommended_order}** assigned to
            **{selected_agent}**.
            """

        )


# ==========================================================
# STEP 4 — DELIVERY ROUTE OPTIMIZATION
# ==========================================================

st.divider()

st.subheader(
    "🗺️ Delivery Route Optimization"
)


# ==========================================================
# CHECK ASSIGNMENT
# ==========================================================

if (
    "best_delivery" in st.session_state
    and "delivery_agent" in st.session_state
    and "assigned_order" in st.session_state
    and "recommended_order" in st.session_state
):

    # ------------------------------------------------------
    # GET EVERYTHING FROM SESSION STATE
    # ------------------------------------------------------

    assigned_order = st.session_state[
        "assigned_order"
    ]

    recommended_order = st.session_state[
        "recommended_order"
    ]

    delivery_agent = st.session_state[
        "delivery_agent"
    ]

    best = st.session_state[
        "best_delivery"
    ]


    # ======================================================
    # GET BOTH ORDERS
    # ======================================================

    route_order_ids = [
        assigned_order,
        recommended_order
    ]


    route_orders = orders_df[
        orders_df["order_id"].isin(
            route_order_ids
        )
    ].copy()


    # ======================================================
    # OPTIMIZE ROUTE
    # ======================================================

    try:

        optimized_route = optimize_delivery_route(
            route_orders
        )


        route_summary = calculate_route_summary(
            optimized_route
        )


        # ==================================================
        # ROUTE SUCCESS
        # ==================================================

        st.success(
            f"🚚 Route successfully assigned to "
            f"**{delivery_agent}**"
        )


        # ==================================================
        # ROUTE KPI CARDS
        # ==================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Delivery Person",
                delivery_agent
            )


        with col2:

            st.metric(
                "Total Orders",
                len(route_orders)
            )


        with col3:

            st.metric(
                "Total Distance",
                f"""
                {route_summary.get(
                    'total_distance_km',
                    0
                ):.2f} km
                """
            )


        with col4:

            st.metric(
                "Estimated Time",
                f"""
                {route_summary.get(
                    'estimated_time_hours',
                    0
                ):.2f} hrs
                """
            )
                    # ==================================================
        # DELIVERY PERSON DASHBOARD
        # ==================================================

        st.markdown(
            "### 👤 Delivery Person Dashboard"
        )

        # --------------------------------------------------
        # CALCULATE DELIVERY PERSON DETAILS
        # --------------------------------------------------

        total_orders = len(route_orders)

        total_weight = pd.to_numeric(
            route_orders["package_weight_kg"],
            errors="coerce"
        ).fillna(0).sum()

        total_distance = route_summary.get(
            "total_distance_km",
            0
        )

        estimated_time = route_summary.get(
            "estimated_time_hours",
            0
        )

        current_status = st.session_state.get(
            "delivery_status",
            "Assigned"
        )

        # --------------------------------------------------
        # STATUS DISPLAY
        # --------------------------------------------------

        status_icons = {
            "Pending": "🟡",
            "Assigned": "🔵",
            "Out for Delivery": "🟠",
            "Delivered": "🟢"
        }

        status_icon = status_icons.get(
            current_status,
            "🟡"
        )

        # --------------------------------------------------
        # DELIVERY PERSON PROFILE
        # --------------------------------------------------

        st.info(
            f"""
            🚚 **Delivery Person:** {delivery_agent}

            **Current Status:** {status_icon} {current_status}

            The delivery person is responsible for the
            consolidated delivery of the selected compatible
            orders.
            """
        )

        # --------------------------------------------------
        # DASHBOARD KPI CARDS
        # --------------------------------------------------

        dash1, dash2, dash3, dash4 = st.columns(4)

        with dash1:

            st.metric(
                "📦 Assigned Orders",
                total_orders
            )

        with dash2:

            st.metric(
                "⚖️ Total Package Weight",
                f"{total_weight:.2f} kg"
            )

        with dash3:

            st.metric(
                "📍 Delivery Stops",
                len(map_points)
                if "map_points" in locals()
                else total_orders
            )

        with dash4:

            st.metric(
                "🛣️ Route Distance",
                f"{total_distance:.2f} km"
            )

        # --------------------------------------------------
        # SECOND ROW
        # --------------------------------------------------

        dash5, dash6, dash7 = st.columns(3)

        with dash5:

            st.metric(
                "⏱️ Estimated Time",
                f"{estimated_time:.2f} hrs"
            )

        with dash6:

            st.metric(
                "👤 Delivery Person",
                delivery_agent
            )

        with dash7:

            st.metric(
                "📦 Delivery Status",
                f"{status_icon} {current_status}"
            )

        # --------------------------------------------------
        # ASSIGNED ORDERS
        # --------------------------------------------------

        st.markdown(
            "#### 📋 Assigned Orders"
        )

        dashboard_orders = route_orders[
            [
                "order_id",
                "customer_id",
                "platform",
                "delivery_location",
                "distance_km",
                "package_weight_kg"
            ]
        ].copy()

        dashboard_orders = dashboard_orders.rename(
            columns={
                "order_id": "Order ID",
                "customer_id": "Customer",
                "platform": "Platform",
                "delivery_location": "Location",
                "distance_km": "Distance (km)",
                "package_weight_kg": "Weight (kg)"
            }
        )

        st.dataframe(
            dashboard_orders,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # DELIVERY SUMMARY
        # --------------------------------------------------

        st.success(
            f"""
            🚚 **{delivery_agent}** has been assigned
            **{total_orders} orders** with a combined package
            weight of **{total_weight:.2f} kg**.

            🛣️ Total planned route distance:
            **{total_distance:.2f} km**

            ⏱️ Estimated delivery time:
            **{estimated_time:.2f} hours**
            """
        )


        # ==================================================
        # OPTIMIZED DELIVERY SEQUENCE
        # ==================================================

        st.markdown(
            "### 🧭 Optimized Delivery Sequence"
        )


        if isinstance(
            optimized_route,
            pd.DataFrame
        ):

            display_columns = [

                col

                for col in [

                    "order_id",
                    "customer_id",
                    "platform",
                    "delivery_location",
                    "distance_km",
                    "package_weight_kg"

                ]

                if col
                in optimized_route.columns

            ]


            st.dataframe(

                optimized_route[
                    display_columns
                ],

                use_container_width=True,

                hide_index=True

            )

        else:

            st.write(
                optimized_route
            )


        # ==================================================
        # INTERACTIVE ROUTE MAP
        # ==================================================

        st.markdown(
            "### 🗺️ Interactive Delivery Route"
        )


        map_points = []


        # --------------------------------------------------
        # CREATE MAP POINTS
        # --------------------------------------------------

        if isinstance(
            optimized_route,
            pd.DataFrame
        ):

            for _, row in (
                optimized_route.iterrows()
            ):

                location = str(
                    row["delivery_location"]
                )


                if (
                    location
                    in LOCATION_COORDINATES
                ):

                    lat, lon = (
                        LOCATION_COORDINATES[
                            location
                        ]
                    )


                    map_points.append({

                        "order_id":
                            row["order_id"],

                        "customer_id":
                            row.get(
                                "customer_id",
                                ""
                            ),

                        "platform":
                            row.get(
                                "platform",
                                ""
                            ),

                        "location":
                            location,

                        "lat":
                            lat,

                        "lon":
                            lon

                    })


        # ==================================================
        # SHOW MAP
        # ==================================================

        if map_points:

            first_point = map_points[0]


            route_map = folium.Map(

                location=[
                    first_point["lat"],
                    first_point["lon"]
                ],

                zoom_start=13,

                control_scale=True

            )


            route_coordinates = []


            # --------------------------------------------------
            # DELIVERY STOPS
            # --------------------------------------------------

            for index, point in enumerate(
                map_points
            ):

                route_coordinates.append(
                    (
                        point["lat"],
                        point["lon"]
                    )
                )


                folium.Marker(

                    location=[
                        point["lat"],
                        point["lon"]
                    ],


                    popup=folium.Popup(

                        f"""
                        <b>🚚 Delivery Stop
                        {index + 1}</b>

                        <br><br>

                        <b>Order:</b>
                        {point["order_id"]}

                        <br>

                        <b>Customer:</b>
                        {point["customer_id"]}

                        <br>

                        <b>Platform:</b>
                        {point["platform"]}

                        <br>

                        <b>Location:</b>
                        {point["location"]}
                        """,

                        max_width=300

                    ),


                    tooltip=(
                        f"Stop {index + 1} - "
                        f"{point['order_id']}"
                    ),


                    icon=folium.DivIcon(

                        html=f"""
                        <div style="
                            background:#2563eb;
                            color:white;
                            border-radius:50%;
                            width:34px;
                            height:34px;
                            text-align:center;
                            line-height:34px;
                            font-weight:bold;
                            border:3px solid white;
                            box-shadow:
                                0 2px 6px
                                rgba(0,0,0,0.3);
                        ">
                            {index + 1}
                        </div>
                        """

                    )

                ).add_to(
                    route_map
                )


            # --------------------------------------------------
            # ROUTE LINE
            # --------------------------------------------------

            if len(
                route_coordinates
            ) > 1:

                folium.PolyLine(

                    route_coordinates,

                    weight=6,

                    opacity=0.8,

                    tooltip=(
                        "🧭 Optimized "
                        "Delivery Route"
                    )

                ).add_to(
                    route_map
                )


            # --------------------------------------------------
            # DELIVERY AGENT MARKER
            # --------------------------------------------------

            folium.Marker(

                location=[
                    first_point["lat"],
                    first_point["lon"]
                ],

                popup=(
                    f"🚚 Delivery Person: "
                    f"{delivery_agent}"
                ),

                tooltip=(
                    "Delivery Route Start"
                ),

                icon=folium.Icon(
                    icon="truck",
                    prefix="fa"
                )

            ).add_to(
                route_map
            )


            # --------------------------------------------------
            # DISPLAY MAP
            # --------------------------------------------------

            st_folium(

                route_map,

                width=None,

                height=550,

                returned_objects=[]

            )


            st.success(

                f"""
                🧭 Optimized route contains
                **{len(map_points)} delivery stops**.
                """

            )


            st.caption(
                "📍 Location_A–E currently use "
                "demo coordinates for map visualization."
            )


        else:

            st.warning(
                "⚠️ No coordinates available "
                "for the selected delivery locations."
            )


        # ==================================================
        # DELIVERY STATUS
        # ==================================================

        st.markdown(
            "### 📦 Delivery Status"
        )


        status = st.session_state.get(
            "delivery_status",
            "Assigned"
        )


        # --------------------------------------------------
        # ASSIGNED
        # --------------------------------------------------

        if status == "Assigned":

            st.info(
                """
                🔵 Orders assigned.
                The delivery person is ready to start.
                """
            )


            if st.button(

                "🚚 START DELIVERY",

                key="start_delivery",

                type="primary",

                use_container_width=True

            ):

                st.session_state[
                    "delivery_status"
                ] = "Out for Delivery"

                st.rerun()


        # --------------------------------------------------
        # OUT FOR DELIVERY
        # --------------------------------------------------

        elif status == "Out for Delivery":

            st.warning(
                """
                🟠 Orders are currently
                out for delivery.
                """
            )


            if st.button(

                "📦 MARK AS DELIVERED",

                key="mark_delivered",

                type="primary",

                use_container_width=True

            ):

                st.session_state[
                    "delivery_status"
                ] = "Delivered"

                st.rerun()


        # --------------------------------------------------
        # DELIVERED
        # --------------------------------------------------

        elif status == "Delivered":

            st.success(
                "🟢 Delivery completed successfully!"
            )


            st.markdown(

                f"""
                ### 🎉 Delivery Completed

                **Delivery Person:**  
                {delivery_agent}

                **Orders Delivered:**  
                {assigned_order} + {recommended_order}

                **Delivery Type:**  
                Collaborative Delivery

                **Status:**  
                🟢 Delivered
                """

            )


            st.balloons()


    # ======================================================
    # ROUTE ERROR
    # ======================================================

    except Exception as e:

        st.error(
            f"⚠️ Route optimization error: {e}"
        )


# ==========================================================
# WAITING FOR ASSIGNMENT
# ==========================================================

else:

    st.info(
        """
        👤 **Delivery route will appear after assignment.**

        Complete:

        **1. Select Order →**
        **2. Find Best Delivery →**
        **3. Select Delivery Person →**
        **4. Confirm Assignment**
        """
    )


# ==========================================================
# PROJECT SUMMARY
# ==========================================================

st.divider()

st.subheader(
    "💡 How the AI Last-Mile Delivery System Works"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        ### 1️⃣ AI Recommendation

        The AI analyzes delivery information
        and recommends suitable delivery
        options based on:

        - Delivery distance
        - Delivery location
        - Package weight
        - Delivery constraints
        """
    )


with col2:

    st.markdown(
        """
        ### 2️⃣ Collaborative Delivery

        Compatible orders are identified using:

        - Customer
        - Location
        - Time window
        - Distance
        - Package capacity

        Compatible orders can be delivered
        together instead of using separate trips.
        """
    )


with col3:

    st.markdown(
        """
        ### 3️⃣ Route Optimization

        The selected orders are assigned
        to one delivery person.

        The system then calculates an efficient
        delivery sequence and tracks the
        delivery status.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "AI Last-Mile Delivery Recommendation System | "
    "Machine Learning + Collaborative Delivery + "
    "Route Optimization"
)