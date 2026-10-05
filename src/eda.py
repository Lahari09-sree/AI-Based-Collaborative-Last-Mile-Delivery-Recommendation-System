import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD CLEAN DATASET
# ==========================================

df = pd.read_csv("data/delivery_logistics_clean.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. DELIVERY PARTNER ANALYSIS
# ==========================================

print("\n========== PARTNER PERFORMANCE ==========")

partner_summary = df.groupby("delivery_partner").agg(
    average_cost=("delivery_cost", "mean"),
    average_time=("delivery_time_hours", "mean"),
    average_rating=("delivery_rating", "mean"),
    delay_rate=("delayed_binary", "mean")
)

partner_summary["delay_rate"] *= 100

print(partner_summary.round(2))


# ==========================================
# 3. VEHICLE ANALYSIS
# ==========================================

print("\n========== VEHICLE PERFORMANCE ==========")

vehicle_summary = df.groupby("vehicle_type").agg(
    average_cost=("delivery_cost", "mean"),
    average_time=("delivery_time_hours", "mean"),
    average_distance=("distance_km", "mean"),
    average_weight=("package_weight_kg", "mean"),
    average_rating=("delivery_rating", "mean"),
    delay_rate=("delayed_binary", "mean")
)

vehicle_summary["delay_rate"] *= 100

print(vehicle_summary.round(2))


# ==========================================
# 4. REGION ANALYSIS
# ==========================================

print("\n========== REGION PERFORMANCE ==========")

region_summary = df.groupby("region").agg(
    average_cost=("delivery_cost", "mean"),
    average_time=("delivery_time_hours", "mean"),
    average_distance=("distance_km", "mean"),
    delay_rate=("delayed_binary", "mean"),
    average_rating=("delivery_rating", "mean")
)

region_summary["delay_rate"] *= 100

print(region_summary.round(2))


# ==========================================
# 5. PACKAGE TYPE ANALYSIS
# ==========================================

print("\n========== PACKAGE TYPE PERFORMANCE ==========")

package_summary = df.groupby("package_type").agg(
    average_cost=("delivery_cost", "mean"),
    average_time=("delivery_time_hours", "mean"),
    average_weight=("package_weight_kg", "mean"),
    average_rating=("delivery_rating", "mean"),
    delay_rate=("delayed_binary", "mean")
)

package_summary["delay_rate"] *= 100

print(package_summary.round(2))


# ==========================================
# 6. PARTNER RATING GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

partner_summary["average_rating"].sort_values().plot(
    kind="barh"
)

plt.title("Average Delivery Rating by Partner")
plt.xlabel("Average Rating")
plt.ylabel("Delivery Partner")
plt.tight_layout()

plt.savefig("data/partner_rating.png")

plt.show()


# ==========================================
# 7. PARTNER COST GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

partner_summary["average_cost"].sort_values().plot(
    kind="barh"
)

plt.title("Average Delivery Cost by Partner")
plt.xlabel("Average Cost")
plt.ylabel("Delivery Partner")
plt.tight_layout()

plt.savefig("data/partner_cost.png")

plt.show()


# ==========================================
# 8. PARTNER DELAY RATE
# ==========================================

plt.figure(figsize=(10, 6))

partner_summary["delay_rate"].sort_values().plot(
    kind="barh"
)

plt.title("Delivery Delay Rate by Partner")
plt.xlabel("Delay Rate (%)")
plt.ylabel("Delivery Partner")
plt.tight_layout()

plt.savefig("data/partner_delay.png")

plt.show()


print("\nEDA completed successfully!")