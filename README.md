# AI-Based Collaborative Last-Mile Delivery Recommendation System

An intelligent recommendation system that identifies compatible orders from multiple delivery platforms and recommends collaborative last-mile deliveries to reduce duplicate trips, delivery cost, fuel consumption, traffic, and delivery time.

## Project Overview

Modern e-commerce platforms such as Myntra, Flipkart, Amazon, and other delivery partners may deliver separate orders to customers located in the same area and within similar time windows.

This project proposes an AI-based collaborative last-mile delivery recommendation system that identifies compatible orders and recommends combining their last-mile delivery into a single delivery trip when the orders satisfy predefined compatibility constraints.

### Example

Suppose:

* Order O101 -> Myntra -> Location A -> 14:00-16:00
* Order O102 -> Flipkart -> Location A -> 14:00-17:00

If the orders have compatible:

* Customer/location
* Delivery time window
* Distance
* Package weight
* Vehicle capacity
* Priority

the system can recommend delivering them together.

This can reduce unnecessary duplicate trips and improve last-mile delivery efficiency.

---

## Objectives

* Identify compatible delivery orders from multiple platforms.
* Recommend collaborative delivery groups.
* Reduce duplicate delivery trips.
* Reduce delivery cost and fuel consumption.
* Improve vehicle and delivery-person utilization.
* Consider package weight and vehicle capacity.
* Consider delivery time windows and priority.
* Predict delivery cost and delay probability.
* Optimize the recommended delivery route.

---

## Recommendation Approach

The system combines collaborative and constraint-based recommendation techniques.

### 1. Order Compatibility

Orders are compared using factors such as:

* Customer ID
* Delivery location
* Delivery time window
* Distance
* Package weight
* Priority
* Vehicle capacity

A compatibility score is calculated to determine whether orders can be collaboratively delivered.

### 2. Collaborative Delivery Grouping

Compatible orders are grouped based on their delivery characteristics and location.

Example:

```text
Location A
 |
 +-- O101 - Myntra
 +-- O102 - Flipkart
 +-- O109 - Amazon
 +-- O114 - Other Partner
```

### 3. Machine Learning

Machine-learning models are used to estimate:

* Delivery cost
* Delivery delay probability

These predictions are incorporated into the recommendation process.

### 4. Route Optimization

After compatible orders are identified, the system determines an efficient delivery route for the selected delivery group.

---

## Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Data Visualization

* Matplotlib

### Backend

* FastAPI
* Uvicorn

### Data and Models

* CSV datasets
* Pickle trained models
* Jupyter Notebook

---

## System Workflow

```text
Delivery Orders
       |
       v
Data Preprocessing
       |
       v
Order Compatibility Analysis
       |
       v
Collaborative Order Grouping
       |
       v
Cost and Delay Prediction
       |
       v
Route Optimization
       |
       v
Recommended Delivery Group
       |
       v
Estimated Savings / Delivery Efficiency
```

---

## Project Structure

```text
AI-Based-Collaborative-Last-Mile-Delivery-Recommendation-System/
|
+-- backend/
|   +-- main.py
|
+-- data/
|   +-- delivery_logistics.csv
|   +-- delivery_logistics_clean.csv
|   +-- delivery_ml.csv
|   +-- orders.csv
|   +-- partner_cost.png
|   +-- partner_delay.png
|   +-- partner_rating.png
|
+-- models/
|   +-- api.py
|   +-- cost_model.py
|   +-- delay_model.py
|   +-- delay_model.pkl
|   +-- recommendation.py
|
+-- notebooks/
|   +-- 01_data_understanding.ipynb
|
+-- src/
|   +-- data_check.py
|   +-- eda.py
|   +-- route_optimizer.py
|
+-- api.py
+-- app.py
+-- .gitignore
+-- LICENSE
+-- README.md
```

**Note:** The large `cost_model.pkl` file is excluded from the repository because GitHub does not accept individual files larger than 100 MB. The model-generation code is included in `models/cost_model.py`.

---

## Dataset

The project uses delivery-logistics data containing information related to:

* Delivery partners
* Delivery locations
* Delivery distance
* Package weight
* Vehicle type
* Delivery mode
* Delivery cost
* Delivery delay
* Partner rating
* Delivery conditions

The data is processed and transformed for machine-learning and recommendation tasks.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Lahari09-sree/AI-Based-Collaborative-Last-Mile-Delivery-Recommendation-System.git
cd AI-Based-Collaborative-Last-Mile-Delivery-Recommendation-System
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn streamlit
```

### 5. Start the FastAPI backend

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## Main API Features

The backend provides functionality for:

* Retrieving delivery orders
* Finding recommendations for an order
* Checking order compatibility
* Finding collaborative deliveries
* Creating delivery groups
* Optimizing delivery routes
* Generating delivery recommendations

---

## Expected Benefits

The proposed system aims to:

* Reduce duplicate delivery trips.
* Reduce fuel consumption.
* Reduce transportation cost.
* Improve delivery efficiency.
* Improve vehicle utilization.
* Reduce unnecessary traffic caused by separate delivery trips.
* Support collaborative logistics across multiple delivery platforms.

---

## Sustainable Development Goals

This project supports:

**SDG 11 - Sustainable Cities and Communities**

It is also related to:

**SDG 13 - Climate Action**

by promoting more efficient transportation and reducing unnecessary delivery trips and associated fuel consumption.

---

## Future Enhancements

Future versions can include:

* Real-time GPS tracking.
* Real-time traffic information.
* Dynamic route optimization.
* Real-time order matching.
* Integration with multiple e-commerce platforms.
* Mobile application integration.
* Advanced optimization algorithms.
* Real-time delivery ETA prediction.
* Cloud deployment.
* Automatic retraining of machine-learning models.

---

## Project

**AI-Based Collaborative Last-Mile Delivery Recommendation System**

Developed as an academic project focusing on:

**Artificial Intelligence - Machine Learning - Recommendation Systems - Logistics Optimization - Sustainable Delivery**
