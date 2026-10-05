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
+-- requirements.txt
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

### 3. Install dependencies

The required Python packages are listed in `requirements.txt`.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Note:** Activating the virtual environment is optional. If PowerShell blocks `.venv\Scripts\activate`, you can directly use `.venv\Scripts\python.exe` as shown above.

### 4. Generate the cost prediction model

The trained `models/cost_model.pkl` file is excluded from GitHub because it is larger than GitHub's 100 MB file limit.

Generate it locally using:

```powershell
.venv\Scripts\python.exe models\cost_model.py
```

This trains the Random Forest regression model and creates:

```text
models/cost_model.pkl
```

### 5. Start the FastAPI backend

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

If the health check returns `healthy`, the backend and ML recommendation system are running successfully.

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

### API Endpoints

| Endpoint                                 | Method | Purpose                                    |
| ---------------------------------------- | ------ | ------------------------------------------ |
| `/`                                      | GET    | API information                            |
| `/health`                                | GET    | Backend health check                       |
| `/orders`                                | GET    | Retrieve all delivery orders               |
| `/orders/{order_id}`                     | GET    | Retrieve a specific order                  |
| `/recommend/options`                     | GET    | View recommendation options                |
| `/recommend`                             | POST   | Generate delivery recommendations          |
| `/orders/{order_id}/recommend`           | POST   | Recommend delivery for an order            |
| `/compatibility/{order1_id}/{order2_id}` | GET    | Check compatibility between two orders     |
| `/collaborative`                         | GET    | Find collaborative delivery pairs          |
| `/orders/{order_id}/collaborative`       | GET    | Find collaborative deliveries for an order |
| `/groups`                                | GET    | Create collaborative delivery groups       |
| `/groups/{group_id}`                     | GET    | View a delivery group                      |
| `/groups/{group_id}/route`               | GET    | Optimize group delivery route              |
| `/orders/{order_id}/route`               | GET    | Generate route for an order                |

---

## Machine Learning Model

The project uses machine learning to predict delivery-related outcomes.

### Cost Prediction

A **Random Forest Regression** model is used to predict delivery cost.

Input features include:

* Delivery partner
* Package type
* Vehicle type
* Delivery mode
* Region
* Weather condition
* Distance
* Package weight
* Delivery time
* Delivery rating

The model is trained using the delivery-logistics dataset.

### Delay Prediction

A trained machine-learning model is used to estimate the probability of delivery delay.

The predicted cost and delay probability are incorporated into the recommendation score.

### Model Performance

The current cost prediction model achieved approximately:

```text
MAE  : 2.32
RMSE : 3.23
R²   : 0.9999
```

These values are based on the current dataset and training configuration.

---

## Recommendation Scoring

The system evaluates delivery options using multiple factors:

```text
Recommendation Score
        |
        +-- Predicted Delivery Cost
        |
        +-- Expected Delivery Time
        |
        +-- Delivery Rating
        |
        +-- Delay Probability
```

Different priorities can be selected:

* **Fast** - gives more importance to delivery time.
* **Economy** - gives more importance to delivery cost.
* **Balanced** - considers cost, time, rating, and delay together.

A lower recommendation score indicates a better delivery option.

---

## Collaborative Delivery Logic

Two orders can be considered compatible when important constraints are satisfied.

The system considers:

```text
Same Customer
      +
Different Delivery Platforms
      +
Same Location
      +
Compatible Delivery Window
      +
Acceptable Distance
      +
Vehicle Capacity
      =
Collaborative Delivery Candidate
```

The system also considers package weight and a maximum package capacity of 10 kg for collaborative delivery grouping.

---

## Delivery Grouping

Compatible orders are combined into delivery groups.

For example:

```text
Delivery Group: GROUP-001

Customer: C001
Location: Location_A

Orders:
    O101 -> Myntra
    O102 -> Flipkart
    O109 -> Amazon
    O114 -> Other Partner

Total Orders: 4
Total Package Weight: 3.6 kg
```

A delivery person is assigned to the collaborative group.

Example:

```text
GROUP-001
   |
   +-- DP-001
```

This allows multiple compatible orders to be delivered as one collaborative last-mile trip.

---

## Route Optimization

After creating a collaborative delivery group, the system generates a delivery route.

The route calculation considers the recorded delivery distance of the orders and produces a route summary containing information such as:

* Total distance
* Number of deliveries
* Estimated delivery time
* Average delivery speed

This can be extended in future versions using real-time GPS and traffic information.

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
