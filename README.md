
# AI-Based Collaborative Last-Mile Delivery Recommendation System

## 1. Project Information

**Project Title:** AI-Based Collaborative Last-Mile Delivery Recommendation System

**Course Code:** CSE4077 – Recommender Systems

**School:** School of Computer Science and Engineering, VIT Chennai

**Programme:** Integrated M.Tech – Computer Science and Engineering (Business Analytics)

**Student Author:** Yamasani Lahari Sree

**Register Number:** 23MIA1129

**Other Team Members:**
- V. Archana
- K. Ashmit

**Faculty / Project Mentor:** Dr S Vigneshwari (54147)

**Slot:** A1 + TA1

---

## 2. Project Overview

The AI-Based Collaborative Last-Mile Delivery Recommendation System is designed to improve the efficiency of last-mile delivery by identifying compatible orders that can be delivered together.

Customers frequently place orders through different e-commerce platforms. When multiple orders have similar delivery locations and compatible delivery requirements, they may be combined into a single delivery group.

The system uses machine learning and recommendation techniques to evaluate delivery compatibility, estimate delivery costs, predict possible delays, and support efficient delivery grouping.

The main objective is to reduce unnecessary delivery trips, transportation costs, fuel consumption, and delivery time while maintaining delivery constraints.

## 3. Problem Statement

Last-mile delivery is one of the challenging stages of the logistics process. Delivering every order separately can increase operational costs, fuel usage, traffic congestion, and delivery time.

Traditional delivery systems may not fully utilize opportunities to combine compatible orders from different delivery partners.

This project proposes an AI-based recommendation system that identifies compatible orders based on location, distance, package weight, delivery time, priority, and other constraints.

## 4. Project Objectives

- Recommend compatible orders for collaborative delivery.
- Group orders with suitable delivery locations and requirements.
- Estimate delivery costs using machine learning.
- Predict the probability of delivery delays.
- Check package weight and delivery compatibility constraints.
- Support delivery route planning.
- Provide API endpoints for accessing delivery recommendations.
- Improve delivery efficiency and reduce unnecessary trips.

## 5. Key Features

### 5.1 Order Management
The system retrieves order information and supports accessing individual orders.

### 5.2 Collaborative Delivery Recommendation
The system identifies orders that may be suitable for collaborative delivery based on their compatibility.

### 5.3 Compatibility Checking
Orders can be evaluated using delivery location, distance, package weight, delivery windows, priority, and other relevant constraints.

### 5.4 Delivery Cost Prediction
A Random Forest regression model is used to estimate delivery costs based on available delivery features.

### 5.5 Delay Prediction
A machine learning model estimates the likelihood of delivery delays.

### 5.6 Delivery Grouping
Compatible orders can be organized into groups to support coordinated deliveries.

### 5.7 Route Planning
The system provides route-related functionality for delivery orders and groups.

### 5.8 API Integration
FastAPI provides endpoints that allow the frontend or other applications to interact with the recommendation system.

## 6. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Backend Framework:** FastAPI
- **API Server:** Uvicorn
- **Data Storage:** CSV datasets
- **Model Storage:** Pickle files
- **Development Tools:** Visual Studio Code, Jupyter Notebook
- **Version Control:** Git and GitHub
- **Frontend:** Use the interface included in the final project files.

## 7. System Workflow

1. Load the delivery order and logistics datasets.
2. Preprocess the available order and delivery information.
3. Extract relevant features such as distance, package weight, delivery partner, delivery mode, and delivery time.
4. Check compatibility between orders using the defined delivery constraints.
5. Generate collaborative delivery recommendations.
6. Estimate delivery cost using the trained cost prediction model.
7. Estimate delivery delay probability using the delay prediction model.
8. Group compatible orders for coordinated delivery.
9. Support route-related operations.
10. Return the results through the API for display in the application.

## 8. Algorithms and Models

### 8.1 Random Forest Regression

Random Forest Regression is used to predict delivery costs. It combines predictions from multiple decision trees to estimate a numerical output.

### 8.2 Delay Prediction Model

A trained machine learning model is used to estimate the probability of delivery delays based on available features.

### 8.3 Collaborative Recommendation

The recommendation process evaluates whether multiple orders can be delivered together while satisfying the project's compatibility constraints.

### 8.4 Constraint-Based Filtering

The system checks delivery requirements such as location, weight, distance, time window, and priority before recommending collaborative delivery.

### 8.5 Route Planning

Route-related functionality supports the planning of deliveries for individual orders and delivery groups.

## 9. Dataset Description

The project uses delivery and logistics data containing order and delivery-related information.

The available dataset and features may include:

- Delivery partner
- Package type
- Vehicle type
- Delivery mode
- Region
- Weather condition
- Distance in kilometres
- Package weight in kilograms
- Delivery time in hours
- Delivery rating
- Delivery cost

The project also includes order records used for testing order compatibility, recommendation generation, and delivery grouping.

**Note:** The exact dataset filenames, column names, and number of records should match the final files included in the submission.

## 10. Project Structure

The project may contain the following directories and files. Update this structure to match the actual final project folder.

```text
AI-Based-Collaborative-Last-Mile-Delivery-Recommendation-System/
│
├── README.md
├── app.py
├── api.py
├── backend/
│   └── main.py
│
├── data/
│   ├── delivery-logistics-dataset-india-multi-partner/
│   ├── delivery_ml.csv
│   └── orders.csv
│
├── models/
│   ├── cost_model.py
│   ├── cost_model.pkl
│   └── delay_model.pkl
│
├── src/
├── notebooks/
├── figures/
├── tables/
├── requirements.txt
└── .gitignore
```

Some files may be located in different directories or may not exist in every version of the project.

## 11. Installation and Setup

### Prerequisites

Install the following software:

- Python 3.11 or another compatible Python version
- pip
- Git (optional, for repository management)

### Step 1: Open the Project Folder

Open PowerShell or the terminal in the project directory.

### Step 2: Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 3: Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

If a `requirements.txt` file is available, run:

```powershell
pip install -r requirements.txt
```

If the project does not contain a requirements file, install the required packages according to the imports used in the source code.

### Step 5: Start the Backend API

From the project root, run the following command if the backend module is located at `backend/main.py`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

If your backend entry point is different, use the command corresponding to the actual project structure.

### Step 6: Access the API

Open the following addresses in a browser:

- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

These addresses work when the backend is running on the same computer.

## 12. API Endpoints

The backend provides endpoints for interacting with delivery orders and recommendations.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Check API health |
| GET | `/orders` | Retrieve orders |
| GET | `/orders/{order_id}` | Retrieve a specific order |
| GET | `/recommend/options` | Retrieve recommendation options |
| POST | `/recommend` | Generate a recommendation |
| POST | `/orders/{order_id}/recommend` | Recommend delivery options for an order |
| GET | `/compatibility/{order1_id}/{order2_id}` | Check compatibility between two orders |
| GET | `/collaborative` | Access collaborative recommendation functionality |
| GET | `/orders/{order_id}/collaborative` | Get collaborative recommendations for an order |
| GET | `/groups` | Retrieve delivery groups |
| GET | `/groups/{group_id}` | Retrieve a specific delivery group |
| GET | `/groups/{group_id}/route` | Retrieve route information for a group |
| GET | `/orders/{order_id}/route` | Retrieve route information for an order |

**Note:** Confirm the available endpoints in `/docs` because endpoint methods and paths can change between versions.

## 13. Model Evaluation

The delivery cost prediction model can be evaluated using the following regression metrics.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted delivery costs.

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while assigning greater weight to larger errors.

### R-squared (R²)

R² indicates how much of the variation in the target variable is explained by the model.

Model evaluation should be performed using a separate test dataset or an appropriate cross-validation method. Data leakage should be checked before reporting final results.

The final report should include the verified evaluation results for the cost and delay prediction models.

## 14. Expected Outcomes

The proposed system aims to:

- Identify suitable orders for collaborative delivery.
- Improve the utilization of delivery resources.
- Estimate delivery costs.
- Identify potentially delayed deliveries.
- Support delivery grouping and route planning.
- Reduce unnecessary trips when compatible orders can be combined.
- Provide a foundation for more efficient last-mile logistics.

Actual cost savings, time reductions, and fuel savings must be measured through experiments before they are reported as achieved results.

## 15. Limitations

- Recommendation quality depends on the quality and completeness of the input data.
- Incorrect location or delivery-time information can affect compatibility decisions.
- Model performance depends on the dataset and training procedure.
- Real-world route optimization may require live traffic, road-network, and vehicle information.
- Actual delivery savings require validation using realistic operational data.
- Trained model files may need to be generated separately if they are not included in the repository.

## 16. Future Enhancements

- Integrate real-time traffic and map services.
- Improve route optimization using advanced optimization algorithms.
- Add real-time delivery tracking.
- Expand the dataset with additional logistics records.
- Compare multiple machine learning models using consistent test data.
- Add a mobile application for delivery partners.
- Deploy the backend and frontend for remote access.
- Measure cost, distance, fuel, and time savings through controlled experiments.

## 17. Demo Link

**Demo URL:** Add the actual deployed application URL here after deployment.

**API Documentation:** http://localhost:8000/docs (local access only when the backend is running).

If a public demo is not yet available, state that the project is demonstrated locally rather than adding an unverified URL.

## 18. Reproducibility

To reproduce the project:

1. Obtain the source code and required datasets.
2. Install the dependencies.
3. Generate any missing trained model files using the provided training scripts.
4. Start the backend API.
5. Open the API documentation or launch the project's frontend.
6. Test order retrieval, compatibility checking, collaborative recommendations, delivery groups, and route-related endpoints.
7. Record the test results and screenshots for the project report.

## 19. Acknowledgement

We express our sincere gratitude to **Dr S Vigneshwari (54147)** for her guidance and support throughout the development of this project under the course **CSE4077 – Recommender Systems** at VIT Chennai.

We also acknowledge the use of Python and its machine learning and data-processing libraries in developing the proposed system.

## 20. Authors

- **Yamasani Lahari Sree** — 23MIA1129
- **V. Archana**
- **K. Ashmit**

**Faculty / Project Mentor:** Dr S Vigneshwari (54147)

**Institution:** Vellore Institute of Technology, Chennai
