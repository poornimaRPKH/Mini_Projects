# Predictive Analysis of Titanic Passenger Survival

## 1. Problem Definition

### 1.1 Business / Real-World Problem Statement
On April 15, 1912, the Titanic sank after colliding with an iceberg, resulting in the deaths of over 1,500 passengers and crew. However, survival was not purely random — factors like gender, age, ticket class, and fare played a significant role in determining who survived. Understanding these patterns helps us learn how socio-economic and demographic factors influence survival outcomes during disasters, which has broader applications in emergency response planning, risk assessment, and resource allocation during crisis situations.

### 1.2 Project Objectives
- Analyze passenger demographic and travel-related patterns (age, gender, class, fare, family size).
- Identify key factors that influenced survival during the Titanic disaster.
- Build predictive models to classify whether a passenger survived or not.
- Compare multiple machine learning classification models (Logistic Regression, Decision Tree, Random Forest, KNN) to find the best-performing one.
- Visualize survival trends to present clear, data-driven insights.
- Deploy the best-performing model using Flask and package it with Docker.

### 1.3 Machine Learning Problem Type
- [x] Supervised Learning – Classification

**Why "Supervised Learning"?**
Supervised learning means we have labeled data — for every passenger in our dataset, we already know the answer (Survived = 0 or 1). Since the "correct answers" already exist historically, we can teach the model: "Here are passenger details (age, sex, class, fare...) → and here's whether they survived or not." The model learns the relationship between inputs (features) and the known output (label).

**Why "Classification" (not Regression)?**
- Regression is used when the target variable is a continuous number (like predicting exact Fare amount).
- Classification is used when the target variable is a category/discrete label.

In our case, `Survived` only has 2 possible values: 0 or 1 (Died or Survived) — this is called **Binary Classification**.

**Summary:** Since we already know which passengers survived in the historical data, and the outcome we're predicting has only two categories, this is a Supervised Classification problem.

**Algorithms suited for this problem type:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## 2. Dataset Understanding

### 2.1 Dataset Source
- **Source:** https://github.com/awesomedata/awesome-public-datasets/blob/master/Datasets/titanic.csv.zip
- **Original Data Provider:** Department of Biostatistics, Vanderbilt University

### 2.2 Dataset Description
- **Number of rows:** 891 passengers
- **Number of columns:** 12
- **Dataset purpose:** To study passenger demographic and travel details to predict survival outcomes from the Titanic disaster (April 1912).
- **Data collection source:** Compiled from real Titanic passenger manifests and survival records, digitized and maintained by Vanderbilt University for academic/statistical research use.

### 2.3 Feature Description

| Feature Name | Data Type | Description |
|---|---|---|
| PassengerId | Numeric | Unique identifier assigned to each passenger |
| Pclass | Categorical (Ordinal) | Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd |
| Name | Text | Full name of the passenger |
| Sex | Categorical | Gender of passenger (male/female) |
| Age | Numeric | Age of passenger in years (contains missing values) |
| SibSp | Numeric | Number of siblings/spouses aboard the Titanic |
| Parch | Numeric | Number of parents/children aboard the Titanic |
| Ticket | Text | Ticket number |
| Fare | Numeric | Passenger fare paid (in British pounds) |
| Cabin | Categorical | Cabin number (majority missing values) |
| Embarked | Categorical | Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton |

### 2.4 Target Variable
- **Name:** Survived
- **Description:** Binary indicator of survival outcome — 0 = Did not survive, 1 = Survived
- **Prediction Goal:** To classify whether a given passenger survived the Titanic disaster based on their demographic and travel details.

## 3. Exploratory Data Analysis (EDA)

### Visualizations Completed (8 total)

| # | Visualization | Type |
|---|---|---|
| 1 | Survival Count | Univariate |
| 2 | Age Distribution | Univariate |
| 3 | Fare Distribution | Univariate |
| 4 | Survival by Sex | Bivariate |
| 5 | Survival by Pclass | Bivariate |
| 6 | Survival by Embarked | Bivariate |
| 7 | Pclass + Sex + Survived | Multivariate |
| 8 | Age Distribution by Survival + Correlation Heatmap | Multivariate |

### Key EDA Insights
Survival on the Titanic was mainly influenced by **Sex, Passenger Class, and Fare** — women, 1st class passengers, and young children had the highest survival rates, while missing data and fare outliers were addressed before modeling.

## 4. Data Preprocessing
- **Missing Values:** Age filled with median, Embarked filled with mode, Cabin column dropped (77% missing)
- **Duplicates:** Checked — no duplicate rows found
- **Outliers:** Removed using IQR method on Fare column (891 → 775 rows)
- **Skewness:** Applied log transformation on Fare (skewness improved from 1.43 to -0.51)
- **Encoding:** Sex converted to 0/1; Embarked one-hot encoded (Embarked_Q, Embarked_S)
- **Feature Scaling:** Age and Fare scaled using StandardScaler

## 5. Feature Engineering & Selection
- Dropped non-predictive columns: PassengerId, Name, Ticket
- **Correlation Analysis:** Sex (0.5), Fare (0.25), Pclass (-0.24) showed strongest correlation with Survived
- **SelectKBest Scores:** Sex (257.5), Fare (50.4), Pclass (46.6), Age (10.9), Embarked_S (8.2), Parch (7.6), Embarked_Q (0.8), SibSp (0.01)
- **Selected Features:** Sex, Fare, Pclass, Age, SibSp, Parch, Embarked_Q, Embarked_S

## 6. Model Building & Evaluation

Four classification models were built and compared:

| Model | Accuracy | Precision (Survived) | Recall (Survived) | F1-Score (Survived) |
|---|---|---|---|---|
| Logistic Regression | 75% | 71% | 58% | 64% |
| **Decision Tree** | **78%** | 74% | **67%** | **70%** |
| Random Forest | 76% | 71% | 62% | 66% |
| KNN | 76% | 73% | 60% | 66% |

### Top-Performing Model
**Decision Tree** achieved the highest accuracy (78%) and best F1-score (70%) for predicting survivors, offering the best balance between precision and recall.

### Overfitting/Underfitting Observations
Decision Trees can be prone to overfitting (memorizing training data). Random Forest, which combines multiple trees, is generally more robust against overfitting, though it scored slightly lower here — likely due to the moderate dataset size (775 rows).

### Bias-Variance Considerations
- **Logistic Regression:** Higher bias, lower variance — simple and consistent but less flexible.
- **Decision Tree:** Lower bias, higher variance — flexible but sensitive to data changes.
- **Random Forest:** Balances bias and variance by averaging multiple trees.
- **KNN:** Performance depends on scaling and choice of K; gave balanced, middle-ground results.

### Conclusion
Based on accuracy, precision, recall, and F1-score, **Decision Tree Classifier** was selected as the best-performing model and is used for deployment.

## 7. How to Run This Project

### Setup
```bash
pip install -r requirements.txt
```

### Run the notebook
Open `Project_Notebook.ipynb` in VS Code or Google Colab and run all cells.

### Run the deployment app (after model is trained)
```bash
python app.py
```

## 8. Project Structure
```
Titanic_Survival_Prediction/
│
├── Project_Notebook.ipynb   # EDA + ML model building
├── app.py                   # Flask app for predictions
├── model.pkl                # Saved trained ML model
├── requirements.txt         # Python dependencies
├── Dockerfile                # Docker containerization instructions
├── README.md                 # Project documentation (this file)
└── Screenshots/               # Output screenshots for report
```
## 9. Deployment (Flask API)

The best-performing model (Decision Tree Classifier) was deployed using Flask as a REST API.

### Running the Flask app locally
```bash
pip install -r requirements.txt
python app.py
```
The app will start at: `http://127.0.0.1:5000`

### API Usage

**Endpoint:** `/predict`
**Method:** POST
**Content-Type:** application/json

**Sample Request Body:**
```json
{
  "Pclass": 1,
  "Sex": 1,
  "Age": 0.5,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 2.5,
  "Embarked_Q": false,
  "Embarked_S": true
}
```

**Sample Response:**
```json
{
  "Survived": 1
}
```

**Test using PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"Pclass": 1, "Sex": 1, "Age": 0.5, "SibSp": 0, "Parch": 0, "Fare": 2.5, "Embarked_Q": false, "Embarked_S": true}'
```

## 10. Docker Containerization

The application was containerized using Docker for consistent deployment across environments.

### Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### Build the Docker image
```bash
docker build -t titanic-app .
```

### Run the Docker container
```bash
docker run -p 5000:5000 titanic-app
```

Once running, the API is accessible at `http://127.0.0.1:5000/predict`, same as the local Flask version.

### Screenshots
See `Screenshots/` folder for:
- `flask_app_running.png` — Flask server running locally
- `docker_api_response.png` — Successful prediction response from the Dockerized API