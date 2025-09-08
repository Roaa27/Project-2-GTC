# 🩺 Diabetes Prediction Model (Machine Learning Project)

## 📌 Project Description
I built a machine learning model that predicts whether a patient is diabetic or non-diabetic based on diagnostic measurements.  
The dataset is a version of the Pima Indians Diabetes Dataset.  
This project covered the full ML pipeline: data cleaning, EDA, preprocessing, modeling, evaluation, and deployment of a prediction function.

---

## 📊 Dataset
- File: diabetes.csv  
- Target variable: Outcome  
  - 0 → Non-Diabetic  
  - 1 → Diabetic  

The dataset contains clinical measurements such as Glucose, BloodPressure, BMI, Age, Insulin, etc.

---

## 🛠️ Steps Performed
### 1. Data Exploration
- Checked shape, info, descriptive statistics, and sample rows.  
- Visualized target distribution (Outcome).  
- Plotted histograms, boxplots, and correlation heatmap.  
- Examined pairwise relationships between important features.

### 2. Data Cleaning & Preprocessing
- Detected missing values and suspicious zeros in features (e.g., Glucose, BMI, BloodPressure).  
- Replaced zeros with NaN and applied median imputation.  
- Detected and capped outliers using the IQR method.  
- Split the dataset into training (80%) and testing (20%) sets (stratified).  
- Standardized features using StandardScaler inside pipelines.

### 3. Modeling
I trained and compared three models:
1. Logistic Regression (baseline)  
2. Support Vector Machine (SVM) with GridSearchCV  
3. Random Forest with GridSearchCV  

- Hyperparameter tuning improved SVM and Random Forest performance.  
- All models were evaluated with accuracy, precision, recall, F1-score, and ROC-AUC.

### 4. Evaluation
- Plotted Confusion Matrices and ROC Curves.  
- Compared model accuracies with a bar chart.  
- Extracted Feature Importances from Random Forest.  
- Drew a Learning Curve to study bias/variance.

### 5. Prediction Function
I created a function predict_patient() that:  
- Accepts new patient data (list of features in the same order as the dataset).  
- Returns:
  - Predicted class (0 or 1)  
  - Predicted label ("Diabetic" / "Non-Diabetic")  
  - Probability of being diabetic  

---

## 📈 Results
- Logistic Regression: ~ X% Accuracy  
- Best SVM: ~ Y% Accuracy  
- Best Random Forest: ~ Z% Accuracy (highest performing model)  

*(Replace X, Y, Z with actual numbers from your notebook)*

- Most important features (Random Forest): Glucose, BMI, Age, Insulin.

---

## 📷 Visualizations
The notebook includes:
- Outcome distribution plot  
- Histograms of features by Outcome  
- Correlation heatmap  
- Pairplot of selected features  
- Confusion Matrices  
- ROC Curves  
- Model accuracy comparison bar chart  
- Feature importance bar chart  
- Learning curve plot  

---

## 🚀 Usage
1. Open the Colab notebook.  
2. Upload diabetes.csv.  
3. Run all cells in order.  
4. Use the predict_patient() function to test new patient data. Example:

```python
sample_patient = [6,148,72,35,0,33.6,0.627,50]  # Replace with real values
predict_patient(sample_patient)
