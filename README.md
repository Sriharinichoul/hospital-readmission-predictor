# 🏥 Hospital Readmission Risk Predictor

## 🔍 Objective
This project builds a machine learning-based tool to **predict the likelihood of a patient being readmitted within 30 days** of hospital discharge. The goal is to support healthcare providers in identifying high-risk patients and improve care planning and follow-up strategies.

---

## 📁 Project Structure

hospital-readmission-predictor/
│
├── hospital-readmission-prediction.ipynb # Full pipeline: preprocessing, training, evaluation
├── hospital_readmissions_30k.csv # Source dataset (30k+ records)
├── Logistic Regression_readmission_model.pkl # Saved best model (logistic regression)
├── scaler.pkl # Saved StandardScaler for preprocessing
├── README.md # Project documentation



---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hospital-readmission-predictor.git
cd hospital-readmission-predictor
2. Install Required Packages
bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn openpyxl
3. Run Training Notebook
Open and execute: hospital-readmission-prediction.ipynb

This will:

Preprocess the data

Train and evaluate 8 different models

Save the best model and scaler

Generate a risk assessment report

4. Predict for New Patients
Open and execute: Single_Patient_Prediction.ipynb

This notebook:

Loads the saved model and scaler

Accepts new patient data

Outputs the risk score and classification

📊 Features and ML Techniques
✅ Skills Demonstrated
AI/ML: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM, KNN, MLP

Critical Thinking: Identified key risk factors, managed missing data, considered real-world deployment implications

Problem Solving: Resolved class imbalance using SMOTE, selected relevant features, scaled inputs

Architecture:

Patient Data → Preprocessing → Feature Engineering → Model Training → Risk Scoring → Report Generation

📈 Evaluation Metrics
Accuracy

Precision, Recall, F1 Score

ROC-AUC Score

Confusion Matrix, ROC Curves

🩺 Risk Assessment Logic
Patients are assigned a risk score between 0–1 (probability of readmission)

Categorized into:

Low Risk: score < 0.33

Medium Risk: 0.33 ≤ score < 0.66

High Risk: score ≥ 0.66

Results exported to Excel with color-coded risk levels

📁 Sample Output (Excel Report)
Patient ID	Risk Score	Risk Level	Predicted Readmission	High Risk Flag
101	0.81	High	1	✅
102	0.27	Low	0	❌

⚖️ Ethical Considerations
Use of this model is intended only for research or academic purposes.

It should not be deployed in a clinical environment without further validation and ethical oversight.

Consider risks of false positives/negatives and their impact on patient care.