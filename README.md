# Salifort Motors Employees Churn Prediction And Analysis 

# Project Summary
Employee turnover represents a significant cost in recruitment, training, and lost institutional knowledge. This project analyzes the human resources dataset for Salifort Motors to uncover the underlying patterns of employee churn. By deploying a suite of machine learning classification models, this analysis not only predicts *which* employees are at risk of leaving but also isolates the *key factors* driving their decisions.

# Project Objective
* **Identify Churn factors:** Pinpoint the most significant variables contributing to employee turnover (e.g., compensation, tenure, department, working hours).
* **Predictive Modeling:** Build a robust, scalable system to flag high-risk employees before they exit.
* **Actionable Insights:** Provide recommendations in order to take pre-emptive measures.

# Methodolody And ML models
1. **Logistic Regression:** Establishes a baseline performance and provides interpretable linear relationships between features and churn.
2. **Decision Tree Classifier:** Captures non-linear patterns and offers clear, rule-based decision paths however, can overfit the data.
3. **Random Forest:** An ensemble method utilized to reduce variance, prevent overfitting, and improve generalization.
4. **XGBoost:** A high-performance gradient boosting algorithm optimized for complex pattern recognition and maximum predictive accuracy.

# Exploratory Data Analysis 
1. Boxplot And Histogram
2. Feature Importance

## Model Performance and Evaluation 
The models were evaluated using standard classification metrics: Accuracy, Precision, Recall, and the F1-Score. In the context of employee churn, a heavy emphasis was placed on **Recall** to minimize false negatives (failing to identify an employee who is actually a flight risk).

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | 0.98 | 0.97 | 0.92 | 0.94 | 0.96 |
| **Random Forest** | 0.98 | 0.98 | 0.90 | 0.94 | 0.97 |
| **XGBoost** | **0.98** | **0.98** | **0.90** | **0.95** | **0.97** |

> **Conclusion:** The XGBoost model outperformed the baseline algorithms, successfully capturing complex relationships in the HR data while maintaining high precision.

# Tech Stack And Local Setup
**Core Technologies:** Python, Pandas, Scikit-Learn, XGBoost, Matplotlib, Seaborn

### Quick Start Guide
1. Clone the repository:
   ```bash
   git clone [https://github.com/anon-public/salifort_motors_analysis.git](https://github.com/anon-public/salifort_motors_analysis.git)
   ```
2. Install the required dependencies
```bash
 pip install -r requirements.txt
```
3. Run the jupiter notebook to view the analysis.






