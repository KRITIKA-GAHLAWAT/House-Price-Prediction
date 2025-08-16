# House-Price-Prediction
# 🏡 House Price Prediction using Machine Learning

This project builds a **Machine Learning regression model** to predict house prices using the **Ames Housing dataset**.  
It applies **Exploratory Data Analysis (EDA), feature engineering, multiple ML models, hyperparameter tuning, and evaluation** to find the best-performing model.

---

## 📂 Dataset
- Source: [Ames Housing Dataset](https://raw.githubusercontent.com/kirenz/datasets/master/ames.csv)  
- Target variable: `Sale_Price`  
- Features: 80+ attributes describing house size, location, and quality.

---

## ⚙️ Workflow
1. **Data Loading**  
   Load the Ames dataset from a public GitHub source.

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of house prices  
   - Correlation heatmap of top features  

3. **Data Preprocessing**  
   - Drop columns with >40% missing values  
   - Fill missing values (median for numeric, mode for categorical)  
   - One-hot encoding for categorical variables  
   - Standard scaling for numerical features  

4. **Model Training**  
   - Linear Regression  
   - Ridge & Lasso Regression  
   - Random Forest  
   - XGBoost  

5. **Evaluation Metrics**  
   - Mean Absolute Error (MAE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score  

6. **Hyperparameter Tuning (GridSearchCV)**  
   - Ridge Regression  
   - Lasso Regression  
   - Random Forest  
   - XGBoost  

7. **Residual Analysis**  
   - Predicted vs Actual prices  
   - Residual distribution plot  

8. **Model Saving**  
   - Best model saved as `.joblib` in `artifacts/` folder  

9. **Prediction Demo**  
   - Run inference on a few test samples  

---

## 📊 Results
- Compares performance of all models.  
- Performs hyperparameter tuning to select the **best model**.  
- Example output (Predicted vs Actual):  

| Predicted Price | Actual Price |
|-----------------|--------------|
| 187500          | 185000       |
| 213000          | 215000       |
| 245000          | 240000       |

---

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
Open Google Colab or Jupyter Notebook and run the script:

bash
Copy
Edit
pip install -r requirements.txt
Run all cells in the notebook.

📦 Requirements
pandas

numpy

scikit-learn

matplotlib

seaborn

xgboost

joblib

Install via:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib

🏆 Key Takeaways

Data preprocessing is crucial for handling missing values and categorical variables.

Regularization (Ridge, Lasso) improves linear regression models.

Ensemble methods (Random Forest, XGBoost) significantly outperform linear models.

Hyperparameter tuning further improves performance.

