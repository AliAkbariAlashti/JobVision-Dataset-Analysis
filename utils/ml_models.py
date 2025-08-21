import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

from data_processing import *


# =============================================================================
# 4. SALARY PREDICTION MODEL
# =============================================================================

print("\n🎯 Salary Prediction Model Development")

# Prepare data for salary prediction (only jobs with salary info)
salary_df = df[df['has_salary_info'] == 1].copy()
print(f"   • Training on {len(salary_df):,} jobs with salary information")

# Feature selection for prediction
feature_columns = [
    'Jobpost_RequiredExperienceYears',
    'software_skills_count',
    'language_skills_count',
    'programming_languages_count',
    'Jobpost_IsRemote',
    'activation_year'
]

# Add categorical features (encoded)
le_province = LabelEncoder()
le_industry = LabelEncoder()
le_company_size = LabelEncoder()
le_work_type = LabelEncoder()

salary_df.loc[:, 'province_encoded'] = le_province.fit_transform(salary_df['Jobpost_ProvinceEn'])
salary_df.loc[:, 'industry_encoded'] = le_industry.fit_transform(salary_df['Jobpost_IndustryEn'])
salary_df.loc[:, 'company_size_encoded'] = le_company_size.fit_transform(salary_df['company_size_category'])
salary_df.loc[:, 'work_type_encoded'] = le_work_type.fit_transform(salary_df['Jobpost_WorkTypeEn'])

feature_columns.extend(['province_encoded', 'industry_encoded', 'company_size_encoded', 'work_type_encoded'])

# Prepare features and target (use Toman for better numerical stability)
X = salary_df[feature_columns].fillna(0)
y = salary_df['avg_salary_million_toman'].fillna(salary_df['avg_salary_million_toman'].median())

# Remove extreme outliers (beyond 3 standard deviations)
z_scores = np.abs(stats.zscore(y))
mask = z_scores < 3
X = X[mask]
y = y[mask]

print(f"   • After outlier removal: {len(X):,} samples")
print(f"   • Salary range: {y.min():.1f} to {y.max():.1f} million Toman")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

model_results = {}

print("\n🤖 Training Models:")
for name, model in models.items():
    # Train model
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'model': model
    }
    
    print(f"   {name}:")
    print(f"     • RMSE: {rmse:.2f} Million Toman (${rmse*42:.0f})")
    print(f"     • MAE: {mae:.2f} Million Toman (${mae*42:.0f})")
    print(f"     • R²: {r2:.3f}")

# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['R²'])
best_model = model_results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name} (R² = {model_results[best_model_name]['R²']:.3f})")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.show()