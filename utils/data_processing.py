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

# Statistical analysis
from scipy import stats
import ast

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("🚀 JobVision Dataset Analysis Notebook")
print("📊 Analyzing Iran's Labor Market Data")
print("=" * 50)

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

# Load the dataset
df = pd.read_csv('jobvision_dataset.csv')  # Replace with your actual file path

print(f"📋 Dataset Overview:")
print(f"   • Total job posts: {len(df):,}")
print(f"   • Features: {df.shape[1]}")
print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Basic info about the dataset
print("\n🔍 Dataset Info:")
print(df.info())

# =============================================================================
# 2. DATA CLEANING AND FEATURE ENGINEERING
# =============================================================================

def parse_json_column(df, column_name):
    """Parse JSON strings in columns safely"""
    parsed_data = []
    for idx, value in enumerate(df[column_name]):
        if pd.isna(value):
            parsed_data.append(None)
        else:
            try:
                parsed_data.append(ast.literal_eval(value))
            except:
                parsed_data.append(None)
    return parsed_data

def extract_skill_counts(skill_data):
    """Extract skill counts from JSON data"""
    if skill_data is None:
        return 0
    if isinstance(skill_data, list):
        return len(skill_data)
    return 0

def extract_programming_languages(skill_data):
    """Extract programming languages from software skills"""
    programming_langs = ['Python', 'JavaScript', 'Java', 'C#', 'PHP', 'React', 'Angular', 'Vue', 'Node.js']
    if skill_data is None:
        return []
    
    found_langs = []
    if isinstance(skill_data, list):
        for skill in skill_data:
            if isinstance(skill, dict):
                title_en = skill.get('TitleEn', '').lower()
                title_fa = skill.get('TitleFa', '').lower()
                for lang in programming_langs:
                    if lang.lower() in title_en or lang.lower() in title_fa:
                        found_langs.append(lang)
    return found_langs

# Feature engineering
print("\n🔧 Feature Engineering...")

# Parse JSON columns
df['parsed_software_skills'] = parse_json_column(df, 'Jobpost_SoftwareSkills')
df['parsed_language_skills'] = parse_json_column(df, 'Jobpost_LanguageSkills')

# Create skill count features
df['software_skills_count'] = df['parsed_software_skills'].apply(extract_skill_counts)
df['language_skills_count'] = df['parsed_language_skills'].apply(extract_skill_counts)

# Extract programming languages
df['programming_languages'] = df['parsed_software_skills'].apply(extract_programming_languages)
df['programming_languages_count'] = df['programming_languages'].apply(len)

# Create salary features
df['has_salary_info'] = df['Jobpost_SalaryCanBeShown'].astype(int)
df['salary_range'] = df['Jobpost_MaxSalary'] - df['Jobpost_MinSalary']
df['avg_salary'] = (df['Jobpost_MinSalary'] + df['Jobpost_MaxSalary']) / 2

# Keep salary in Toman for better model performance (millions of Toman)
df['avg_salary_million_toman'] = df['avg_salary'] / 1000000
# Also create USD version for reference (1 USD ≈ 42,000 Toman as of 2024)
df['avg_salary_usd'] = df['avg_salary'] / 42000

# Create experience bins
df['experience_level'] = pd.cut(df['Jobpost_RequiredExperienceYears'], 
                               bins=[-1, 0, 2, 5, 10, float('inf')],
                               labels=['Entry Level', 'Junior', 'Mid-level', 'Senior', 'Expert'])

# Create company size categories
size_mapping = {
    'Less than 10 employees': 'Startup',
    '11 - 50 employees': 'Small',
    '51 - 200 employees': 'Medium',
    '201 - 500 employees': 'Large',
    '501 - 1000 employees': 'Enterprise',
    '1001 - 5000 employees': 'Corporate',
    'More than 5000 employees': 'Multinational'
}
df['company_size_category'] = df['Company_SizeEn'].map(size_mapping)

# Extract year and month from activation time
df['activation_year'] = df['Jobpost_ActivationTime_YEAR_MONTH'].str[:4].astype(int)
df['activation_month'] = df['Jobpost_ActivationTime_YEAR_MONTH'].str[5:].astype(int)

print(f"✅ Feature engineering completed!")
print(f"   • New features created: {df.shape[1] - 43}")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n📊 Exploratory Data Analysis")

# Basic statistics
print("\n📈 Basic Statistics:")
print(f"   • Jobs with salary info: {df['has_salary_info'].sum():,} ({df['has_salary_info'].mean()*100:.1f}%)")
print(f"   • Remote jobs: {df['Jobpost_IsRemote'].sum():,} ({df['Jobpost_IsRemote'].mean()*100:.1f}%)")
print(f"   • Internships: {df['Jobpost_IsInternship'].sum():,} ({df['Jobpost_IsInternship'].mean()*100:.1f}%)")
print(f"   • Jobs requiring military service: {df['Jobpost_RequiredMilitaryServiceCard'].sum():,}")

# Create visualization plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Job distribution by province
top_provinces = df['Jobpost_ProvinceEn'].value_counts().head(10)
top_provinces.plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Top 10 Provinces by Job Count')
axes[0,0].set_xlabel('Province')
axes[0,0].set_ylabel('Number of Jobs')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Experience level distribution
df['experience_level'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
axes[0,1].set_title('Distribution by Experience Level')

# 3. Work type distribution
df['Jobpost_WorkTypeEn'].value_counts().plot(kind='bar', ax=axes[1,0], color='lightcoral')
axes[1,0].set_title('Work Type Distribution')
axes[1,0].set_xlabel('Work Type')
axes[1,0].set_ylabel('Count')

# 4. Company size distribution
df['company_size_category'].value_counts().plot(kind='bar', ax=axes[1,1], color='lightgreen')
axes[1,1].set_title('Company Size Distribution')
axes[1,1].set_xlabel('Company Size')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
