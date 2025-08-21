from data_processing import df, salary_df, model_results, best_model_name
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
# 5. LABOR MARKET ANALYSIS
# =============================================================================

print("\n🏭 Labor Market Analysis")

# 5.1 Regional skill gaps analysis
def analyze_regional_skills():
    print("\n🌍 Regional Skill Gap Analysis:")
    
    # Focus on major provinces
    major_provinces = df['Jobpost_ProvinceEn'].value_counts().head(5).index
    
    skill_demand = {}
    for province in major_provinces:
        province_df = df[df['Jobpost_ProvinceEn'] == province]
        
        # Count programming languages
        all_langs = []
        for langs in province_df['programming_languages'].dropna():
            all_langs.extend(langs)
        
        lang_counts = pd.Series(all_langs).value_counts()
        skill_demand[province] = lang_counts
        
        print(f"\n📍 {province}:")
        if len(lang_counts) > 0:
            print(f"   Top skills: {', '.join(lang_counts.head(3).index.tolist())}")
            print(f"   Total tech jobs: {len(province_df[province_df['programming_languages_count'] > 0])}")

analyze_regional_skills()

# 5.2 Salary trends by industry and experience
def analyze_salary_trends():
    print("\n💰 Salary Trends Analysis:")
    
    # Salary by industry (convert back to USD for display)
    industry_salary = salary_df.groupby('Jobpost_IndustryEn')['avg_salary_million_toman'].agg(['mean', 'count']).reset_index()
    industry_salary['mean_usd'] = industry_salary['mean'] * 42  # Convert to USD
    industry_salary = industry_salary[industry_salary['count'] >= 50]  # Filter for statistical significance
    industry_salary = industry_salary.sort_values('mean_usd', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=industry_salary, x='mean_usd', y='Jobpost_IndustryEn')
    plt.title('Average Salary by Industry (USD)')
    plt.xlabel('Average Salary (USD)')
    plt.show()
    
    # Salary by experience level
    exp_salary = salary_df.groupby('experience_level')['avg_salary_million_toman'].agg(['mean', 'median', 'count'])
    exp_salary['mean_usd'] = exp_salary['mean'] * 42
    exp_salary['median_usd'] = exp_salary['median'] * 42
    print("\n📊 Salary by Experience Level:")
    print(exp_salary)
    
    plt.figure(figsize=(10, 6))
    salary_df_plot = salary_df.copy()
    salary_df_plot['avg_salary_usd_display'] = salary_df_plot['avg_salary_million_toman'] * 42
    sns.boxplot(data=salary_df_plot, x='experience_level', y='avg_salary_usd_display')
    plt.title('Salary Distribution by Experience Level')
    plt.xticks(rotation=45)
    plt.ylabel('Salary (USD)')
    plt.show()

analyze_salary_trends()

# =============================================================================
# 6. GEOSPATIAL VISUALIZATION
# =============================================================================

print("\n🗺️ Geospatial Analysis")

def create_geospatial_analysis():
    # Job density by province
    province_stats = df.groupby('Jobpost_ProvinceEn').agg({
        'RowNumber': 'count',
        'avg_salary_million_toman': 'mean',
        'Jobpost_IsRemote': 'mean',
        'software_skills_count': 'mean'
    }).reset_index()
    
    province_stats.columns = ['Province', 'Job_Count', 'Avg_Salary_Million_Toman', 'Remote_Percentage', 'Avg_Skills']
    province_stats['Avg_Salary_USD'] = province_stats['Avg_Salary_Million_Toman'] * 42
    province_stats = province_stats.sort_values('Job_Count', ascending=False)
    
    print("\n📍 Top 10 Provinces by Job Count:")
    print(province_stats.head(10)[['Province', 'Job_Count', 'Avg_Salary_USD', 'Remote_Percentage']])
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Job count by province
    top_provinces = province_stats.head(10)
    sns.barplot(data=top_provinces, x='Job_Count', y='Province', ax=ax1)
    ax1.set_title('Job Count by Province')
    
    # Average salary by province
    salary_provinces = province_stats.dropna(subset=['Avg_Salary_USD']).head(10)
    sns.barplot(data=salary_provinces, x='Avg_Salary_USD', y='Province', ax=ax2)
    ax2.set_title('Average Salary by Province (USD)')
    
    # Remote work percentage
    sns.barplot(data=top_provinces, x='Remote_Percentage', y='Province', ax=ax3)
    ax3.set_title('Remote Work Percentage by Province')
    ax3.set_xlabel('Remote Work Percentage')
    
    # Skills requirement
    sns.barplot(data=top_provinces, x='Avg_Skills', y='Province', ax=ax4)
    ax4.set_title('Average Skills Required by Province')
    ax4.set_xlabel('Average Number of Skills')
    
    plt.tight_layout()
    plt.show()
    
    return province_stats

province_analysis = create_geospatial_analysis()

# =============================================================================
# 7. COMPANY HIRING STRATEGY ANALYSIS
# =============================================================================

print("\n🏢 Company Hiring Strategy Analysis")

def analyze_hiring_strategies():
    print("\n📋 Hiring Strategy Insights:")
    
    # Remote work trends
    remote_stats = df.groupby('Jobpost_IndustryEn')['Jobpost_IsRemote'].agg(['mean', 'count']).reset_index()
    remote_stats = remote_stats[remote_stats['count'] >= 100]
    remote_stats = remote_stats.sort_values('mean', ascending=False).head(10)
    
    print(f"\n🏠 Remote Work Analysis:")
    print(f"   • Overall remote jobs: {df['Jobpost_IsRemote'].mean()*100:.1f}%")
    print(f"   • Top industries for remote work:")
    for _, row in remote_stats.head(5).iterrows():
        print(f"     - {row['Jobpost_IndustryEn']}: {row['mean']*100:.1f}%")
    
    # Gender preferences analysis
    gender_prefs = df['Jobpost_PreferredGender'].value_counts()
    print(f"\n👥 Gender Preference Analysis:")
    for gender, count in gender_prefs.items():
        print(f"   • {gender}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Benefits analysis
    benefits_df = df[df['Jobpost_BenefitEn'].notna()]
    print(f"\n🎁 Benefits offered in {len(benefits_df):,} job posts:")
    
    # Company size hiring patterns
    size_hiring = df.groupby('company_size_category').agg({
        'Jobpost_IsRemote': 'mean',
        'avg_salary_million_toman': 'mean',
        'software_skills_count': 'mean',
        'RowNumber': 'count'
    }).reset_index()
    
    size_hiring['avg_salary_usd'] = size_hiring['avg_salary_million_toman'] * 42
    
    print(f"\n🏭 Hiring Patterns by Company Size:")
    print(size_hiring)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Remote work by industry
    sns.barplot(data=remote_stats.head(8), x='mean', y='Jobpost_IndustryEn', ax=axes[0,0])
    axes[0,0].set_title('Remote Work by Industry')
    axes[0,0].set_xlabel('Remote Work Percentage')
    
    # Gender preferences
    gender_prefs.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
    axes[0,1].set_title('Gender Preferences in Job Posts')
    
    # Company size vs remote work
    sns.barplot(data=size_hiring, x='Jobpost_IsRemote', y='company_size_category', ax=axes[1,0])
    axes[1,0].set_title('Remote Work by Company Size')
    axes[1,0].set_xlabel('Remote Work Percentage')
    
    # Skills requirement by company size
    sns.barplot(data=size_hiring, x='software_skills_count', y='company_size_category', ax=axes[1,1])
    axes[1,1].set_title('Skills Required by Company Size')
    axes[1,1].set_xlabel('Average Skills Count')
    
    plt.tight_layout()
    plt.show()

analyze_hiring_strategies()

# =============================================================================
# 8. DEMAND FORECASTING
# =============================================================================

print("\n📈 Demand Forecasting Analysis")

def analyze_hiring_trends():
    print("\n📊 Hiring Trends Over Time:")
    
    # Monthly trends
    monthly_trends = df.groupby(['activation_year', 'activation_month']).agg({
        'RowNumber': 'count',
        'Jobpost_IsRemote': 'mean',
        'avg_salary_million_toman': 'mean'
    }).reset_index()
    
    monthly_trends['avg_salary_usd'] = monthly_trends['avg_salary_million_toman'] * 42
    
    # Create date column properly
    monthly_trends['date'] = pd.to_datetime({
        'year': monthly_trends['activation_year'],
        'month': monthly_trends['activation_month'], 
        'day': 1
    })
    monthly_trends = monthly_trends.sort_values('date')
    
    # Industry trends
    industry_trends = df.groupby(['activation_year', 'Jobpost_IndustryEn']).size().reset_index(name='job_count')
    top_industries = df['Jobpost_IndustryEn'].value_counts().head(5).index
    
    # Province trends
    province_trends = df.groupby(['activation_year', 'Jobpost_ProvinceEn']).size().reset_index(name='job_count')
    top_provinces = df['Jobpost_ProvinceEn'].value_counts().head(5).index
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    # Monthly job posting trends
    axes[0].plot(monthly_trends['date'], monthly_trends['RowNumber'], marker='o')
    axes[0].set_title('Monthly Job Posting Trends')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Number of Job Posts')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Industry trends
    for industry in top_industries:
        industry_data = industry_trends[industry_trends['Jobpost_IndustryEn'] == industry]
        axes[1].plot(industry_data['activation_year'], industry_data['job_count'], 
                    marker='o', label=industry[:30])
    axes[1].set_title('Job Posting Trends by Industry')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Number of Job Posts')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Province trends
    for province in top_provinces:
        province_data = province_trends[province_trends['Jobpost_ProvinceEn'] == province]
        axes[2].plot(province_data['activation_year'], province_data['job_count'], 
                    marker='o', label=province)
    axes[2].set_title('Job Posting Trends by Province')
    axes[2].set_xlabel('Year')
    axes[2].set_ylabel('Number of Job Posts')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"\n🔍 Key Trends:")
    latest_year = df['activation_year'].max()
    prev_year = latest_year - 1
    
    if prev_year in df['activation_year'].values:
        current_jobs = len(df[df['activation_year'] == latest_year])
        prev_jobs = len(df[df['activation_year'] == prev_year])
        growth_rate = ((current_jobs - prev_jobs) / prev_jobs) * 100
        print(f"   • YoY job posting growth: {growth_rate:.1f}%")
    
    remote_trend = df.groupby('activation_year')['Jobpost_IsRemote'].mean()
    print(f"   • Remote work trend: {remote_trend.iloc[-1]*100:.1f}% in {latest_year}")

analyze_hiring_trends()

# =============================================================================
# 9. SUMMARY AND INSIGHTS
# =============================================================================

print("\n" + "="*60)
print("📊 JOBVISION DATASET ANALYSIS SUMMARY")
print("="*60)

print(f"\n🎯 KEY FINDINGS:")
print(f"   📈 Dataset Overview:")
print(f"      • Total job posts analyzed: {len(df):,}")
print(f"      • Jobs with salary info: {df['has_salary_info'].sum():,} ({df['has_salary_info'].mean()*100:.1f}%)")
print(f"      • Remote jobs: {df['Jobpost_IsRemote'].sum():,} ({df['Jobpost_IsRemote'].mean()*100:.1f}%)")

print(f"\n   💰 Salary Insights:")
if len(salary_df) > 0:
    avg_salary_toman = salary_df['avg_salary_million_toman'].mean()
    median_salary_toman = salary_df['avg_salary_million_toman'].median()
    print(f"      • Average salary: {avg_salary_toman:.1f}M Toman (${avg_salary_toman*42:.0f} USD)")
    print(f"      • Median salary: {median_salary_toman:.1f}M Toman (${median_salary_toman*42:.0f} USD)")
    print(f"      • Best model performance: R² = {model_results[best_model_name]['R²']:.3f}")

print(f"\n   🌍 Geographic Insights:")
top_3_provinces = df['Jobpost_ProvinceEn'].value_counts().head(3)
for i, (province, count) in enumerate(top_3_provinces.items(), 1):
    print(f"      {i}. {province}: {count:,} jobs ({count/len(df)*100:.1f}%)")

print(f"\n   🏢 Company Insights:")
print(f"      • Most common company size: {df['company_size_category'].mode().iloc[0]}")
print(f"      • Tech jobs requiring programming: {len(df[df['programming_languages_count'] > 0]):,}")

print(f"\n   📱 Skills in Demand:")
all_programming_langs = []
for langs in df['programming_languages'].dropna():
    all_programming_langs.extend(langs)
top_skills = pd.Series(all_programming_langs).value_counts().head(5)
for i, (skill, count) in enumerate(top_skills.items(), 1):
    print(f"      {i}. {skill}: {count:,} mentions")

print(f"\n🚀 RECOMMENDATIONS:")
print(f"   • Focus on Tehran, {top_3_provinces.index[1]}, and {top_3_provinces.index[2]} for maximum reach")
print(f"   • {remote_trend.iloc[-1]*100:.1f}% remote work adoption shows growing flexibility")
print(f"   • Top skills like {top_skills.index[0]} and {top_skills.index[1]} are in high demand")
print(f"   • {best_model_name} model can predict salaries with {model_results[best_model_name]['R²']:.1%} accuracy")

print(f"\n" + "="*60)
print("✅ Analysis completed successfully!")
print("💾 All models and insights are ready for deployment")
print("="*60)