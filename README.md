# Data_Analytics_Project
    Data_Analytics_Project
# Case Study: Advanced Analysis of Data Analyst Job Listings in India

# Data Importing and Description

    import pandas as pd
    import numpy as np
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import OneHotEncoder

# Load the dataset
    df = pd.read_csv("/content/dataAnalystJobsIndia_7th_July_2024.csv")
    
    df
    
    df.describe()
    
    print(df.columns.tolist())
    
    print(df.dtypes)

    missing = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_summary = pd.DataFrame({
     "Missing Count": missing,
     "Missing %": missing_percent.round(2)
    })
    print(missing_summary)

    print(df.describe(include="number"))
    print(df.describe(include="object"))
    print("Number of duplicate rows:", df.duplicated().sum())

# 1. DATA CLEANING AND PREPROCESSING

# Clean salary columns
    def extract_salary_number(x):
     if pd.isna(x):
         return np.nan
     nums = re.findall(r"\d+\.?\d*", str(x).replace(",", ""))
     return float(nums[0]) if nums else np.nan

    df["base_salary_num"] = df["base salary"].apply(extract_salary_number)
    df["max_salary_num"] = df["max salary"].apply(extract_salary_number)

# Fixing column name
    if "jobListed(days ago" in df.columns:
     df.rename(columns={"jobListed(days ago": "jobListed(days ago)"}, inplace=True)

# Cleaning experience values
    def clean_experience(x):
     if pd.isna(x):
         return np.nan
     x = str(x)
     match = re.findall(r"(\d+)", x)
     if len(match) == 1:
         return int(match[0])
     elif len(match) == 2:
         return (int(match[0]) + int(match[1])) / 2
     return np.nan

    df["exp_span"] = df["experience"].apply(clean_experience)

# Cleaning experience column (e.g., "4-8 Yrs" → 4, 8)
    def clean_exp(x):
     nums = re.findall(r"\d+", str(x))
     if len(nums) == 2:
         return int(nums[0]), int(nums[1])
     return np.nan, np.nan

    df["min_exp_clean"], df["max_exp_clean"] = zip(*df["experience"].map(clean_exp))
    df["exp_mid"] = df[["min_exp_clean", "max_exp_clean"]].mean(axis=1)

# Cleaning reviews count
    df["reviews_count_num"] = df["reviews count"].apply(lambda x: extract_salary_number(x))

# Standardize location
    df["location_clean"] = df["location"].str.split(",").str[0].str.strip()

# Removing rows where essential values are missing
    df = df.dropna(subset=["rating", "reviews_count_num", "exp_mid"])

# 2. EXPLORATORY DATA ANALYSIS"""

# Rating distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df["rating"], kde=True)
    plt.title("Distribution of Job Ratings")
    plt.show()

# Reviews count distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df["reviews_count_num"], kde=True)
    plt.title("Distribution of Review Counts")
    plt.show()

# Jobs by location
    plt.figure(figsize=(17,7))
    df["location_clean"].value_counts().plot(kind="bar")
    plt.title("Job Listings by Location")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.show()

# Experience mid value distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df["exp_mid"], kde=True)
    plt.title("Distribution of Required Experience")
    plt.show()

# 3. STATISTICAL ANALYSIS

    print("Top locations:\n", df["location"].value_counts())
    print("\nTop companies:\n", df["company"].value_counts())

# Correlation tests
    clean_corr = df[["base salary", "reviews count"]].dropna()

    if len(clean_corr) >= 2:
     pearson_r, pearson_p = stats.pearsonr(
         clean_corr["base salary"], clean_corr["reviews count"]
     )
     spearman_rho, spearman_p = stats.spearmanr(
        clean_corr["base salary"], clean_corr["reviews count"]
     )

    print("\nPearson:", pearson_r, pearson_p)
    print("Spearman:", spearman_rho, spearman_p)

# Comparing ratings on Naukri vs iimjobs
    naukri = df[df["postedIn"] == "Naukri"]["rating"]
    iim = df[df["postedIn"] == "iimjobs"]["rating"]

    print("\nT-test: Naukri vs iimjobs ratings")
    print(stats.ttest_ind(naukri, iim, nan_policy="omit"))

# Correlation (ratings vs reviews count)
    print("\nCorrelation (ratings vs reviews count)")
    print(stats.pearsonr(df["rating"], df["reviews_count_num"]))

# 4. MACHINE LEARNING MODEL

# Keeping only rows where salary exists
    ml_df = df.dropna(subset=["base_salary_num"])

    if len(ml_df) < 5:
       print("\n❌ Not enough salary data to train ML model (need ≥ 5 rows).")
    else:
# Selecting features
    X = ml_df[["exp_mid", "rating", "reviews_count_num", "jobListed(days ago)"]]
    y = ml_df["base_salary_num"]

# Handle remaining NaN
    X = X.fillna(X.mean())

# Train-test splitting
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nMODEL RESULTS")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²:", r2_score(y_test, y_pred))

# Feature Importance
    feat_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFEATURE IMPORTANCE")
    print(feat_imp)

# Plot feature importance
    plt.figure(figsize=(7,5))
    sns.barplot(data=feat_imp, x="Importance", y="Feature")
    plt.title("Feature Importance in Salary Prediction")
    plt.show()

# ANALYSIS BY MD RIYAZ UDDIN.

**THANK YOU...**
