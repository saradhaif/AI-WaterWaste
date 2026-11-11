# Exploring the Association of Water Withdrawal and Technology Across Countries

**Machine Learning (ST3189) Coursework | 2025–2026**

## Project Overview

This project investigates the relationship between technological development and freshwater withdrawal across countries. Using World Bank World Development Indicators (WDI) data for 2022, it explores whether countries with higher technology adoption also tend to have higher freshwater withdrawal, while considering economic, demographic, and agricultural factors.

The analysis combines exploratory data analysis, unsupervised learning, regression, and classification.

## Research Questions

1. Do countries form distinct technology-development clusters, and do these clusters differ in freshwater withdrawal?
2. Is technology adoption associated with freshwater withdrawal after accounting for economic, demographic, and agricultural factors?
3. Can machine learning distinguish countries with high versus low freshwater withdrawal?

## Dataset

**Source:** World Bank World Development Indicators (WDI)  
**Year:** 2022  
**Original coverage:** 266 countries/territories in the filtered data  
**Analysis sample:** 130 countries after removing rows with missing values

The target is freshwater withdrawal as a percentage of renewable freshwater resources. Variables described in the report include:

| Variable | Description |
| --- | --- |
| Water | Freshwater withdrawal (target) |
| Energy | Energy use per capita |
| Electricity | Electric power consumption per capita |
| Internet | Individuals using the internet (% of population) |
| Broadband | Fixed broadband subscriptions per 100 people |
| Mobile_Subscriptions | Mobile cellular subscriptions per 100 people |
| GDP | GDP per capita |
| Population | Total population |
| Agriculture | Agricultural freshwater withdrawals (% of total) |

## Methodology

### 1. Data Preparation and Exploratory Analysis
- Filtered data to 2022 and reshaped indicators into country-level columns.
- Removed observations with missing values and examined distributions, correlations, and summary statistics.
- Applied logarithmic transformations to selected skewed variables.
- Used principal component analysis (PCA) to construct a **Technology Index** from energy, electricity, internet, and broadband indicators.

### 2. Unsupervised Learning
- Applied **K-Means clustering** to explore country development profiles.
- Used the elbow method to select three clusters: low, middle, and high technology/development groups.
- Compared freshwater withdrawal across clusters.

### 3. Regression
Compared models for predicting freshwater withdrawal:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression

The report uses 10-fold cross-validation and discusses RMSE and R². It also distinguishes results on the log-transformed scale from results back-transformed to the original water scale.

### 4. Classification
Classified countries using a threshold of **25% freshwater withdrawal**:
- **Low withdrawal:** below 25%
- **High withdrawal:** 25% or above

Compared:
- Logistic Regression
- Random Forest Classification
- XGBoost Classification

Evaluation included ROC-AUC, precision, recall, and F1 score, with 10-fold cross-validation.

## Key Findings

- **Clustering:** Three technology-development groups were identified. The high-development cluster had substantially higher average freshwater withdrawal than the other two groups.
- **Linear regression:** The report found a positive association between the Technology Index and freshwater withdrawal; the linear model reported a cross-validated R² of **0.557** on the log-water scale.
- **Random Forest regression:** Reported a cross-validated R² of **0.683**, with the lowest RMSE (**15.222**) in the report's comparison on the original water scale.
- **Classification:** XGBoost achieved the highest reported ROC-AUC (**0.958**), followed by Random Forest (**0.951**) and Logistic Regression (**0.938**).
- **Model selection:** The report highlights trade-offs between predictive performance and interpretability; Logistic Regression was recommended as a simpler option for policy applications.

## Limitations

- Technology indicators are **proxies** rather than direct measurements of data-center water consumption.
- Missing observations reduced the sample to 130 countries, potentially introducing selection bias and overfitting risk.
- The analysis focuses on a single year (2022), so it cannot establish a causal relationship or capture changes over time.

## Recommendations

The report recommends integrating water-resource planning into technology development, encouraging water-efficient cooling and recycled-water practices, and extending future analysis across multiple years.

## References

The full report contains its reference list, including the World Bank WDI data source, UNEP water-stress indicators, and literature on data-center and AI water consumption.

## Note on Interpretation

Reported findings are associations and predictive results, not evidence that technology adoption directly causes greater freshwater withdrawal. Numerical results above are reproduced from the coursework report.
