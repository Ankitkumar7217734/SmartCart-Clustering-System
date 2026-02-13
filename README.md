# SmartCart Clustering System

Customer segmentation project using unsupervised learning on SmartCart customer data.

## Project Overview

This project groups customers into meaningful segments based on demographics, spending behavior, and campaign response features.

The workflow is implemented in a Jupyter notebook:

- `SmartCart_clustering_system_1.ipynb`

Main techniques used:

- Data cleaning and preprocessing
- Feature engineering
- One-hot encoding for categorical features
- Feature scaling with `StandardScaler`
- Dimensionality reduction with `PCA` (3 components)
- Clustering with `KMeans` and `AgglomerativeClustering`
- Cluster profiling and visualization

Dataset used in the notebook:

- `smartcart_customers.csv` (expected in the same folder)

## Workflow Implemented

1. Load data and inspect shape/null values.
2. Fill missing values (`Income` median imputation).
3. Engineer features:
   - `Age` from `Year_Birth`
   - `Customer_Tenure_Days` from `Dt_Customer`
   - `Total_Spending` from product spend columns
   - `Total_Children` from `Kidhome` + `Teenhome`
4. Consolidate categories:
   - Education into `UnderGraduate` / `Graduate` / `PostGraduate`
   - Marital status into `Living_With` (`Partner` / `Alone`)
5. Drop non-clustering columns (`ID`, original date and granular spend/home fields).
6. Detect and remove outliers (e.g., extreme `Age`, `Income`).
7. Encode categorical variables (`Education`, `Living_With`) with `OneHotEncoder`.
8. Scale all features using `StandardScaler`.
9. Apply `PCA` and visualize 3D principal component space.
10. Select `k` using elbow method (`kneed.KneeLocator`) and silhouette score.
11. Train clustering models:
    - `KMeans`
    - `AgglomerativeClustering` (Ward linkage)
12. Characterize clusters using:
    - Cluster counts
    - Income vs spending scatter plot
    - Mean feature summary by cluster

## Requirements

Install Python dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed jupyter
```

## How to Run

1. Place `smartcart_customers.csv` in this project directory.
2. Open `SmartCart_clustering_system_1.ipynb`.
3. Run cells from top to bottom.
4. Review:
   - PCA explained variance and visualizations
   - Elbow + silhouette plots
   - Final cluster plots and cluster summary table

## Output

The notebook produces segmented customer groups and their aggregate behavior profiles, which can be used for:

- Targeted marketing campaigns
- Personalized offers
- Customer retention and value-based strategies

## Project Achievements

- Built a complete end-to-end customer segmentation pipeline from raw data to final cluster insights.
- Improved data quality by handling missing values, engineering meaningful features, and removing outliers.
- Reduced feature complexity using PCA while preserving useful customer behavior patterns.
- Applied and compared two clustering approaches (`KMeans` and `AgglomerativeClustering`) for stronger analysis.
- Determined cluster structure using both elbow method and silhouette score instead of guesswork.
- Produced cluster-level business profiles (income, spending, household behavior) to support marketing decisions.
- Delivered interpretable visual outputs (3D PCA plots, cluster distribution, spending vs income patterns).

## Notes

- The notebook currently uses a fixed year (`2026`) for age calculation.
- For long-term reuse, replace with dynamic current year logic if needed.
