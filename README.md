## Ames Housing Price Estimation

### Objective:
    ##### To develop predictive models that estimate residential property prices using the Ames Housing dataset, leveraging machine            ##### learning techniques for accurate and interpretable results.

### Approach:

    #### Data Preparation:

        ##### - Dropped high-missing-value columns and imputed missing data (mean for numeric, 'None' for categorical).

        ##### - Applied log transformation to normalize the skewed SalePrice distribution.

    #### Feature Selection:

        ##### - Chose top correlated features with SalePrice, such as OverallQual, GrLivArea, and GarageCars.

    #### Modeling:

        ##### - Trained three regression models: Linear Regression, Decision Tree, and Random Forest.

        ##### - Evaluated using RMSE and R² metrics.

    #### Results:

        ##### - Random Forest showed superior performance.

        ##### - Visual comparison of actual vs. predicted prices confirmed model accuracy.

### Conclusion:
    ##### The project effectively demonstrates end-to-end housing price prediction, highlighting the importance of feature selection, data     ##### preprocessing, and model evaluation in real estate analytics.

