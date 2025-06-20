## Ames Housing Price Estimation

### Objective:
#####     The primary goal of this project is to develop machine learning models that accurately predict house prices using the Ames      #####     Housing dataset. This dataset includes 80 explanatory variables describing various aspects of residential homes in Ames, Iowa.

### Dataset:
#####     The dataset was obtained from an online source and loaded using Pandas. It contains detailed information about housing features, #####     with both numerical and categorical variables. The target variable is SalePrice, which represents the final sale price of each #####     house.

### Steps Taken in the Project:

####   Data Exploration:

#####     The dataset's shape, structure, and summary statistics were examined.

#####     Missing values were identified and handled:

#####     Columns with excessive missing values (e.g., PoolQC, MiscFeature, etc.) were dropped.

#####     Missing numerical values were filled with the mean of each column.

#####     Missing categorical values were replaced with the string 'None'.

####   Data Visualization:

#####     The distribution of the target variable SalePrice was visualized.

#####     To normalize the skewness in SalePrice, a log transformation (np.log1p) was applied.

####   Feature Selection:

#####     A correlation matrix was computed to identify variables most strongly associated with SalePrice.

#####     The top features based on correlation included: 
#####     OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt.

####   Model Building:

#####     The selected features were split into training and test datasets (80% train / 20% test).

#####     Three regression models were trained:

#####       Linear Regression

#####       Decision Tree Regressor (with a maximum depth of 5)

#####       Random Forest Regressor (with 100 trees and a maximum depth of 10)

####   Model Evaluation:

#####     Models were evaluated using:

#####       Root Mean Squared Error (RMSE)

#####       R-squared (R²) Score

#####     The evaluation metrics were printed for each model.

####   Result Visualization:

#####     A scatter plot was generated to compare the actual and predicted values for the Random Forest model (log-transformed values), #####     showing how well the model fits the data.

### Conclusion:
#####     The project demonstrates a full machine learning workflow for predicting house prices, including preprocessing, feature     #####     selection, model training, and evaluation. Among the models tested, the Random Forest Regressor likely performed best due to its #####     ability to capture complex nonlinear relationships in the data. The use of log-transformation helped to stabilize variance and #####     improve model accuracy.
