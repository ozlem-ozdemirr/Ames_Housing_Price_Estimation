import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv"
df = pd.read_csv(url)
df.shape

df.info()
df.describe().T
df.head()

missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# Drop columns with many missing values
df = df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])

# Fill numerical missing values with the mean
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical missing values with 'None'
cat_cols = df.select_dtypes(include=[object]).columns
df[cat_cols] = df[cat_cols].fillna('None')

sns.histplot(df['SalePrice'], kde=True)
plt.title("Sales Price Distribution")
plt.show()

df['SalePrice'] = np.log1p(df['SalePrice'])

corr = df.corr()
top_corr_features = corr['SalePrice'].abs().sort_values(ascending=False).head(10)
print(top_corr_features)

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
            '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

dt = DecisionTreeRegressor(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Value (log)")
plt.ylabel("Predicted Value (log)")
plt.title("Random Forest - Actual vs Prediction")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()