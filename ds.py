import pandas as pd

df = pd.read_csv(r"Data science salary.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nSample data:")
print(df.head())

#drop columns
df = df.drop(columns=['salary', 'salary_currency'])

#check again
print("\nAfter dropping columns:")
print(df.head())


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nAfter encoding:")
print(df.head())


# Split features and target
X = df.drop('salary_in_usd', axis=1)
y = df['salary_in_usd']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


#Train model 
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


#predict
y_pred = model.predict(X_test)


#evaluate 
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)


import pandas as pd

importance = model.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feat_imp)


print("\nInterpretation:")
print(f"Model is off by ~${mae:.0f} on average")

top_jobs = df['job_title'].value_counts().nlargest(10).index
df['job_title'] = df['job_title'].apply(lambda x: x if x in top_jobs else 'Other')
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))

import joblib

joblib.dump(model, "model.pkl")
print("Model saved")