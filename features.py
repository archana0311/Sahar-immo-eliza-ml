import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import randint, uniform



def features():
        
    None  
        
      
    # Load the data
data = pd.read_csv("data/properties.csv")
    
        # Define features to use
num_features = ["nbr_frontages", 'nbr_bedrooms',"latitude", "longitude", "total_area_sqm",
                     'surface_land_sqm','terrace_sqm','garden_sqm']
fl_features = ["fl_terrace", 'fl_garden', 'fl_swimming_pool']
cat_features = ["province", 'heating_type', 'state_building',
                    "property_type", "epc", 'locality', 'subproperty_type','region']

    # Split the data into features and target
X = data[num_features + fl_features + cat_features]
y = data["price"]

    # Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train[num_features])
X_train[num_features] = imputer.transform(X_train[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])


    # Convert categorical columns with one-hot encoding using OneHotEncoder
enc = OneHotEncoder()
enc.fit(X_train[cat_features])
X_train_cat = enc.transform(X_train[cat_features]).toarray()
X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

# Use the best parameters found during RandomizedSearchCV
best_params = {
        'n_estimators': 130,
        'max_depth': 10,
        'learning_rate': 0.3,
        'subsample': 0.9,
        'colsample_bytree': 1.0,
        'gamma': 5,
        'reg_alpha': 1.5,
        'reg_lambda': 1.0,
    }
  

    # Train the final model using the best parameters
final_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
final_model.fit(X_train, y_train)


    # Evaluate the final model
train_score = r2_score(y_train, final_model.predict(X_train))
test_score = r2_score(y_test, final_model.predict(X_test))

print(f"Train R² score: {train_score}")
print(f"Test R² score: {test_score}")


# Extract feature importance
feature_importance = final_model.feature_importances_

    # Get the names of the features
feature_names = X_train.columns

    # Sort feature importance in descending order
sorted_indices = np.argsort(feature_importance)[::-1]

    # Print feature importance scores with corresponding names
print("Feature Importance Scores:")
for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. Feature '{feature_names[idx]}': {feature_importance[idx]}")

    # Select top N features
top_n = 15  # Change this to select a different number of top features
top_features = sorted_indices[:top_n]

print("\nTop", top_n, "Features:")
for i, idx in enumerate(top_features):
        print(f"{i+1}. Feature '{feature_names[idx]}'")


if __name__ == "__main__":
      features()