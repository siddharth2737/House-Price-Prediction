import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import sys
import os

# Add the 'src' folder to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can import the preprocessing module
from preprocessing import load_data, clean_data, encode_data, scale_features, split_data

# Train and evaluate the models
def train_models(file_path):
    # Load and preprocess data
    df = load_data(file_path)
    df = clean_data(df)
    df = encode_data(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Initialize the models
    rf_model = RandomForestRegressor(random_state=42)
    gb_model = GradientBoostingRegressor(random_state=42)

    # Train the models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    # Make predictions
    rf_preds = rf_model.predict(X_test)
    gb_preds = gb_model.predict(X_test)

    # Evaluate the models
    rf_rmse = root_mean_squared_error(y_test, rf_preds)
    gb_rmse = root_mean_squared_error(y_test, gb_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    gb_r2 = r2_score(y_test, gb_preds)

    # Print results
    print(f"Random Forest - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")
    print(f"Gradient Boosting - RMSE: {gb_rmse:.2f}, R²: {gb_r2:.2f}")

    # Save the models
    with open('rf_model.pkl', 'wb') as rf_file:
        pickle.dump(rf_model, rf_file)
    
    with open('gb_model.pkl', 'wb') as gb_file:
        pickle.dump(gb_model, gb_file)

# Run the model training
if __name__ == "__main__":
    train_models('data/dataset.csv')
