from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

app = Flask(__name__)

# Load the trained models
with open('rf_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('gb_model.pkl', 'rb') as gb_file:
    gb_model = pickle.load(gb_file)

# Load and fit the scaler with the original price range
df = pd.read_csv('data/dataset.csv')
price_scaler = StandardScaler()
price_scaler.fit(df[['price']])

def format_price(price):
    """Format price in Indian format with Rupees symbol"""
    price = int(price)
    formatted_price = f"₹{price:,}"
    # Convert to crores/lakhs for better readability
    if price >= 10000000:  # 1 crore = 10 million
        formatted_price += f" ({price/10000000:.2f} Crores)"
    elif price >= 100000:  # 1 lakh = 100 thousand
        formatted_price += f" ({price/100000:.2f} Lakhs)"
    return formatted_price

# Home route to render the prediction form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    
    # Encode categorical variables correctly
    mainroad = 1 if request.form['mainroad'].lower() == 'yes' else 0
    guestroom = 1 if request.form['guestroom'].lower() == 'yes' else 0
    basement = 1 if request.form['basement'].lower() == 'yes' else 0
    hotwaterheating = 1 if request.form['hotwaterheating'].lower() == 'yes' else 0
    airconditioning = 1 if request.form['airconditioning'].lower() == 'yes' else 0
    parking = int(request.form['parking'])
    prefarea = 1 if request.form['prefarea'].lower() == 'yes' else 0
    
    # Handle furnishingstatus using one-hot encoding
    furnishingstatus = request.form['furnishingstatus'].lower()
    # Create the two binary columns (furnished and semi-furnished, unfurnished is the reference)
    furnishing_furnished = 1 if furnishingstatus == 'furnished' else 0
    furnishing_semifurnished = 1 if furnishingstatus == 'semi-furnished' else 0
    
    # Prepare the features list
    features = [
        area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
        hotwaterheating, airconditioning, parking, prefarea, 
        furnishing_furnished, furnishing_semifurnished
    ]
    
    # Convert features into an appropriate shape (e.g., 2D array)
    features = np.array(features).reshape(1, -1)
    
    # Make predictions and inverse transform to get actual prices
    rf_prediction = price_scaler.inverse_transform([[rf_model.predict(features)[0]]])[0][0]
    gb_prediction = price_scaler.inverse_transform([[gb_model.predict(features)[0]]])[0][0]
    
    # Format the predictions
    rf_prediction_formatted = format_price(rf_prediction)
    gb_prediction_formatted = format_price(gb_prediction)
    
    # Return the predictions to the user
    return render_template('index.html', 
                         rf_prediction=rf_prediction_formatted, 
                         gb_prediction=gb_prediction_formatted)


if __name__ == "__main__":
    app.run(debug=True)
