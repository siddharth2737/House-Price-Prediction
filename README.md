# House Price Prediction

A machine learning web application that predicts house prices using Random Forest and Gradient Boosting models. The application provides an intuitive web interface for users to input house features and get price predictions.

## Features

- Interactive web interface for house price prediction
- Dual model approach using Random Forest and Gradient Boosting
- Data preprocessing and feature engineering
- Model performance metrics (RMSE and R² score)
- Responsive design with modern UI

## Project Structure

```
├── app/
│   ├── static/          # CSS and static assets
│   ├── templates/       # HTML templates
│   └── app.py          # Flask application
├── data/
│   └── dataset.csv     # Training dataset
├── src/
│   ├── model.py        # Model training and evaluation
│   └── preprocessing.py # Data preprocessing functions
├── gb_model.pkl        # Trained Gradient Boosting model
├── rf_model.pkl        # Trained Random Forest model
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app/app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the house features in the form and click "Predict" to get the estimated price

## Dependencies

- Flask >= 2.3.0
- scikit-learn >= 1.3.0
- pandas >= 2.1.0
- numpy >= 1.24.0

## Model Information

The application uses two machine learning models:
1. Random Forest Regressor
2. Gradient Boosting Regressor

Both models are trained on the provided dataset with various house features including location, size, number of rooms, and other relevant attributes.

## Contributing

Feel free to open issues and submit pull requests to contribute to this project.
