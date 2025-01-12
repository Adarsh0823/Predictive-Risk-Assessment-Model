import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Function to preprocess input data
def preprocess_data(new_data):
    """
    Preprocesses the input data to match the format of the training data.

    Parameters:
        new_data (pd.DataFrame): DataFrame containing input data.

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    # Encode categorical features (e.g., Gender)
    if 'Gender' in new_data.columns:
        encoder = LabelEncoder()
        new_data['Gender'] = encoder.fit_transform(new_data['Gender'])

    # Standardize numerical features
    numerical_features = ['Age', 'BloodPressure', 'Cholesterol']
    scaler = StandardScaler()
    new_data[numerical_features] = scaler.fit_transform(new_data[numerical_features])

    return new_data

# Load the trained model
def load_model(model_path):
    """
    Loads the trained model from the specified path.

    Parameters:
        model_path (str): Path to the saved model file.

    Returns:
        model: Trained model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Main function to make predictions
def main():
    # Path to the trained model
    model_path = 'models/risk_assessment_model.pkl'

    # Load the model
    print("[INFO] Loading the trained model...")
    model = load_model(model_path)

    # Example new data (replace with user input or API integration)
    print("[INFO] Preprocessing input data...")
    new_data = pd.DataFrame({
        'Age': [50, 30],
        'Gender': ['Male', 'Female'],
        'BloodPressure': [150, 120],
        'Cholesterol': [220, 180],
        'Diabetes': [1, 0],
        'Smoker': [0, 1],
        'HeartDisease': [0, 0]
    })

    # Preprocess the new data
    preprocessed_data = preprocess_data(new_data)

    # Make predictions
    print("[INFO] Making predictions...")
    predictions = model.predict(preprocessed_data)
    probabilities = model.predict_proba(preprocessed_data)[:, 1]  # Probability of being high-risk

    # Display results
    for i, prediction in enumerate(predictions):
        risk_status = "High Risk" if prediction == 1 else "Low Risk"
        print(f"Member {i+1}: {risk_status} (Probability: {probabilities[i]:.2f})")

if __name__ == "__main__":
    main()
