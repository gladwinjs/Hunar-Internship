import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
file_path = 'E:\\Hunar Intern\\weatherHistory.csv'  # Use the updated path
data = pd.read_csv(file_path)

# Select features and targets for temperature and weather summary prediction
features = ['Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
            'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
target_temp = 'Temperature (C)'
target_summary = 'Summary'

# Feature and target selection
X = data[features]
y_temp = data[target_temp]

# Encode categorical summary data for logistic regression
label_encoder = LabelEncoder()
data['Encoded_Summary'] = label_encoder.fit_transform(data[target_summary])
y_summary = data['Encoded_Summary']

# Split data for both temperature and summary predictions
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
X_train_summary, X_test_summary, y_train_summary, y_test_summary = train_test_split(X, y_summary, test_size=0.2, random_state=42)

# Train the Linear Regression model for temperature
temp_model = LinearRegression()
temp_model.fit(X_train_temp, y_train_temp)

# Train the Logistic Regression model for weather summary
summary_model = LogisticRegression(max_iter=1000)
summary_model.fit(X_train_summary, y_train_summary)

# Make predictions for temperature
y_pred_temp = temp_model.predict(X_test_temp)

# Calculate evaluation metrics for temperature
mse = mean_squared_error(y_test_temp, y_pred_temp)
r2 = r2_score(y_test_temp, y_pred_temp)

# Display evaluation metrics
print("Evaluation Metrics for Temperature Prediction:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Plotting the actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.scatter(y_test_temp, y_pred_temp, alpha=0.3, color='blue', label='Predicted vs Actual')
plt.plot([y_test_temp.min(), y_test_temp.max()], [y_test_temp.min(), y_test_temp.max()], 'r--', linewidth=2, label='Ideal Line')
plt.xlabel('Actual Temperature (C)')
plt.ylabel('Predicted Temperature (C)')
plt.title('Actual vs Predicted Temperature (Linear Regression)')
plt.legend()
plt.show()

# Function to get predictions for new input
def predict_new_data():
    print("Enter the following values to predict Temperature and Weather Summary:")
    
    # Collect user input for each feature
    apparent_temp = float(input("Apparent Temperature (C): "))
    humidity = float(input("Humidity (0 to 1): "))
    wind_speed = float(input("Wind Speed (km/h): "))
    wind_bearing = float(input("Wind Bearing (degrees): "))
    visibility = float(input("Visibility (km): "))
    pressure = float(input("Pressure (millibars): "))
    
    # Create a DataFrame with the new input data
    new_data = pd.DataFrame([[apparent_temp, humidity, wind_speed, wind_bearing, visibility, pressure]], 
                            columns=features)
    
    # Predict Temperature
    temp_prediction = temp_model.predict(new_data)[0]
    
    # Predict Summary
    summary_prediction_encoded = summary_model.predict(new_data)[0]
    summary_prediction = label_encoder.inverse_transform([summary_prediction_encoded])[0]
    
    # Display predictions
    print("\nPredictions:")
    print("Predicted Temperature (C):", temp_prediction)
    print("Predicted Weather Summary:", summary_prediction)

# Run the prediction function for new data input
predict_new_data()
