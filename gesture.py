import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv(r"C:\Users\neash\Downloads\vl53l0x_gesture_data.csv")

# Feature engineering
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute
data['second'] = data['timestamp'].dt.second

# Select features and target
features = ['sensor_1','sensor_2', 'hour', 'minute', 'second']
X = data[features]
y = data['gesture']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=50)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=50)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
