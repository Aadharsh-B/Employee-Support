import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (update path if needed)
df = pd.read_csv("cleaned_sensor_data.csv")  

# Split features and labels
X = df[['sensor1', 'sensor2', 'sensor3', 'sensor4']]  # Features
y = df['label']  # Labels (0 or 1)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Decision Tree Accuracy: {accuracy * 100:.2f}%")

# Show feature importance
importances = clf.feature_importances_
for i, feature in enumerate(X.columns):
    print(f"{feature} importance: {importances[i]:.4f}")
