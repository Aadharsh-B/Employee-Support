import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned CSV file
df = pd.read_csv("cleaned_sensor_data.csv")

# 1️⃣ Check for missing values
print("\n🔍 Checking for missing values:")
print(df.isnull().sum())

# 2️⃣ Label distribution
print("\n📊 Label Distribution:")
print(df['label'].value_counts())

# 3️⃣ Sensor value statistics
print("\n📈 Sensor Value Statistics:")
print(df.describe())

# 4️⃣ Visualizing label distribution
plt.figure(figsize=(5,5))
df['label'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title("Label Distribution (0 = Bad Posture, 1 = Good Posture)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks([0,1], ["Bad (0)", "Good (1)"], rotation=0)
plt.show()
