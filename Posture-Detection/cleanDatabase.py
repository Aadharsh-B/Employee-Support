import sqlite3
import pandas as pd

# Path to your SQLite database file
db_path = r"C:\Aadharsh\binaryClassifier\sensor_data.db"  # Change to your actual path

# Connect to the database
conn = sqlite3.connect(db_path)

# Read the data while ignoring 'id' and 'timestamp' columns
query = "SELECT sensor1, sensor2, sensor3, sensor4, label FROM logs"
df = pd.read_sql_query(query, conn)

# Close database connection
conn.close()

# Display first 5 rows of cleaned data
print(df.head())

# Save cleaned data to CSV for verification (Optional)
df.to_csv("cleaned_sensor_data.csv", index=False)
print("✅ Cleaned data saved as 'cleaned_sensor_data.csv'")
