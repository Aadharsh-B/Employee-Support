from flask import Flask, request
import sqlite3
import datetime

app = Flask(__name__)

# Initialize Database
def init_db():
    conn = sqlite3.connect("sensor_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY, sensor1 INTEGER, sensor2 INTEGER, sensor3 INTEGER, sensor4 INTEGER, label INTEGER, timestamp TEXT)''')
    conn.commit()
    conn.close()

@app.route('/log_data', methods=['POST'])
def log_data():
    data = request.data.decode("utf-8")  # Read raw text data
    if data:
        try:
            values = list(map(int, data.split(",")))  # Convert CSV string to list of integers
            sensor1, sensor2, sensor3, sensor4, label = values

            conn = sqlite3.connect("sensor_data.db")
            c = conn.cursor()
            c.execute("INSERT INTO logs (sensor1, sensor2, sensor3, sensor4, label, timestamp) VALUES (?, ?, ?, ?, ?, ?)", 
                      (sensor1, sensor2, sensor3, sensor4, label, datetime.datetime.now()))
            conn.commit()
            conn.close()

            return "Data logged successfully!", 200
        except Exception as e:
            return f"Error: {str(e)}", 400

    return "Invalid Data", 400

@app.route('/view_logs', methods=['GET'])
def view_logs():
    conn = sqlite3.connect("sensor_data.db")
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY id DESC")  # Show latest logs first
    logs = c.fetchall()
    conn.close()

    if not logs:
        return "No data available!"

    return "<br>".join([f"ID: {row[0]}, S1: {row[1]}, S2: {row[2]}, S3: {row[3]}, S4: {row[4]}, Label: {row[5]}, Timestamp: {row[6]}" for row in logs])

if __name__ == '__main__':
    init_db()  # Ensure database exists
    app.run(host='0.0.0.0', port=5000, debug=True)
