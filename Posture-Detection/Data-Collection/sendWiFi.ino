#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi Credentials
const char* ssid = "aadharsh";
const char* password = "aadharsh";
const char* serverUrl = "http://192.168.83.248:5000/log_data"; // Flask server IP

// Button pin definitions
const int button1 = 12;
const int button2 = 13;
const int button3 = 14;

// Analog input pins
const int analogPin1 = 32;
const int analogPin2 = 33;
const int analogPin3 = 34;
const int analogPin4 = 35;

// Structure to store data
struct SensorData {
    int data1;
    int data2;
    int data3;
    int data4;
    bool label;
} sensorData;

// Debounce settings
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50; // 50ms debounce

void setup() {
    Serial.begin(115200);

    pinMode(button1, INPUT_PULLUP);
    pinMode(button2, INPUT_PULLUP);
    pinMode(button3, INPUT_PULLUP);

    // Connect to Wi-Fi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        delay(1000);
    }
    Serial.println("\nConnected to WiFi");
}

void loop() {
    if (digitalRead(button2) == LOW) {  // Button 2 pressed
        delay(debounceDelay);
        sensorData.label = true;
        captureAnalogData();
        Serial.println("Label set to 1, data captured");
    }

    if (digitalRead(button3) == LOW) {  // Button 3 pressed
        delay(debounceDelay);
        sensorData.label = false;
        captureAnalogData();
        Serial.println("Label set to 0, data captured");
    }

    if (digitalRead(button1) == LOW) {  // Button 1 pressed
        delay(debounceDelay);
        sendData();
    }
}

void captureAnalogData() {
    sensorData.data1 = analogRead(analogPin1);
    sensorData.data2 = analogRead(analogPin2);
    sensorData.data3 = analogRead(analogPin3);
    sensorData.data4 = analogRead(analogPin4);
}

void sendData() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "text/plain");

        String data = String(sensorData.data1) + "," + 
                      String(sensorData.data2) + "," + 
                      String(sensorData.data3) + "," + 
                      String(sensorData.data4) + "," + 
                      String(sensorData.label ? 1 : 0);

        int httpResponseCode = http.POST(data);

        Serial.print("HTTP Response Code: ");
        Serial.println(httpResponseCode);

        http.end();
    } else {
        Serial.println("WiFi not connected!");
    }
}
