#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// Initialize the LCD (I2C address 0x27, 16x2)
LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  lcd.init();
  lcd.backlight();
  Serial.begin(9600);

  // Initial message
  lcd.setCursor(0, 0);
  lcd.print("No Input");
}

void loop() {
  if (Serial.available() > 0) {  // If data is available
    char input = Serial.read();  // Read the first character
    lcd.clear();
    lcd.setCursor(0, 0);

    switch (input) {
      case '0':
        lcd.print("Productive");
        break;
      case '1':
        lcd.print("Spotify");
        break;
      case '2':
        lcd.print("WhatsApp");
        break;
      case '3':
        lcd.print("Spotify");
        lcd.setCursor(0, 1);
        lcd.print("WhatsApp");
        break;
      case '4':
        lcd.print("Waiting...");
        break;
      default:
        lcd.print("Unknown input");
        break;
    }
    
    delay(500);  // Small delay to prevent flickering
  }
}
