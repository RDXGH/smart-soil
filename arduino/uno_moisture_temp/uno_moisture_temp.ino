#include <OneWire.h>
#include <DallasTemperature.h>

const int MOISTURE_PIN = A0;     // Capacitive sensor connected to A0
const int ONE_WIRE_BUS = 2;      // DS18B20 connected to digital pin 2

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(9600);
  sensors.begin();
  delay(500);
}

float readTemperatureC() {
  sensors.requestTemperatures();
  return sensors.getTempCByIndex(0); // first DS18B20 on the bus
}

int readMoisturePercent() {
  int raw = analogRead(MOISTURE_PIN);

  // --- Calibrate these values based on your sensor ---
  const int DRY_RAW = 900;  // Reading when sensor is in dry air
  const int WET_RAW = 300;  // Reading when fully submerged in water

  int pct = map(raw, DRY_RAW, WET_RAW, 0, 100);
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  return pct;
}

void loop() {
  int moisture = readMoisturePercent();
  float tempC = readTemperatureC();

  // Print in the exact format that serial_forwarder.py expects:
  Serial.print("moisture:");
  Serial.print(moisture);
  Serial.print(",temperature:");
  Serial.println(tempC, 2);  // 2 decimal places

  delay(2000); // Read every 2 seconds
}



