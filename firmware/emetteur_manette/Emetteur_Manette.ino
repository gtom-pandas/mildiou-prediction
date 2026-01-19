
#include <RF24.h>
#include <SPI.h>


// ==================== PINOUT ====================
// Joystick Gauche
#define JOY_LEFT_X A0
#define JOY_LEFT_Y A1
// Joystick Droit
#define JOY_RIGHT_X A2
#define JOY_RIGHT_Y A3
// Bouton
#define BTN_PIN 5
// Radio
#define CE_PIN 9
#define CSN_PIN 10

// ==================== NRF24 ====================
RF24 radio(CE_PIN, CSN_PIN);
const byte adresse[6] = "ROVE1";

// ==================== STRUCTURE (4 AXES + BOUTON) ====================
struct DataPackage {
  int16_t lx;  // Left X
  int16_t ly;  // Left Y
  int16_t rx;  // Right X
  int16_t ry;  // Right Y
  uint8_t btn; // Bouton
};

DataPackage data;

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  pinMode(BTN_PIN, INPUT_PULLUP);

  Serial.println(F("=== EMETTEUR NANO ==="));
  Serial.print(F("sizeof(DataPackage) = "));
  Serial.println(sizeof(DataPackage)); // Doit afficher 9

  if (!radio.begin()) {
    Serial.println(F("[ERREUR] NRF24!"));
    while (1)
      ;
  }

  radio.setPALevel(RF24_PA_LOW);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(108);
  radio.openWritingPipe(adresse);
  radio.stopListening();

  Serial.println(F("[OK] Radio TX - Mode 2 Joysticks"));
}

// ==================== LOOP ====================
void loop() {
  // Lecture des 4 axes
  data.lx = (int16_t)analogRead(JOY_LEFT_X);
  data.ly = (int16_t)analogRead(JOY_LEFT_Y);
  data.rx = (int16_t)analogRead(JOY_RIGHT_X);
  data.ry = (int16_t)analogRead(JOY_RIGHT_Y);
  data.btn = !digitalRead(BTN_PIN);

  bool ok = radio.write(&data, sizeof(DataPackage));

  // Debug
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 300) {
    lastPrint = millis();
    Serial.print(F("LY:"));
    Serial.print(data.ly);
    Serial.print(F(" RY:"));
    Serial.print(data.ry);
    Serial.println(ok ? F(" [OK]") : F(" [FAIL]"));
  }

  delay(20);
}
