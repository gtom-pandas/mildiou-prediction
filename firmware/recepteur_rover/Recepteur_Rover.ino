

#include <RF24.h>
#include <SPI.h>

// ==================== PINOUT ====================
#define CE_PIN 9
#define CSN_PIN 10

#define ENA 5
#define IN1 2
#define IN2 3
#define ENB 6
#define IN3 4
#define IN4 7

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

// ==================== FAILSAFE ====================
unsigned long lastSignal = 0;
const unsigned long TIMEOUT = 500;

const int DEADZONE = 100;

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  stopMotors();

  Serial.println(F("=== RECEPTEUR R4 - TANK DRIVE ==="));
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
  radio.openReadingPipe(1, adresse);
  radio.startListening();

  Serial.println(F("[OK] Radio RX prete"));
}

// ==================== LOOP ====================
void loop() {
  // Réception
  if (radio.available()) {
    radio.read(&data, sizeof(DataPackage));
    lastSignal = millis();

    processTankDrive();

    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 300) {
      lastPrint = millis();
      Serial.print(F("LY:"));
      Serial.print(data.ly);
      Serial.print(F(" RY:"));
      Serial.println(data.ry);
    }
  }

  // Failsafe
  if (millis() - lastSignal > TIMEOUT) {
    stopMotors();
    static unsigned long lastWarn = 0;
    if (millis() - lastWarn > 1000) {
      lastWarn = millis();
      Serial.println(F("[!] Signal perdu"));
    }
  }
}

// ==================== TANK DRIVE ====================
void processTankDrive() {
  /*
   * Tank Drive:
   * - Joystick Gauche Y (ly) -> Moteur Gauche
   * - Joystick Droit Y (ry)  -> Moteur Droit
   *
   * Joystick: 0-1023, centre = 512
   * ATTENTION: Sur certains joysticks, pousser vers le HAUT donne une valeur
   * BASSE Il faut donc INVERSER le signe
   */

  // Centrer autour de zéro et INVERSER (512 - valeur au lieu de valeur - 512)
  int leftJoy = 512 - data.ly;  // INVERSÉ
  int rightJoy = 512 - data.ry; // INVERSÉ

  // Appliquer zone morte
  if (abs(leftJoy) < DEADZONE)
    leftJoy = 0;
  if (abs(rightJoy) < DEADZONE)
    rightJoy = 0;

  // Mapper vers -255 à +255
  int leftSpeed = map(leftJoy, -512, 511, -255, 255);
  int rightSpeed = map(rightJoy, -512, 511, -255, 255);

  // Appliquer aux moteurs
  setMotorLeft(leftSpeed);
  setMotorRight(rightSpeed);
}

// ==================== MOTEURS ====================
void setMotorLeft(int speed) {
  if (speed > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, speed);
  } else if (speed < 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, -speed);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
  }
}

void setMotorRight(int speed) {
  // INVERSÉ: IN3/IN4 échangés pour compenser le câblage du moteur
  if (speed > 0) {
    digitalWrite(IN3, LOW);  // INVERSÉ (était HIGH)
    digitalWrite(IN4, HIGH); // INVERSÉ (était LOW)
    analogWrite(ENB, speed);
  } else if (speed < 0) {
    digitalWrite(IN3, HIGH); // INVERSÉ (était LOW)
    digitalWrite(IN4, LOW);  // INVERSÉ (était HIGH)
    analogWrite(ENB, -speed);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    analogWrite(ENB, 0);
  }
}

void stopMotors() {
  setMotorLeft(0);
  setMotorRight(0);
}
