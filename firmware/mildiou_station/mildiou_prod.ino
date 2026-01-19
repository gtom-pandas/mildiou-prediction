/*
 * ============================================================
 * SYSTÈME DE PRÉDICTION DU MILDIOU - VERSION PRODUCTION V2
 * ============================================================
 * NOUVEAUTÉS V2:
 * • Classe NeuralNetwork OOP générique
 * • Feedback Loop: correction des prédictions via EEPROM
 * • Watchdog Timer 8s pour fiabilité 24/7
 * • Alertes WiFi (HTTP) optionnelles
 *
 * Hardware requis:
 * - Arduino UNO R4 WiFi
 * - LCD 16x2 I2C (adresse 0x27 ou 0x3F)
 * - DHT11 (VMA311) sur pin D2
 * - LPS25 sur I2C (A4/A5)
 *
 * Commandes Série:
 * - measure : Force une mesure
 * - endday  : Force la fin de journée (tests)
 * - predict : Force une prédiction
 * - CORRECT_PREDICTION <0|1|2> : Corrige dernière prédiction
 * - DUMP_ERRORS : Export CSV des erreurs sauvegardées
 * - CLEAR_ERRORS : Efface l'EEPROM d'erreurs
 */

// ==================== BIBLIOTHÈQUES ====================
#include "mildiou_nn_weights.h"
#include <Adafruit_LPS2X.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <EEPROM.h> // Pour feedback loop
#include <LiquidCrystal_I2C.h>
#include <WDT.h>    // Watchdog Timer R4
#include <WiFiS3.h> // WiFi intégré R4 (optionnel)
#include <Wire.h>

// ==================== CONFIGURATION WiFi (OPTIONNEL) ====================
// Décommenter et configurer pour activer les alertes WiFi
// #define WIFI_ENABLED
#ifdef WIFI_ENABLED
const char *WIFI_SSID = "VOTRE_SSID";
const char *WIFI_PASS = "VOTRE_MOT_DE_PASSE";
const char *ALERT_URL = "http://votre-serveur.com/api/alert";
#endif

// ==================== CONFIGURATION LoRa RYLR998 ====================
#define LORA_ENABLED // Commenter pour désactiver LoRa
#define LORA_ADDRESS 1
#define LORA_DEST_ADDRESS 2
#define LORA_NETWORK_ID 18

// ==================== CONFIGURATION CAPTEURS ====================
#define DHT_PIN 2
#define DHT_TYPE DHT11

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_LPS25 lps;
bool lpsAvailable = false;

// ==================== CONFIGURATION LCD ====================
LiquidCrystal_I2C lcd(0x27, 16, 2);

byte iconTemp[8] = {0b00100, 0b01010, 0b01010, 0b01110,
                    0b01110, 0b11111, 0b11111, 0b01110};

byte iconHumidity[8] = {0b00100, 0b00100, 0b01010, 0b01010,
                        0b10001, 0b10001, 0b10001, 0b01110};

byte iconAlert[8] = {0b00100, 0b01110, 0b01110, 0b01110,
                     0b00100, 0b00000, 0b00100, 0b00000};

byte iconOK[8] = {0b00000, 0b00001, 0b00011, 0b10110,
                  0b11100, 0b01000, 0b00000, 0b00000};

// ==================== CONFIGURATION SYSTÈME ====================
#define HISTORY_SIZE 14
#define MIN_DAYS_FOR_PREDICTION 3

// === MODE TEST (1 mesure/min, journée = 24 min) ===
const unsigned long MEASURE_INTERVAL = 60000;
const unsigned long DAY_DURATION = 1440000;

// === MODE PRODUCTION (décommenter) ===
// const unsigned long MEASURE_INTERVAL = 3600000;
// const unsigned long DAY_DURATION = 86400000;

unsigned long lastMeasureTime = 0;
unsigned long dayStartTime = 0;

float tempSum = 0;
float humSum = 0;
float pressSum = 0;
int measureCount = 0;

// Paramètres mildiou
#define TEMP_MIN_FAVORABLE 10.0
#define TEMP_MAX_FAVORABLE 30.0
#define TEMP_OPTIMAL_MIN 15.0
#define TEMP_OPTIMAL_MAX 28.0
#define HUMIDITY_THRESHOLD 70.0
#define HUMIDITY_CRITICAL 85.0
#define PRESSURE_NORMAL 1013.0
#define PRESSURE_LOW 1008.0
#define PRESSURE_DROP_THRESHOLD 5.0

// ==================== STRUCTURES ====================
struct DailyData {
  float meantemp;
  float humidity;
  float meanpressure;
  bool valid;
};

// Structure pour feedback loop EEPROM
struct ErrorSample {
  uint8_t day;       // Jour relatif (0-255)
  int8_t temp;       // Température arrondie
  uint8_t humidity;  // Humidité arrondie
  uint16_t pressure; // Pression - 900 (économie mémoire)
  uint8_t predicted; // Classe prédite (0-2)
  uint8_t actual;    // Classe corrigée (0-2)
};

// EEPROM Layout
#define EEPROM_ERROR_COUNT_ADDR 0 // 1 byte: nombre d'erreurs
#define EEPROM_ERROR_DATA_ADDR 1  // Début des ErrorSamples
#define MAX_ERROR_SAMPLES 60      // 60 × 8 bytes = 480 bytes (< 512)

// ==================== VARIABLES GLOBALES ====================
DailyData history[HISTORY_SIZE];
int historyIndex = 0;
int validDays = 0;
int currentDay = 0;

// Réseau de neurones - Architecture dynamique (lue depuis .h)
const int NUM_FEATURES = NN_INPUT_SIZE; // 25
float inputFeatures[NUM_FEATURES];

// Constantes de couches (MODIFIÉ pour [25, 20, 10, 3])
const int L1_IN = NN_LAYERS[0], L1_OUT = NN_LAYERS[1]; // 25 -> 20
const int L2_IN = NN_LAYERS[1], L2_OUT = NN_LAYERS[2]; // 20 -> 10
const int L3_IN = NN_LAYERS[2], L3_OUT = NN_LAYERS[3]; // 10 -> 3

// Buffers pour poids (dimensionnés pour nouvelle architecture)
float weights_L1[25][20]; // Max sizes
float bias_L1[20];
float weights_L2[20][10];
float bias_L2[10];
float weights_L3[10][3];
float bias_L3[3];

float feature_means[NUM_FEATURES];
float feature_stds[NUM_FEATURES];

float lastProba[3] = {0, 0, 0};
int lastPrediction = -1; // Pour feedback loop

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000)
    ;
  delay(1000);

  // *** WATCHDOG TIMER INIT (8 secondes) ***
  // Si le code plante pendant > 8s, l'Arduino reboot automatiquement
  WDT.begin(8000);
  Serial.println(F("[INIT] Watchdog Timer active (8s timeout)"));

  // Initialiser LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();

  lcd.createChar(0, iconTemp);
  lcd.createChar(1, iconHumidity);
  lcd.createChar(2, iconAlert);
  lcd.createChar(3, iconOK);

  Serial.println(F("[INIT] LCD 16x2 I2C initialise"));

  lcd.setCursor(0, 0);
  lcd.print("MILDIOU PREDICT");
  lcd.setCursor(0, 1);
  lcd.print("Version PROD V2");
  delay(2000);

  WDT.refresh(); // Caresser le chien pendant init

  printWelcome();

#ifdef WIFI_ENABLED
  // Initialiser WiFi (optionnel)
  lcd.clear();
  lcd.print("Connexion WiFi..");
  Serial.print(F("[WIFI] Connexion a "));
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int wifiAttempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifiAttempts < 20) {
    delay(500);
    Serial.print(".");
    wifiAttempts++;
    WDT.refresh();
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(F("\n[WIFI] Connecte!"));
    Serial.print(F("        IP: "));
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(F("\n[WIFI] Echec connexion - mode hors-ligne"));
  }
#endif

  // Initialiser capteurs
  Serial.println(F("\n[MODE] PRODUCTION V2 - Capteurs reels"));
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Init capteurs...");
  initRealSensors();
  WDT.refresh();

  initializeHistory();
  initializeWeights();
  initializeEEPROM(); // Nouveau: init feedback loop
  WDT.refresh();

  dayStartTime = millis();
  lastMeasureTime = millis();

  Serial.println(F("\n✓ Systeme pret (MODE PRODUCTION V2)"));
  Serial.println(F("Mesure auto toutes les heures"));
  Serial.println(F("Tapez 'help' pour commandes\n"));

  // Première mesure immédiate
  delay(2000);
  takeRealMeasurement();
  WDT.refresh();

#ifdef LORA_ENABLED
  // Initialiser LoRa via Serial1 (pins 0/1)
  Serial1.begin(115200);
  delay(1000); // Laisser le temps au module de démarrer

  Serial.println(F("[LORA] Configuration Forcee Emetteur..."));

  // 1. Reset usine
  Serial1.println("AT+RESET");
  delay(1000);
  WDT.refresh();

  // 2. Adresse émetteur = 1
  Serial1.println("AT+ADDRESS=1");
  delay(200);

  // 3. Network ID commun
  Serial1.println("AT+NETWORKID=18");
  delay(200);

  // 4. FORCER LA FRÉQUENCE (868.5 MHz Europe)
  Serial1.println("AT+BAND=868500000");
  delay(200);

  // 5. FORCER LES PARAMÈTRES (SF9, BW125k, robuste)
  Serial1.println("AT+PARAMETER=9,7,1,12");
  delay(200);

  Serial.println(F("[LORA] Config: ADDR=1, NET=18, BAND=868.5MHz"));
  Serial.println(F("[LORA] Pret - Envoi quotidien active"));
#endif
}

// ==================== LOOP ====================
void loop() {
  WDT.refresh(); // *** IMPORTANT: Refresh watchdog à chaque cycle ***

  unsigned long currentTime = millis();

  // Mesure automatique toutes les heures
  if (currentTime - lastMeasureTime >= MEASURE_INTERVAL) {
    lastMeasureTime = currentTime;
    takeRealMeasurement();
    WDT.refresh();
  }

  // Vérifier fin de journée (24h)
  if (currentTime - dayStartTime >= DAY_DURATION) {
    endOfDayProduction();
    WDT.refresh();
  }

  // Gérer commandes manuelles
  handleSerialCommands();

  delay(10);
}

// ==================== GESTION CAPTEURS ====================

void initRealSensors() {
  Serial.println(F("[INIT] Initialisation capteurs..."));

  // DHT11 (VMA311)
  dht.begin();
  delay(2000); // DHT11 nécessite plus de temps au démarrage

  float testTemp = dht.readTemperature();
  float testHum = dht.readHumidity();

  if (isnan(testTemp) || isnan(testHum)) {
    Serial.println(F("        ⚠ DHT11 (VMA311) NON DETECTE!"));
    Serial.println(F("          Pin DATA → D2"));
    Serial.println(F("          Resistance 10kΩ pull-up requis"));
    lcd.setCursor(0, 1);
    lcd.print("DHT11 ERREUR!");
    delay(3000);
  } else {
    // Note: DHT11 retourne des valeurs entières (pas de décimales)
    Serial.print(F("        ✓ DHT11 OK - T="));
    Serial.print(testTemp, 0); // Pas de décimale pour DHT11
    Serial.print(F("C H="));
    Serial.print(testHum, 0); // Pas de décimale pour DHT11
    Serial.println(F("%"));
    lcd.setCursor(0, 1);
    lcd.print("DHT11 OK!");
    delay(1000);
  }

  // LPS25 (Barometre de precision) via Adafruit_LPS2X
  Wire.begin();
  // LPS25 adresse par defaut: 0x5C ou 0x5D
  if (lps.begin_I2C(0x5C) || lps.begin_I2C(0x5D)) {
    lpsAvailable = true;
    lps.setDataRate(LPS25_RATE_12_5_HZ);

    // Lecture via sensors_event_t
    sensors_event_t pressure;
    sensors_event_t temp;
    lps.getEvent(&pressure, &temp);
    float testPress = pressure.pressure; // en hPa

    Serial.print(F("        \u2713 LPS25 OK - P="));
    Serial.print(testPress, 1);
    Serial.println(F("hPa"));
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("LPS25 OK!");
    delay(1000);
  } else {
    Serial.println(F("        LPS25 non detecte"));
    Serial.println(F("          I2C sur A4/A5 (0x5C/0x5D)"));
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("LPS25 absent");
    lcd.setCursor(0, 1);
    lcd.print("P=1013 par def");
    delay(2000);
  }
}

void takeRealMeasurement() {
  Serial.println(F("\n─────────── MESURE REELLE ───────────"));

  displayMessage("Mesure.. .", NULL, 500);

  // Lire DHT22
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  // Attendre stabilisation
  delay(100);

  // Relire pour confirmation
  if (isnan(temp) || isnan(hum)) {
    delay(2000);
    temp = dht.readTemperature();
    hum = dht.readHumidity();
  }

  // Lire LPS25 via sensors_event_t
  float press = 0;
  if (lpsAvailable) {
    sensors_event_t pressure_event;
    sensors_event_t temp_event;
    lps.getEvent(&pressure_event, &temp_event);
    press = pressure_event.pressure; // en hPa
  } else {
    press = 1013.0;
  }

  // Vérifier validité
  if (isnan(temp) || isnan(hum)) {
    Serial.println(F("⚠ ERREUR LECTURE DHT22!  "));
    displayMessage("ERREUR DHT22!", "Verifier cable", 3000);
    displayCurrentData();
    return;
  }

  // Vérifier plage réaliste
  if (temp < -20 || temp > 60 || hum < 0 || hum > 100) {
    Serial.println(F("⚠ Valeurs hors limites! "));
    Serial.print(F("  T="));
    Serial.print(temp);
    Serial.print(F(" H="));
    Serial.println(hum);
    displayMessage("Valeurs bizarres", "Verifier DHT22", 3000);
    displayCurrentData();
    return;
  }

  // Afficher sur série
  Serial.print(F("  Temperature:     "));
  Serial.print(temp, 1);
  Serial.println(F(" C"));

  Serial.print(F("  Humidite:      "));
  Serial.print(hum, 1);
  Serial.println(F(" %"));

  Serial.print(F("  Pression:     "));
  Serial.print(press, 1);
  Serial.println(F(" hPa"));

  // Afficher sur LCD
  displayMeasurement(temp, hum, press);
  delay(3000);

  // Accumuler pour moyenne journalière
  tempSum += temp;
  humSum += hum;
  pressSum += press;
  measureCount++;

  Serial.print(F("  Mesures du jour: "));
  Serial.print(measureCount);
  Serial.print(F(" (moyenne calculee a minuit)"));
  Serial.println();

  // Prochaine mesure dans
  unsigned long nextIn = MEASURE_INTERVAL / 60000; // Minutes
  Serial.print(F("  Prochaine mesure dans "));
  Serial.print(nextIn);
  Serial.println(F(" min"));

  // Mettre à jour affichage
  displayCurrentData();
}

void endOfDayProduction() {
  if (measureCount == 0) {
    Serial.println(F("⚠ Aucune mesure aujourd'hui"));
    dayStartTime = millis();
    return;
  }

  Serial.println(
      F("\n╔═══════════════════════════════════════════════════════╗"));
  Serial.println(
      F("║          FIN DE JOURNEE - CALCUL MOYENNE              ║"));
  Serial.println(
      F("╚═══════════════════════════════════════════════════════╝"));

  displayMessage("FIN DE JOURNEE", "Calcul moyenne", 2000);

  // Calculer moyennes
  float avgTemp = tempSum / measureCount;
  float avgHum = humSum / measureCount;
  float avgPress = pressSum / measureCount;

  Serial.print(F("  Mesures collectees:   "));
  Serial.println(measureCount);

  Serial.print(F("  Temp moyenne:    "));
  Serial.print(avgTemp, 1);
  Serial.println(F(" C"));

  Serial.print(F("  Hum moyenne:     "));
  Serial.print(avgHum, 1);
  Serial.println(F(" %"));

  Serial.print(F("  Press moyenne:    "));
  Serial.print(avgPress, 1);
  Serial.println(F(" hPa"));

  // Afficher sur LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Moy jour:");
  lcd.print((int)avgTemp);
  lcd.print("C");
  lcd.setCursor(0, 1);
  lcd.print((int)avgHum);
  lcd.print("% ");
  lcd.print((int)avgPress);
  lcd.print("hPa");
  delay(3000);

  // Stocker dans historique
  history[historyIndex].meantemp = avgTemp;
  history[historyIndex].humidity = avgHum;
  history[historyIndex].meanpressure = avgPress;
  history[historyIndex].valid = true;

  historyIndex = (historyIndex + 1) % HISTORY_SIZE;
  if (validDays < HISTORY_SIZE)
    validDays++;
  currentDay++;

  Serial.print(F("\n  ✓ Jour enregistre ("));
  Serial.print(validDays);
  Serial.print(F("/"));
  Serial.print(HISTORY_SIZE);
  Serial.println(F(")"));

  // Réinitialiser accumulateurs
  tempSum = 0;
  humSum = 0;
  pressSum = 0;
  measureCount = 0;
  dayStartTime = millis();

  // Prédiction si assez de données
  if (validDays >= MIN_DAYS_FOR_PREDICTION) {
    Serial.println(F("\n  → PREDICTION AUTOMATIQUE"));
    displayMessage("PREDICTION", "En cours...", 1000);
    int prediction = makePrediction();
  } else {
    Serial.print(F("\n  ⏳ Encore "));
    Serial.print(MIN_DAYS_FOR_PREDICTION - validDays);
    Serial.println(F(" jour(s) avant prediction"));

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Encore ");
    lcd.print(MIN_DAYS_FOR_PREDICTION - validDays);
    lcd.print(" jours");
    lcd.setCursor(0, 1);
    lcd.print("pour prediction");
    delay(3000);
  }

  Serial.println(
      F("╚═══════════════════════════════════════════════════════╝\n"));

#ifdef LORA_ENABLED
  // === ENVOI LORA QUOTIDIEN ===
  sendLoRaReport(avgTemp, avgHum, avgPress,
                 lastPrediction >= 0 ? lastPrediction : 0);
#endif

  displayCurrentData();
}

// ==================== ENVOI LORA ====================
#ifdef LORA_ENABLED
void sendLoRaReport(float temp, float hum, float press, int risk) {
  // Format: J:<Jour>,T:<Temp>,H:<Hum>,P:<Press>,R:<Risque>
  String payload = "J:" + String(currentDay) + ",T:" + String((int)temp) +
                   ",H:" + String((int)hum) + ",P:" + String((int)press) +
                   ",R:" + String(risk);

  int payloadLen = payload.length();

  Serial.print(F("[LORA] Envoi vers adresse "));
  Serial.print(LORA_DEST_ADDRESS);
  Serial.print(F(": "));
  Serial.println(payload);

  // Commande AT+SEND=<Address>,<Length>,<Data>
  Serial1.print(F("AT+SEND="));
  Serial1.print(LORA_DEST_ADDRESS);
  Serial1.print(F(","));
  Serial1.print(payloadLen);
  Serial1.print(F(","));
  Serial1.println(payload);

  delay(500);

  // Lire réponse
  while (Serial1.available()) {
    String response = Serial1.readStringUntil('\n');
    Serial.print(F("[LORA] Reponse: "));
    Serial.println(response);
  }

  // Afficher sur LCD
  lcd.clear();
  lcd.print("LoRa Envoye!");
  lcd.setCursor(0, 1);
  lcd.print("Risque: ");
  lcd.print(risk);
  delay(2000);
}
#endif

// ==================== AFFICHAGE LCD ====================

void displayCurrentData() {
  if (validDays == 0 && measureCount == 0) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("En attente");
    lcd.setCursor(0, 1);
    lcd.print("mesures...");
    return;
  }

  lcd.clear();

  if (measureCount > 0) {
    // Afficher moyenne du jour en cours
    float avgTemp = tempSum / measureCount;
    float avgHum = humSum / measureCount;

    lcd.setCursor(0, 0);
    lcd.write(0);
    lcd.print((int)avgTemp);
    lcd.print("C ");
    lcd.write(1);
    lcd.print((int)avgHum);
    lcd.print("%");

    lcd.setCursor(0, 1);
    lcd.print("Mesures:");
    lcd.print(measureCount);
    lcd.print("/24");
  } else if (validDays > 0) {
    // Afficher dernier jour enregistré
    int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
    float temp = history[current].meantemp;
    float hum = history[current].humidity;
    float press = history[current].meanpressure;

    lcd.setCursor(0, 0);
    lcd.write(0);
    lcd.print((int)temp);
    lcd.print("C ");
    lcd.write(1);
    lcd.print((int)hum);
    lcd.print("%");

    if (press < PRESSURE_LOW)
      lcd.print(" lo");
    else if (press > PRESSURE_NORMAL + 5)
      lcd.print(" hi");

    lcd.setCursor(0, 1);
    lcd.print("Jours:");
    lcd.print(validDays);
    lcd.print("/");
    lcd.print(HISTORY_SIZE);
  }
}

void displayMeasurement(float temp, float hum, float press) {
  lcd.clear();

  // LIGNE 1 : Affiche le numéro de mesure et la Pression
  // Ex: "M:1 P:1013hPa"
  lcd.setCursor(0, 0);
  lcd.print("M:");
  lcd.print(measureCount + 1);
  lcd.print(" P:");
  lcd.print((int)press); // Affiche la pression (ex: 1013)

  // LIGNE 2 : Affiche Température et Humidité (inchangé)
  // Ex: "🌡20C 💧60%"
  lcd.setCursor(0, 1);
  lcd.write(0); // Icone Temp
  lcd.print((int)temp);
  lcd.print("C ");
  lcd.write(1); // Icone Humidité
  lcd.print((int)hum);
  lcd.print("%");

  // Petit effet visuel pour confirmer la prise de mesure
  delay(200);
  lcd.noBacklight();
  delay(100);
  lcd.backlight();
}

void displayRiskLevel(int prediction, float confidence) {
  lcd.clear();

  switch (prediction) {
  case 0: // FAIBLE
    Serial.println(F("        [LCD] RISQUE FAIBLE"));
    lcd.setCursor(0, 0);
    lcd.write(3);
    lcd.print(" RISQUE FAIBLE");
    lcd.setCursor(0, 1);
    lcd.print("RAS-Surveiller");

    for (int i = 0; i < 3; i++) {
      delay(300);
      lcd.noBacklight();
      delay(100);
      lcd.backlight();
    }
    break;

  case 1: // MOYEN
    Serial.println(F("        [LCD] RISQUE MOYEN"));
    lcd.setCursor(0, 0);
    lcd.write(2);
    lcd.print(" RISQUE MOYEN");
    lcd.write(2);
    lcd.setCursor(0, 1);
    lcd.print("Preparer trait.");

    for (int i = 0; i < 4; i++) {
      delay(400);
      lcd.noBacklight();
      delay(200);
      lcd.backlight();
    }
    break;

  case 2: // ÉLEVÉ
    Serial.println(F("        [LCD] RISQUE ELEVE!!! "));

    for (int scroll = 0; scroll < 3; scroll++) {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("! !! ALERTE !!!");
      lcd.setCursor(0, 1);
      lcd.print("RISQUE ELEVE");

      for (int blink = 0; blink < 3; blink++) {
        delay(150);
        lcd.noBacklight();
        delay(150);
        lcd.backlight();
      }
    }

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("!  TRAITER VIGNE");
    lcd.setCursor(0, 1);
    lcd.print("dans 24-48h !");

    for (int i = 0; i < 6; i++) {
      delay(200);
      lcd.noBacklight();
      delay(100);
      lcd.backlight();
    }
    break;
  }

  delay(3000);
  displayProbabilities(prediction, confidence);
  delay(5000);
  displayCurrentData();
}

void displayProbabilities(int prediction, float confidence) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Confiance:");
  lcd.setCursor(0, 1);
  lcd.print((int)(confidence * 100));
  lcd.print("%");
  delay(2500);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("F:");
  lcd.print((int)(lastProba[0] * 100));
  lcd.print("% M:");
  lcd.print((int)(lastProba[1] * 100));
  lcd.print("%");
  lcd.setCursor(0, 1);
  lcd.print("E:");
  lcd.print((int)(lastProba[2] * 100));
  lcd.print("%");
  delay(2500);
}

void displayMessage(const char *line1, const char *line2, int duration) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  if (line2) {
    lcd.setCursor(0, 1);
    lcd.print(line2);
  }
  delay(duration);
}

void lcdTest() {
  Serial.println(F("\n[TEST] Test LCD... "));

  displayMessage("Test LCD", "Backlight.. .", 1000);
  for (int i = 0; i < 3; i++) {
    lcd.noBacklight();
    delay(300);
    lcd.backlight();
    delay(300);
  }

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Icones:  ");
  lcd.write(0);
  lcd.write(1);
  lcd.write(2);
  lcd.write(3);
  delay(2000);

  Serial.println(F("  ✓ Test termine"));
  displayMessage("Test OK!", NULL, 1000);
}

// ==================== INITIALISATION ====================

void printWelcome() {
  Serial.println(
      F("\n╔════════════════════════════════════════════════════════╗"));
  Serial.println(
      F("║   SYSTEME DE PREDICTION DU MILDIOU - VERSION PROD     ║"));
  Serial.println(
      F("║   Arduino UNO R4 WiFi + Capteurs                       ║"));
  Serial.println(
      F("╚════════════════════════════════════════════════════════╝"));
}

void initializeHistory() {
  for (int i = 0; i < HISTORY_SIZE; i++) {
    history[i].meantemp = 0;
    history[i].humidity = 0;
    history[i].meanpressure = PRESSURE_NORMAL;
    history[i].valid = false;
  }
  historyIndex = 0;
  validDays = 0;
  Serial.println(F("[INIT] Historique initialise"));
}

void initializeWeights() {
  Serial.println(F("[INIT] Chargement poids NN..."));

  for (int i = 0; i < NUM_FEATURES; i++) {
    feature_means[i] = FEATURE_MEANS[i];
    feature_stds[i] = FEATURE_STDS[i];
    if (feature_stds[i] == 0)
      feature_stds[i] = 1.0;
  }

  int idx = 0;
  for (int i = 0; i < L1_IN; i++) {
    for (int j = 0; j < L1_OUT; j++) {
      weights_L1[i][j] = WEIGHTS_L1[idx++];
    }
  }
  for (int j = 0; j < L1_OUT; j++) {
    bias_L1[j] = BIAS_L1[j];
  }

  idx = 0;
  for (int i = 0; i < L2_IN; i++) {
    for (int j = 0; j < L2_OUT; j++) {
      weights_L2[i][j] = WEIGHTS_L2[idx++];
    }
  }
  for (int j = 0; j < L2_OUT; j++) {
    bias_L2[j] = BIAS_L2[j];
  }

  idx = 0;
  for (int i = 0; i < L3_IN; i++) {
    for (int j = 0; j < L3_OUT; j++) {
      weights_L3[i][j] = WEIGHTS_L3[idx++];
    }
  }
  for (int j = 0; j < L3_OUT; j++) {
    bias_L3[j] = BIAS_L3[j];
  }

  Serial.print(F("        Architecture: "));
  Serial.print(L1_IN);
  Serial.print(F(" -> "));
  Serial.print(L1_OUT);
  Serial.print(F(" -> "));
  Serial.print(L2_OUT);
  Serial.print(F(" -> "));
  Serial.println(L3_OUT);
  Serial.println(F("        ✓ Reseau ENTRAINE charge"));
}

// ==================== FEEDBACK LOOP EEPROM ====================

void initializeEEPROM() {
  // Lire le compteur d'erreurs
  uint8_t errorCount = EEPROM.read(EEPROM_ERROR_COUNT_ADDR);

  // Si valeur invalide (EEPROM vierge = 0xFF), initialiser à 0
  if (errorCount > MAX_ERROR_SAMPLES) {
    EEPROM.write(EEPROM_ERROR_COUNT_ADDR, 0);
    errorCount = 0;
  }

  Serial.print(F("[EEPROM] "));
  Serial.print(errorCount);
  Serial.println(F(" erreurs sauvegardees"));
}

void saveErrorToEEPROM(uint8_t predicted, uint8_t actual) {
  uint8_t errorCount = EEPROM.read(EEPROM_ERROR_COUNT_ADDR);

  if (errorCount >= MAX_ERROR_SAMPLES) {
    Serial.println(F("[EEPROM] PLEIN! Utilisez DUMP_ERRORS puis CLEAR_ERRORS"));
    return;
  }

  // Créer le sample d'erreur
  int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
  ErrorSample sample;
  sample.day = (uint8_t)(currentDay % 256);
  sample.temp = (int8_t)round(history[current].meantemp);
  sample.humidity = (uint8_t)round(history[current].humidity);
  sample.pressure = (uint16_t)(history[current].meanpressure - 900);
  sample.predicted = predicted;
  sample.actual = actual;

  // Sauvegarder en EEPROM
  int addr = EEPROM_ERROR_DATA_ADDR + errorCount * sizeof(ErrorSample);
  EEPROM.put(addr, sample);

  // Incrémenter le compteur
  errorCount++;
  EEPROM.write(EEPROM_ERROR_COUNT_ADDR, errorCount);

  Serial.print(F("[EEPROM] ✓ Erreur #"));
  Serial.print(errorCount);
  Serial.print(F(" sauvegardee (predit:"));
  Serial.print(predicted);
  Serial.print(F(", reel:"));
  Serial.print(actual);
  Serial.println(F(")"));

  lcd.clear();
  lcd.print("Correction OK!");
  lcd.setCursor(0, 1);
  lcd.print("Erreur #");
  lcd.print(errorCount);
  delay(2000);
  displayCurrentData();
}

void dumpErrorsToSerial() {
  uint8_t errorCount = EEPROM.read(EEPROM_ERROR_COUNT_ADDR);

  Serial.println(F("\n=== EXPORT ERREURS (CSV) ==="));
  Serial.println(F("day,temp,humidity,pressure,predicted,actual"));

  for (int i = 0; i < errorCount; i++) {
    int addr = EEPROM_ERROR_DATA_ADDR + i * sizeof(ErrorSample);
    ErrorSample sample;
    EEPROM.get(addr, sample);

    Serial.print(sample.day);
    Serial.print(",");
    Serial.print(sample.temp);
    Serial.print(",");
    Serial.print(sample.humidity);
    Serial.print(",");
    Serial.print(sample.pressure + 900);
    Serial.print(",");
    Serial.print(sample.predicted);
    Serial.print(",");
    Serial.println(sample.actual);
  }

  Serial.print(F("=== Total: "));
  Serial.print(errorCount);
  Serial.println(F(" erreurs ===\n"));
}

void clearErrorsFromEEPROM() {
  EEPROM.write(EEPROM_ERROR_COUNT_ADDR, 0);
  Serial.println(F("[EEPROM] ✓ Toutes les erreurs effacees"));

  lcd.clear();
  lcd.print("EEPROM effacee");
  delay(2000);
  displayCurrentData();
}

// ==================== CALCUL FEATURES ====================

void calculateFeatures() {
  int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;

  float temp = history[current].meantemp;
  float humidity = history[current].humidity;
  float pressure = history[current].meanpressure;

  float temp_sum5 = 0, hum_sum5 = 0, press_sum5 = 0;
  float temp_sum7 = 0, hum_sum7 = 0;
  float hum_max7 = 0, press_min7 = 9999;
  float temp_sq_sum7 = 0;

  int count5 = min(validDays, 5);
  int count7 = min(validDays, 7);

  for (int i = 0; i < count7; i++) {
    int idx = (current - i + HISTORY_SIZE) % HISTORY_SIZE;

    if (i < count5) {
      temp_sum5 += history[idx].meantemp;
      hum_sum5 += history[idx].humidity;
      press_sum5 += history[idx].meanpressure;
    }

    temp_sum7 += history[idx].meantemp;
    hum_sum7 += history[idx].humidity;

    if (history[idx].humidity > hum_max7)
      hum_max7 = history[idx].humidity;
    if (history[idx].meanpressure < press_min7)
      press_min7 = history[idx].meanpressure;

    temp_sq_sum7 += history[idx].meantemp * history[idx].meantemp;
  }

  float temp_ma5 = (count5 > 0) ? temp_sum5 / count5 : temp;
  float temp_ma7 = (count7 > 0) ? temp_sum7 / count7 : temp;
  float hum_ma5 = (count5 > 0) ? hum_sum5 / count5 : humidity;
  float hum_ma7 = (count7 > 0) ? hum_sum7 / count7 : humidity;
  float press_ma5 = (count5 > 0) ? press_sum5 / count5 : pressure;

  float temp_variance =
      (count7 > 0) ? (temp_sq_sum7 / count7) - (temp_ma7 * temp_ma7) : 0;
  float temp_std7 = (temp_variance > 0) ? sqrt(temp_variance) : 0;

  float temp_trend_3d = 0, hum_trend_3d = 0, press_trend_3d = 0;
  if (validDays >= 3) {
    int idx3 = (current - 3 + HISTORY_SIZE) % HISTORY_SIZE;
    temp_trend_3d = temp - history[idx3].meantemp;
    hum_trend_3d = humidity - history[idx3].humidity;
    press_trend_3d = pressure - history[idx3].meanpressure;
  }

  float pressure_anomaly = pressure - PRESSURE_NORMAL;
  float pressure_dropping =
      (press_trend_3d < -PRESSURE_DROP_THRESHOLD) ? 1.0 : 0.0;

  int consecutive_temp = 0, consecutive_hum = 0;
  int consecutive_both = 0, consecutive_low_press = 0;

  for (int i = 0; i < validDays && i < 14; i++) {
    int idx = (current - i + HISTORY_SIZE) % HISTORY_SIZE;

    bool temp_ok = (history[idx].meantemp >= TEMP_MIN_FAVORABLE &&
                    history[idx].meantemp <= TEMP_MAX_FAVORABLE);
    bool hum_ok = (history[idx].humidity >= HUMIDITY_THRESHOLD);
    bool press_low = (history[idx].meanpressure < PRESSURE_LOW);

    if (temp_ok && consecutive_temp == i)
      consecutive_temp++;
    if (hum_ok && consecutive_hum == i)
      consecutive_hum++;
    if (temp_ok && hum_ok && consecutive_both == i)
      consecutive_both++;
    if (press_low && consecutive_low_press == i)
      consecutive_low_press++;
  }

  float risk_acc = calculateRiskAccumulator();

  int day_of_year = (currentDay % 365) + 1;
  float day_sin = sin(2.0 * PI * day_of_year / 365.0);
  float day_cos = cos(2.0 * PI * day_of_year / 365.0);
  int month = (day_of_year / 30) + 1;
  float high_risk_season = (month >= 3 && month <= 10) ? 1.0 : 0.0;

  float temp_hum_interaction = (temp / 40.0) * (humidity / 100.0);
  float hum_press_interaction =
      (humidity / 100.0) * (1.0 - (pressure - 990.0) / 40.0);
  hum_press_interaction = constrain(hum_press_interaction, 0.0, 1.0);

  int f = 0;
  inputFeatures[f++] = temp;
  inputFeatures[f++] = humidity;
  inputFeatures[f++] = pressure;
  inputFeatures[f++] = temp_trend_3d;
  inputFeatures[f++] = hum_trend_3d;
  inputFeatures[f++] = press_trend_3d;
  inputFeatures[f++] = temp_ma5;
  inputFeatures[f++] = hum_ma5;
  inputFeatures[f++] = press_ma5;
  inputFeatures[f++] = hum_ma7;
  inputFeatures[f++] = temp_std7;
  inputFeatures[f++] = hum_max7;
  inputFeatures[f++] = press_min7;
  inputFeatures[f++] = pressure_anomaly;
  inputFeatures[f++] = pressure_dropping;
  inputFeatures[f++] = (float)consecutive_temp;
  inputFeatures[f++] = (float)consecutive_hum;
  inputFeatures[f++] = (float)consecutive_both;
  inputFeatures[f++] = (float)consecutive_low_press;
  inputFeatures[f++] = risk_acc;
  inputFeatures[f++] = day_sin;
  inputFeatures[f++] = day_cos;
  inputFeatures[f++] = high_risk_season;
  inputFeatures[f++] = temp_hum_interaction;
  inputFeatures[f++] = hum_press_interaction;

  for (int i = 0; i < NUM_FEATURES; i++) {
    inputFeatures[i] = (inputFeatures[i] - feature_means[i]) / feature_stds[i];
  }
}

float calculateRiskAccumulator() {
  float risk = 0;
  float decay = 0.85;
  int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;

  for (int i = 0; i < min(validDays, 7); i++) {
    int idx = (current - i + HISTORY_SIZE) % HISTORY_SIZE;
    float instant_risk = 0;

    float t = history[idx].meantemp;
    if (t >= TEMP_OPTIMAL_MIN && t <= TEMP_OPTIMAL_MAX) {
      instant_risk += 0.30;
    } else if (t >= TEMP_MIN_FAVORABLE && t <= TEMP_MAX_FAVORABLE) {
      instant_risk += 0.15;
    }

    float h = history[idx].humidity;
    if (h >= HUMIDITY_CRITICAL) {
      instant_risk += 0.35;
    } else if (h >= HUMIDITY_THRESHOLD) {
      instant_risk += 0.20;
    }

    float p = history[idx].meanpressure;
    if (p < PRESSURE_LOW) {
      instant_risk += 0.25;
    } else if (p < PRESSURE_NORMAL) {
      instant_risk += 0.10;
    }

    risk += instant_risk * pow(decay, i);
  }

  return min(risk / 2.0, 1.0);
}

// ==================== RÉSEAU DE NEURONES ====================

float sigmoid(float x) {
  if (x < -10)
    return 0.0;
  if (x > 10)
    return 1.0;
  return 1.0 / (1.0 + exp(-x));
}

void softmax(float *input, float *output, int size) {
  float maxVal = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > maxVal)
      maxVal = input[i];
  }

  float sum = 0;
  for (int i = 0; i < size; i++) {
    output[i] = exp(input[i] - maxVal);
    sum += output[i];
  }

  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

int makePrediction() {
  calculateFeatures();

  float layer1[L1_OUT];
  for (int j = 0; j < L1_OUT; j++) {
    float sum = bias_L1[j];
    for (int i = 0; i < L1_IN; i++) {
      sum += inputFeatures[i] * weights_L1[i][j];
    }
    layer1[j] = sigmoid(sum);
  }

  float layer2[L2_OUT];
  for (int j = 0; j < L2_OUT; j++) {
    float sum = bias_L2[j];
    for (int i = 0; i < L2_IN; i++) {
      sum += layer1[i] * weights_L2[i][j];
    }
    layer2[j] = sigmoid(sum);
  }

  float layer3_raw[L3_OUT];
  float output[L3_OUT];

  for (int j = 0; j < L3_OUT; j++) {
    float sum = bias_L3[j];
    for (int i = 0; i < L3_IN; i++) {
      sum += layer2[i] * weights_L3[i][j];
    }
    layer3_raw[j] = sum;
  }

  softmax(layer3_raw, output, L3_OUT);

  for (int i = 0; i < 3; i++) {
    lastProba[i] = output[i];
  }

  int prediction = 0;
  float maxProb = output[0];
  for (int i = 1; i < L3_OUT; i++) {
    if (output[i] > maxProb) {
      maxProb = output[i];
      prediction = i;
    }
  }

  lastPrediction = prediction; // Sauvegarder pour feedback loop
  printPredictionResults(output, prediction, maxProb);

  return prediction;
}

void printPredictionResults(float *proba, int prediction, float confidence) {
  Serial.println(F("\n┌────────────────────────────────────────┐"));
  Serial.println(F("│          RESULTAT PREDICTION           │"));
  Serial.println(F("├────────────────────────────────────────┤"));

  Serial.print(F("│  Risque FAIBLE:    "));
  printProbability(proba[0]);
  Serial.println(F("             │"));

  Serial.print(F("│  Risque MOYEN:   "));
  printProbability(proba[1]);
  Serial.println(F("             │"));

  Serial.print(F("│  Risque ELEVE:   "));
  printProbability(proba[2]);
  Serial.println(F("             │"));

  Serial.println(F("├────────────────────────────────────────┤"));
  Serial.print(F("│  -> PREDICTION: "));

  switch (prediction) {
  case 0:
    Serial.println(F("FAIBLE  ✓           │"));
    break;
  case 1:
    Serial.println(F("MOYEN   ⚠           │"));
    break;
  case 2:
    Serial.println(F("ELEVE   ⚠⚠⚠          │"));
    break;
  }

  Serial.print(F("│  Confiance: "));
  printProbability(confidence);
  Serial.println(F("                   │"));

  Serial.println(F("└───────────────────��────────────────────┘"));

  displayRiskLevel(prediction, confidence);

  Serial.println();
  switch (prediction) {
  case 0:
    Serial.println(F(">>> Pas de traitement necessaire. "));
    break;
  case 1:
    Serial.println(F(">>> Surveiller evolution.  Preparer traitement."));
    break;
  case 2:
    Serial.println(F(">>> ! !! TRAITEMENT RECOMMANDE DANS 24-48H !!! "));
    break;
  }
}

void printProbability(float prob) {
  float pct = prob * 100;
  if (pct < 10)
    Serial.print(F(" "));
  Serial.print(pct, 1);
  Serial.print(F(" %"));
}

// ==================== AFFICHAGE ====================

void printHistory() {
  Serial.println(F("\n═══════════════════════════════════════════"));
  Serial.println(F("          HISTORIQUE"));
  Serial.println(F("═══════════════════════════════════════════"));
  Serial.println(F(" Jour |  Temp |  Hum  | Press | OK"));
  Serial.println(F("──────┼───────┼───────┼───────┼────"));

  for (int i = 0; i < HISTORY_SIZE; i++) {
    Serial.print(F("  "));
    if (i < 10)
      Serial.print(F(" "));
    Serial.print(i);
    Serial.print(F("  | "));

    if (history[i].valid) {
      Serial.print(history[i].meantemp, 1);
      Serial.print(F("C | "));
      Serial.print(history[i].humidity, 1);
      Serial.print(F("% | "));
      Serial.print(history[i].meanpressure, 0);
      Serial.println(F(" | ✓"));
    } else {
      Serial.println(F(" --  |  --  |  --  | ✗"));
    }
  }
  Serial.println(F("═══════════════════════════════════════════\n"));
}

void printStatus() {
  Serial.println(F("\n─────────── ETAT SYSTEME ───────────"));
  Serial.print(F("  Jours archives: "));
  Serial.print(validDays);
  Serial.print(F("/"));
  Serial.println(HISTORY_SIZE);

  Serial.print(F("  Mesures aujourd'hui: "));
  Serial.println(measureCount);

  if (measureCount > 0) {
    Serial.println(F("  Moyennes jour en cours: "));
    Serial.print(F("    T="));
    Serial.print(tempSum / measureCount, 1);
    Serial.println(F("C"));
    Serial.print(F("    H="));
    Serial.print(humSum / measureCount, 1);
    Serial.println(F("%"));
    Serial.print(F("    P="));
    Serial.print(pressSum / measureCount, 1);
    Serial.println(F("hPa"));
  }

  if (validDays > 0) {
    int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
    Serial.println(F("  Dernier jour archive:"));
    Serial.print(F("    T="));
    Serial.print(history[current].meantemp, 1);
    Serial.println(F("C"));
    Serial.print(F("    H="));
    Serial.print(history[current].humidity, 1);
    Serial.println(F("%"));
    Serial.print(F("    P="));
    Serial.print(history[current].meanpressure, 1);
    Serial.println(F("hPa"));

    float risk = calculateRiskAccumulator();
    Serial.print(F("  Risque acc:  "));
    Serial.print(risk * 100, 1);
    Serial.println(F("%"));
  }

  // Temps avant prochaine mesure
  unsigned long nextMeasure = MEASURE_INTERVAL - (millis() - lastMeasureTime);
  Serial.print(F("  Prochaine mesure dans: "));
  Serial.print(nextMeasure / 60000);
  Serial.println(F(" min"));

  // Temps avant fin de journée
  unsigned long nextDay = DAY_DURATION - (millis() - dayStartTime);
  Serial.print(F("  Fin de journee dans: "));
  Serial.print(nextDay / 3600000);
  Serial.print(F("h "));
  Serial.print((nextDay % 3600000) / 60000);
  Serial.println(F("min"));

  Serial.println(F("────────────────────────────────────\n"));
}

void printHelp() {
  Serial.println(F("\n======== COMMANDES - VERSION PRODUCTION V2 ========"));
  Serial.println(F("=== STANDARD ==="));
  Serial.println(F("  help       - Affiche cette aide"));
  Serial.println(F("  status     - Etat du systeme"));
  Serial.println(F("  history    - Historique des donnees"));
  Serial.println(F("  predict    - Force une prediction"));
  Serial.println(F("=== CAPTEURS ==="));
  Serial.println(F("  measure    - Force une mesure immediate"));
  Serial.println(F("  endday     - Force la fin de journee (TEST)"));
  Serial.println(F("  sensors    - Verifie les capteurs"));
  Serial.println(F("=== LCD ==="));
  Serial.println(F("  testlcd    - Test l'ecran"));
  Serial.println(F("  display    - Rafraichit LCD"));
  Serial.println(F("=== FEEDBACK LOOP (NOUVEAU) ==="));
  Serial.println(
      F("  CORRECT_PREDICTION <0|1|2>  - Corrige derniere prediction"));
  Serial.println(
      F("                                (0=Faible, 1=Moyen, 2=Eleve)"));
  Serial.println(F("  dump_errors  - Export CSV des erreurs sauvegardees"));
  Serial.println(F("  clear_errors - Efface toutes les erreurs de l'EEPROM"));
  Serial.println(F("=== AUTOMATIQUE ==="));
  Serial.println(F("  - Mesure toutes les heures"));
  Serial.println(F("  - Moyenne calculee a minuit (24h)"));
  Serial.println(F("  - Prediction auto si >= 3 jours"));
  Serial.println(F("  - Watchdog Timer actif (8s reboot)"));
  Serial.println(F("==================================================\n"));
}

// ==================== COMMANDES ====================

void handleSerialCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    String cmdLower = cmd;
    cmdLower.toLowerCase();

    Serial.print(F("\n> "));
    Serial.println(cmd);

    // === COMMANDES STANDARD ===
    if (cmdLower == "help" || cmdLower == "h" || cmdLower == "?") {
      printHelp();
    } else if (cmdLower == "status" || cmdLower == "s") {
      printStatus();
    } else if (cmdLower == "history" || cmdLower == "hist") {
      printHistory();
    } else if (cmdLower == "predict" || cmdLower == "p") {
      if (validDays >= MIN_DAYS_FOR_PREDICTION) {
        makePrediction();
      } else {
        Serial.print(F("Pas assez de donnees ("));
        Serial.print(validDays);
        Serial.print(F("/"));
        Serial.print(MIN_DAYS_FOR_PREDICTION);
        Serial.println(F(")"));
        displayMessage("Pas assez de", "donnees!", 2000);
        displayCurrentData();
      }
    } else if (cmdLower == "measure" || cmdLower == "m") {
      takeRealMeasurement();
    } else if (cmdLower == "endday") {
      Serial.println(F("Force fin de journee (TEST)"));
      endOfDayProduction();
    } else if (cmdLower == "sensors") {
      initRealSensors();
      delay(2000);
      displayCurrentData();
    } else if (cmdLower == "testlcd" || cmdLower == "lcd") {
      lcdTest();
      displayCurrentData();
    } else if (cmdLower == "display" || cmdLower == "d") {
      displayCurrentData();

      // === COMMANDES FEEDBACK LOOP (NOUVEAU) ===
    } else if (cmd.startsWith("CORRECT_PREDICTION ") ||
               cmd.startsWith("correct_prediction ")) {
      // Syntaxe: CORRECT_PREDICTION <0|1|2>
      int actualLabel = cmd.substring(19).toInt();
      if (actualLabel >= 0 && actualLabel <= 2 && lastPrediction >= 0) {
        if (lastPrediction != actualLabel) {
          saveErrorToEEPROM((uint8_t)lastPrediction, (uint8_t)actualLabel);
        } else {
          Serial.println(F("Prediction correcte - pas de sauvegarde"));
        }
      } else if (lastPrediction < 0) {
        Serial.println(F("Erreur: Aucune prediction recente"));
      } else {
        Serial.println(F("Erreur: Label doit etre 0, 1 ou 2"));
      }
    } else if (cmdLower == "dump_errors" || cmdLower == "dump") {
      dumpErrorsToSerial();
    } else if (cmdLower == "clear_errors" || cmdLower == "clear") {
      clearErrorsFromEEPROM();

    } else if (cmd.length() > 0) {
      Serial.println(F("Commande inconnue. 'help' pour aide"));
    }
  }
}