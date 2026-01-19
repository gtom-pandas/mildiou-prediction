/*
 * ============================================================
 * SYSTÈME DE PRÉDICTION DU MILDIOU - VERSION DEV
 * ============================================================
 * • MODE SIMULATION uniquement (pas de capteurs réels)
 * • Affichage sur LCD 16x2 I2C
 * • Réseau de neurones entraîné
 * • Pour tests et démonstrations
 *
 * Hardware requis:
 * - Arduino UNO R4 WiFi
 * - LCD 16x2 I2C (adresse 0x27 ou 0x3F)
 *
 * Commandes disponibles:
 * - simulate / highrisk / lowrisk :  Charger scénarios
 * - predict :  Faire une prédiction
 * - testlcd : Tester l'écran
 * - help : Aide complète
 */

// ==================== BIBLIOTHÈQUES ====================
#include "mildiou_nn_weights.h"
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

// ==================== CONFIGURATION LCD ====================
LiquidCrystal_I2C lcd(0x27, 16, 2); // Change en 0x3F si nécessaire

// Caractères personnalisés
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

// ==================== VARIABLES GLOBALES ====================
DailyData history[HISTORY_SIZE];
int historyIndex = 0;
int validDays = 0;
int currentDay = 0;

// Réseau de neurones
const int NUM_FEATURES = 25;
float inputFeatures[NUM_FEATURES];

const int L1_IN = 25, L1_OUT = 12;
const int L2_IN = 12, L2_OUT = 6;
const int L3_IN = 6, L3_OUT = 3;

float weights_L1[L1_IN][L1_OUT];
float bias_L1[L1_OUT];
float weights_L2[L2_IN][L2_OUT];
float bias_L2[L2_OUT];
float weights_L3[L3_IN][L3_OUT];
float bias_L3[L3_OUT];

float feature_means[NUM_FEATURES];
float feature_stds[NUM_FEATURES];

// Dernières probabilités (pour affichage)
float lastProba[3] = {0, 0, 0};

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000)
    ;
  delay(1000);

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
  lcd.print("Version DEV");
  delay(2000);

  printWelcome();
  initializeHistory();
  initializeWeights();

  Serial.println(F("\n✓ Systeme pret (MODE DEV)"));
  Serial.println(F("Tapez 'help' pour les commandes.\n"));

  displayWelcomeScreen();
}

// ==================== LOOP ====================
void loop() {
  handleSerialCommands();
  delay(10);
}

// ==================== AFFICHAGE LCD ====================

void displayWelcomeScreen() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("MODE SIMULATION");
  lcd.setCursor(0, 1);
  lcd.print("Tapez 'simulate'");
}

void displayCurrentData() {
  if (validDays == 0) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Pas de donnees");
    lcd.setCursor(0, 1);
    lcd.print("'simulate' first");
    return;
  }

  int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
  float temp = history[current].meantemp;
  float hum = history[current].humidity;
  float press = history[current].meanpressure;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(0);
  lcd.print((int)temp);
  lcd.print("C ");
  lcd.write(1);
  lcd.print((int)hum);
  lcd.print("%");

  if (press < PRESSURE_LOW)
    lcd.print(" P: lo");
  else if (press > PRESSURE_NORMAL + 5)
    lcd.print(" P:hi");
  else
    lcd.print(" P:ok");

  lcd.setCursor(0, 1);
  lcd.print("Jours:");
  lcd.print(validDays);
  lcd.print("/");
  lcd.print(HISTORY_SIZE);
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

  // Écran 1: Confiance
  lcd.setCursor(0, 0);
  lcd.print("Confiance:");
  lcd.setCursor(0, 1);
  lcd.print((int)(confidence * 100));
  lcd.print("%");
  delay(2500);

  // Écran 2: Détails risques
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

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("AAAAAAAAAAAAAAAA");
  lcd.setCursor(0, 1);
  lcd.print("AAAAAAAAAAAAAAAA");
  delay(1500);

  Serial.println(F("  ✓ Test termine"));
  displayMessage("Test OK!", "LCD fonctionne", 2000);
}

// ==================== INITIALISATION ====================

void printWelcome() {
  Serial.println(
      F("\n╔════════════════════════════════════════════════════════╗"));
  Serial.println(
      F("║   SYSTEME DE PREDICTION DU MILDIOU - VERSION DEV      ║"));
  Serial.println(
      F("║   Arduino UNO R4 WiFi + LCD 16x2                       ║"));
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

  Serial.print(F("        Architecture:  "));
  Serial.print(L1_IN);
  Serial.print(F(" -> "));
  Serial.print(L1_OUT);
  Serial.print(F(" -> "));
  Serial.print(L2_OUT);
  Serial.print(F(" -> "));
  Serial.println(L3_OUT);
  Serial.println(F("        ✓ Reseau ENTRAINE charge"));
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

  // Sauvegarder pour affichage
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

  printPredictionResults(output, prediction, maxProb);

  return prediction;
}

void printPredictionResults(float *proba, int prediction, float confidence) {
  Serial.println(F("\n┌────────────────────────────────────────┐"));
  Serial.println(F("│          RESULTAT PREDICTION           │"));
  Serial.println(F("├────────────────────────────────────────┤"));

  Serial.print(F("│  Risque FAIBLE:     "));
  printProbability(proba[0]);
  Serial.println(F("             │"));

  Serial.print(F("│  Risque MOYEN:    "));
  printProbability(proba[1]);
  Serial.println(F("             │"));

  Serial.print(F("│  Risque ELEVE:    "));
  printProbability(proba[2]);
  Serial.println(F("             │"));

  Serial.println(F("├────────────────────────────────────────┤"));
  Serial.print(F("│  -> PREDICTION:  "));

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

  Serial.print(F("│  Confiance:  "));
  printProbability(confidence);
  Serial.println(F("                   │"));

  Serial.println(F("└────────────────────────────────────────┘"));

  displayRiskLevel(prediction, confidence);

  Serial.println();
  switch (prediction) {
  case 0:
    Serial.println(F(">>> Pas de traitement necessaire. "));
    break;
  case 1:
    Serial.println(F(">>> Surveiller l'evolution.  Preparer traitement."));
    break;
  case 2:
    Serial.println(F(">>> ! !! TRAITEMENT RECOMMANDE DANS 24-48H !!!"));
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

// ==================== SIMULATIONS ====================

void simulateData() {
  Serial.println(F("\n═══════════════════════════════════════════"));
  Serial.println(F("      SIMULATION 7 JOURS NORMAUX"));
  Serial.println(F("═══════════════════════════════════════════"));

  float temps[] = {18.0, 20.0, 22.0, 21.0, 23.0, 19.0, 17.0};
  float hums[] = {65.0, 72.0, 78.0, 85.0, 88.0, 82.0, 75.0};
  float press[] = {1015.0, 1012.0, 1008.0, 1005.0, 1006.0, 1010.0, 1013.0};

  initializeHistory();

  displayMessage("Chargement.. .", "7 jours", 1000);

  for (int i = 0; i < 7; i++) {
    history[i].meantemp = temps[i];
    history[i].humidity = hums[i];
    history[i].meanpressure = press[i];
    history[i].valid = true;

    Serial.print(F("  Jour "));
    Serial.print(i + 1);
    Serial.print(F(":  T="));
    Serial.print(temps[i], 1);
    Serial.print(F("C H="));
    Serial.print(hums[i], 1);
    Serial.print(F("% P="));
    Serial.print(press[i], 1);
    Serial.println(F("hPa"));
  }

  historyIndex = 7;
  validDays = 7;
  currentDay = 7;

  Serial.println(F("\n  ✓ Simulation OK"));
  Serial.println(F("  Tapez 'predict'\n"));

  displayMessage("Donnees OK!", "Tapez 'predict'", 2000);
  displayCurrentData();
}

void simulateHighRisk() {
  Serial.println(F("\n═══════════════════════════════════════════"));
  Serial.println(F("    SIMULATION RISQUE ELEVE"));
  Serial.println(F("═══════════════════════════════════════════"));

  float temps[] = {20.0, 21.0, 22.0, 23.0, 22.0, 21.0, 20.0};
  float hums[] = {75.0, 82.0, 88.0, 92.0, 95.0, 93.0, 90.0};
  float press[] = {1012.0, 1008.0, 1004.0, 1002.0, 1001.0, 1003.0, 1005.0};

  initializeHistory();
  displayMessage("SCENARIO:", "Risque eleve", 1000);

  for (int i = 0; i < 7; i++) {
    history[i].meantemp = temps[i];
    history[i].humidity = hums[i];
    history[i].meanpressure = press[i];
    history[i].valid = true;

    Serial.print(F("  Jour "));
    Serial.print(i + 1);
    Serial.print(F(": T="));
    Serial.print(temps[i], 1);
    Serial.print(F("C H="));
    Serial.print(hums[i], 1);
    Serial.print(F("% P="));
    Serial.print(press[i], 1);
    Serial.println(F("hPa"));
  }

  historyIndex = 7;
  validDays = 7;
  currentDay = 7;

  Serial.println(F("\n  ✓ Scenario charge"));
  displayMessage("Donnees OK!", "Tapez 'predict'", 2000);
  displayCurrentData();
}

void simulateLowRisk() {
  Serial.println(F("\n═══════════════════════════════════════════"));
  Serial.println(F("    SIMULATION RISQUE FAIBLE"));
  Serial.println(F("═══════════════════════════════════════════"));

  float temps[] = {15.0, 18.0, 22.0, 25.0, 28.0, 26.0, 23.0};
  float hums[] = {45.0, 50.0, 48.0, 52.0, 55.0, 50.0, 48.0};
  float press[] = {1018.0, 1020.0, 1022.0, 1019.0, 1017.0, 1018.0, 1020.0};

  initializeHistory();
  displayMessage("SCENARIO:", "Risque faible", 1000);

  for (int i = 0; i < 7; i++) {
    history[i].meantemp = temps[i];
    history[i].humidity = hums[i];
    history[i].meanpressure = press[i];
    history[i].valid = true;

    Serial.print(F("  Jour "));
    Serial.print(i + 1);
    Serial.print(F(": T="));
    Serial.print(temps[i], 1);
    Serial.print(F("C H="));
    Serial.print(hums[i], 1);
    Serial.print(F("% P="));
    Serial.print(press[i], 1);
    Serial.println(F("hPa"));
  }

  historyIndex = 7;
  validDays = 7;
  currentDay = 7;

  Serial.println(F("\n  ✓ Scenario charge"));
  displayMessage("Donnees OK!", "Tapez 'predict'", 2000);
  displayCurrentData();
}

// ==================== AFFICHAGE ====================

void printHistory() {
  Serial.println(F("\n═══════════════════════════════════════════"));
  Serial.println(F("          HISTORIQUE"));
  Serial.println(F("═══════════════════════════════════════════"));
  Serial.println(F(" Jour |  Temp |  Hum  | Press | OK"));
  Serial.println(F("────���─┼───────┼───────┼───────┼────"));

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
  Serial.println(F("\n─────────── ETAT ───────────"));
  Serial.print(F("  Jours:  "));
  Serial.print(validDays);
  Serial.print(F("/"));
  Serial.println(HISTORY_SIZE);

  if (validDays > 0) {
    int current = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
    Serial.println(F("  Dernieres valeurs: "));
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
  Serial.println(F("────────────────────────────\n"));
}

void printHelp() {
  Serial.println(
      F("\n╔════════════════════════════════════════════════════════╗"));
  Serial.println(
      F("║              COMMANDES - VERSION DEV                   ║"));
  Serial.println(
      F("╠════════════════════════════════════════════════════════╣"));
  Serial.println(
      F("║  help       - Affiche cette aide                       ║"));
  Serial.println(
      F("║  status     - Etat du systeme                          ║"));
  Serial.println(
      F("║  history    - Historique des donnees                   ��"));
  Serial.println(
      F("║  predict    - Faire une prediction                     ║"));
  Serial.println(
      F("╠════════════════════════════════════════════════════════╣"));
  Serial.println(
      F("║  SIMULATIONS:                                            ║"));
  Serial.println(
      F("║  simulate   - Charger 7 jours normaux                  ║"));
  Serial.println(
      F("║  highrisk   - Scenario risque eleve                    ║"));
  Serial.println(
      F("║  lowrisk    - Scenario risque faible                   ║"));
  Serial.println(
      F("║  reset      - Reinitialiser                            ║"));
  Serial.println(
      F("╠════════════════════════════════════════════════════════╣"));
  Serial.println(
      F("║  LCD:                                                    ║"));
  Serial.println(
      F("║  testlcd    - Tester l'ecran                           ║"));
  Serial.println(
      F("║  display    - Rafraichir LCD                           ║"));
  Serial.println(
      F("╚════════════════════════════════════════════════════════╝\n"));
}

// ==================== COMMANDES ====================

void handleSerialCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toLowerCase();

    Serial.print(F("\n> "));
    Serial.println(cmd);

    if (cmd == "help" || cmd == "h" || cmd == "?") {
      printHelp();
    } else if (cmd == "status" || cmd == "s") {
      printStatus();
    } else if (cmd == "history" || cmd == "hist") {
      printHistory();
    } else if (cmd == "predict" || cmd == "p") {
      if (validDays >= MIN_DAYS_FOR_PREDICTION) {
        makePrediction();
      } else {
        Serial.print(F("⚠ Pas assez de donnees ("));
        Serial.print(validDays);
        Serial.print(F("/"));
        Serial.print(MIN_DAYS_FOR_PREDICTION);
        Serial.println(F(")"));
        displayMessage("Pas assez de", "donnees!", 2000);
      }
    } else if (cmd == "simulate" || cmd == "sim") {
      simulateData();
    } else if (cmd == "highrisk" || cmd == "high") {
      simulateHighRisk();
    } else if (cmd == "lowrisk" || cmd == "low") {
      simulateLowRisk();
    } else if (cmd == "reset" || cmd == "r") {
      Serial.println(F("Reinitialisation..."));
      initializeHistory();
      displayMessage("RESET", "OK", 1000);
      displayWelcomeScreen();
      Serial.println(F("✓ OK"));
    } else if (cmd == "testlcd" || cmd == "lcd") {
      lcdTest();
    } else if (cmd == "display" || cmd == "d") {
      displayCurrentData();
    } else if (cmd.length() > 0) {
      Serial.println(F("Commande inconnue.  'help' pour aide"));
    }
  }
}