/*
 * RÉCEPTEUR MILDIOU - VERSION "CHESS ENGINE"
 * Arduino UNO R3 + Shield DFRobot (Variante NHD)
 * * Basé sur la configuration matérielle validée par Chess.pde
 */

#include "U8glib.h"

// ==================== LA LIGNE MAGIQUE (Tirée de votre Chess.pde) ====================
// SCK=13, MOSI=11, CS=10, A0=9, RST=8
U8GLIB_NHD_C12864 u8g(13, 11, 10, 9, 8); 

// ==================== VARIABLES ====================
int jour = 0;
int temperature = 0;
int humidite = 0;
int pression = 0;
int risque = 0;
bool donneesRecues = false;

// ==================== SETUP ====================
void setup() {
  // 1. Sécurité Rétroéclairage (Au cas où)
  pinMode(7, OUTPUT);
  digitalWrite(7, HIGH); 

  // 2. Rotation de l'écran (Comme dans Chess.pde)
  u8g.setRot180();

  // 3. LoRa
  Serial.begin(115200);

  // Config LoRa (Aveugle)
  delay(1000);
  Serial.println(F("AT+Address=2"));
  delay(100);
  Serial.println(F("AT+NETWORKID=18"));
  delay(100);
  Serial.println(F("AT+BAND=868500000"));
  delay(100);
  Serial.println(F("AT+PARAMETER=9,7,1,12"));
}

// ==================== LOOP ====================
void loop() {
  // 1. Écoute Radio (Non bloquante)
  while (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    if (msg.startsWith("+RCV=")) {
      parseLoRaMessage(msg);
      donneesRecues = true;
    }
  }

  // 2. Boucle d'affichage (Picture Loop U8glib)
  u8g.firstPage();  
  do {
    if (donneesRecues) {
      dessinerDonnees();
    } else {
      dessinerAttente();
    }
  } while( u8g.nextPage() );
  
  // Petit délai pour la stabilité
  delay(50);
}

// ==================== DESSIN ====================
void dessinerAttente() {
  u8g.setFont(u8g_font_6x10); // Police standard U8glib
  u8g.drawStr(10, 20, "STATION MILDIOU");
  u8g.drawStr(10, 40, "Attente LoRa...");
  
  // Petit cadre comme dans Chess pour confirmer le dessin
  u8g.drawFrame(0, 0, 128, 64);
}

void dessinerDonnees() {
  u8g.setFont(u8g_font_6x10);
  u8g.drawStr(2, 10, "STATION MILDIOU");
  u8g.drawLine(0, 12, 127, 12);

  // Risque (Texte plus grand)
  u8g.setFont(u8g_font_helvB18); 
  if (risque == 0) u8g.drawStr(15, 35, "FAIBLE");
  else if (risque == 1) u8g.drawStr(15, 35, "MOYEN");
  else u8g.drawStr(15, 35, "ELEVE !");

  // Valeurs
  u8g.setFont(u8g_font_6x10);
  u8g.setPrintPos(0, 50);
  u8g.print("J:"); u8g.print(jour);
  u8g.print(" T:"); u8g.print(temperature); u8g.print("C");
  
  u8g.setPrintPos(0, 62);
  u8g.print("H:"); u8g.print(humidite); u8g.print("% P:"); u8g.print(pression);
}

// ==================== PARSING ====================
void parseLoRaMessage(String msg) {
  int secondComma = msg.indexOf(',', msg.indexOf(',') + 1);
  int dataEnd = msg.lastIndexOf(",-");
  if (dataEnd < 0) dataEnd = msg.length();
  String payload = msg.substring(secondComma + 1, dataEnd);

  if (payload.indexOf("J:") >= 0) jour = extraireValeur(payload, "J:");
  if (payload.indexOf("T:") >= 0) temperature = extraireValeur(payload, "T:");
  if (payload.indexOf("H:") >= 0) humidite = extraireValeur(payload, "H:");
  if (payload.indexOf("P:") >= 0) pression = extraireValeur(payload, "P:");
  if (payload.indexOf("R:") >= 0) risque = extraireValeur(payload, "R:");
}

int extraireValeur(String data, String key) {
  int start = data.indexOf(key);
  if (start == -1) return 0;
  start += key.length();
  int end = data.indexOf(',', start);
  if (end == -1) end = data.length();
  return data.substring(start, end).toInt();
}