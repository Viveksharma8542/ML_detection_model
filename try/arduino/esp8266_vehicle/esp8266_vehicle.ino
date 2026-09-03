/*
 * ESP8266 Pothole Alert Receiver
 * -------------------------------
 * Listens for UDP packet "POTHOLE" on port 4210.
 * On reception, pulses GPIO5 (D1) HIGH for 150 ms,
 * triggering the ISD1820 voice module's PLAYE pin.
 *
 * Hardware:
 *   ISD1820 VCC   -> ESP8266 VIN
 *   ISD1820 GND   -> ESP8266 GND
 *   ISD1820 PLAYE -> ESP8266 D1 (GPIO5)
 *   Speaker       -> ISD1820 SP+ / SP-
 *
 * === ARDUINO IDE SETTINGS ===
 * Board:       "Generic ESP8266 Module"  (or "NodeMCU 1.0 (ESP-12E Module)")
 * Flash Size:  "4MB (FS:2MB OTA:~1019KB)"a
 * CPU Freq:    "80 MHz"
 * Flash Mode:  "DIO"
 * Flash Freq:  "40 MHz"
 * Upload Speed:"115200"
 * Port:        Select the correct COM port for your ESP8266
 * Programmer:  "Arduino as ISP"
 *
 * Install ESP8266 board support via:
 *   File -> Preferences -> Additional Boards Manager URLs:
 *   http://arduino.esp8266.com/stable/package_esp8266com_index.json
 *   Then Tools -> Board -> Boards Manager -> search "ESP8266" -> install.
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

const char* WIFI_SSID     = "RBCET LAB";
const char* WIFI_PASSWORD = "";

const uint16_t UDP_PORT   = 4210;
const uint8_t  VOICE_PIN  = 5;
const uint16_t PULSE_MS   = 150;

WiFiUDP udp;
char packetBuffer[64];

void setup() {
  Serial.begin(115200);
  pinMode(VOICE_PIN, OUTPUT);
  digitalWrite(VOICE_PIN, LOW);

  connectToWiFi();

  Serial.print("[UDP] Starting on port ");
  Serial.println(UDP_PORT);
  udp.begin(UDP_PORT);
  Serial.println("[UDP] Ready.");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Connection lost. Reconnecting ...");
    connectToWiFi();
  }

  handleUdpPacket();
  delay(1);
}

void connectToWiFi() {
  Serial.print("[WiFi] Connecting to ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("[WiFi] Connected. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("[WiFi] Failed to connect. Restarting ...");
    ESP.restart();
  }
}

void handleUdpPacket() {
  int packetSize = udp.parsePacket();
  if (packetSize == 0) {
    return;
  }

  int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
  if (len > 0) {
    packetBuffer[len] = '\0';
  }

  Serial.print("[UDP] Received (");
  Serial.print(len);
  Serial.print(" bytes): ");
  Serial.println(packetBuffer);

  if (strcmp(packetBuffer, "POTHOLE") == 0) {
    triggerVoiceModule();
  }
}

void triggerVoiceModule() {
  Serial.println("[VOICE] Triggering ISD1820 ...");
  digitalWrite(VOICE_PIN, HIGH);
  delay(PULSE_MS);
  digitalWrite(VOICE_PIN, LOW);
}
