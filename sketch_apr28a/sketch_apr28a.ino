#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>
#include <time.h>
#include <sys/time.h>

// ============================================================
// IDENTIFICAÇÃO DO NÓ
// ============================================================
#define NODE_ID 'A'

// ============================================================
// REDE
// ============================================================
const char* WIFI_SSID     = "CAVALO_DE_TROIA";
const char* WIFI_PASSWORD = "Cavalo1020301@";
const char* UDP_HOST_IP   = "192.168.0.171";
const int   UDP_PORT      = 5005;
WiFiUDP udp;

// Porta de controle para receber offset de tempo do Python
#define UDP_CTRL_PORT 5006
WiFiUDP udp_ctrl;

// ============================================================
// PINOS I2S0 - MIC 1 + MIC 2
// ============================================================
#define I2S0_WS_PIN   12
#define I2S0_SCK_PIN  10
#define I2S0_SD_PIN   11
#define I2S0_PORT     I2S_NUM_0

// ============================================================
// PINOS I2S1 - MIC 3
// ============================================================
#define I2S1_WS_PIN   36
#define I2S1_SCK_PIN  35
#define I2S1_SD_PIN   0
#define I2S1_PORT     I2S_NUM_1

// ============================================================
// PARÂMETROS DE ÁUDIO
// ============================================================
#define SAMPLE_RATE   44100
#define BLOCK_SIZE    128   // reduzido para manter o pacote UDP pequeno e estável

// ============================================================
// SINCRONIZAÇÃO
// ============================================================
// Offset calculado pelo servidor Python e enviado de volta via UDP
static int64_t time_offset_us = 0;

// ============================================================
// BUFFERS
// ============================================================
// I2S0: stereo -> Mic1 (L) + Mic2 (R)
static int32_t raw_i2s0[BLOCK_SIZE * 2];

// I2S1: mono -> Mic3
static int32_t raw_i2s1[BLOCK_SIZE];

// Conversão para 16 bits
static int16_t mic1_buf[BLOCK_SIZE];
static int16_t mic2_buf[BLOCK_SIZE];
static int16_t mic3_buf[BLOCK_SIZE];

// ============================================================
// PACOTE UDP
// [0]      uint8   Node ID
// [1-4]    uint32  Frame counter
// [5-12]   uint64  Timestamp I2S0 em µs
// [13-20]  uint64  Timestamp I2S1 em µs
// [21-22]  uint16  Num amostras por bloco
// [23-24]  uint16  Sample rate
// [25+]    int16[] Mic1, Mic2, Mic3 intercalados
// ============================================================
#define HEADER_SIZE 25
static uint8_t pkt_buf[HEADER_SIZE + BLOCK_SIZE * 3 * sizeof(int16_t)];

uint32_t frame_counter = 0;

// ============================================================
// TIMESTAMP CORRIGIDO
// ============================================================
static inline uint64_t get_corrected_time_us() {
  int64_t raw_us = (int64_t)esp_timer_get_time();
  return (uint64_t)(raw_us + time_offset_us);
}

// ============================================================
// I2S SETUP GENÉRICO
// ============================================================
void setup_i2s_port(i2s_port_t port, int ws_pin, int sck_pin, int sd_pin, i2s_channel_fmt_t channel_fmt) {
  i2s_config_t cfg = {};
  cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate          = SAMPLE_RATE;
  cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format       = channel_fmt;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count        = 8;
  cfg.dma_buf_len          = 128;
  cfg.use_apll             = true;
  cfg.tx_desc_auto_clear   = false;
  cfg.fixed_mclk           = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num   = sck_pin;
  pins.ws_io_num    = ws_pin;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = sd_pin;

  ESP_ERROR_CHECK(i2s_driver_install(port, &cfg, 0, NULL));
  ESP_ERROR_CHECK(i2s_set_pin(port, &pins));
  i2s_start(port);

  // Limpa buffers iniciais
  size_t br = 0;
  int32_t flush_buf[BLOCK_SIZE * 2];
  for (int i = 0; i < 4; i++) {
    i2s_read(port, flush_buf, sizeof(flush_buf), &br, pdMS_TO_TICKS(100));
  }
}

// ============================================================
// WIFI + NTP
// ============================================================
void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("[WiFi] Conectando");
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WiFi] FALHA — reiniciando em 3s");
    delay(3000);
    ESP.restart();
  }

  Serial.println("\n[WiFi] IP: " + WiFi.localIP().toString());

  // NTP
  Serial.print("[NTP] Sincronizando");
  configTime(0, 0, "pool.ntp.org", "time.nist.gov", "time.google.com");

  struct timeval tv;
  int tries = 0;
  do {
    delay(500);
    Serial.print(".");
    gettimeofday(&tv, NULL);
    tries++;
  } while (tv.tv_sec < 24 * 3600 && tries < 30);

  if (tv.tv_sec > 24 * 3600) {
    Serial.printf("\n[NTP] Sincronizado: epoch=%lld\n", (long long)tv.tv_sec);
  } else {
    Serial.println("\n[NTP] AVISO: Não sincronizado — timestamps não confiáveis!");
  }

  udp_ctrl.begin(UDP_CTRL_PORT);
  Serial.printf("[CTRL] Porta de controle: %d\n", UDP_CTRL_PORT);
}

// ============================================================
// RECEBE OFFSET DO SERVIDOR PYTHON
// Protocolo: 8 bytes int64 little-endian
// ============================================================
void check_time_offset() {
  int pktSize = udp_ctrl.parsePacket();
  if (pktSize == 8) {
    uint8_t buf[8];
    udp_ctrl.read(buf, 8);
    int64_t new_offset;
    memcpy(&new_offset, buf, 8);

    // suavização para evitar salto brusco
    time_offset_us = (time_offset_us * 3 + new_offset) / 4;

    Serial.printf("[SYNC] Offset aplicado: %lld µs\n", (long long)time_offset_us);
  }
}

// ============================================================
// CONVERSÃO 32-bit -> 16-bit
// INMP441 fornece amostras em 32 bits, com dado útil nos bits altos
// ============================================================
static inline int16_t i32_to_i16(int32_t x) {
  return (int16_t)(x >> 16);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n================================================");
  Serial.printf("  Sistema Acústico v7 — Nó %c (3 Mics)\n", NODE_ID);
  Serial.printf("  RAM livre: %u bytes\n", ESP.getFreeHeap());
  Serial.printf("  Bloco: %d amostras = %.2f ms\n", BLOCK_SIZE, (BLOCK_SIZE * 1000.0f) / SAMPLE_RATE);
  Serial.println("================================================");

  setup_wifi();

  // I2S0 = Mic1 + Mic2
  setup_i2s_port(I2S0_PORT, I2S0_WS_PIN, I2S0_SCK_PIN, I2S0_SD_PIN, I2S_CHANNEL_FMT_RIGHT_LEFT);

  // I2S1 = Mic3
  setup_i2s_port(I2S1_PORT, I2S1_WS_PIN, I2S1_SCK_PIN, I2S1_SD_PIN, I2S_CHANNEL_FMT_ONLY_LEFT);

  Serial.printf("[OK] RAM após init: %u bytes\n", ESP.getFreeHeap());
  Serial.println("[OK] Streaming ativo...\n");
}

// ============================================================
// LOOP
// Captura dos 3 microfones e envio em um único pacote UDP
// ============================================================
void loop() {
  check_time_offset();

  // Timestamp de início do ciclo
  uint64_t ts_i2s0 = get_corrected_time_us();

  size_t br0 = 0;
  esp_err_t err0 = i2s_read(
    I2S0_PORT,
    raw_i2s0,
    sizeof(raw_i2s0),
    &br0,
    pdMS_TO_TICKS(100)
  );

  if (err0 != ESP_OK || br0 == 0) return;

  uint64_t ts_i2s1 = get_corrected_time_us();

  size_t br1 = 0;
  esp_err_t err1 = i2s_read(
    I2S1_PORT,
    raw_i2s1,
    sizeof(raw_i2s1),
    &br1,
    pdMS_TO_TICKS(100)
  );

  if (err1 != ESP_OK || br1 == 0) return;

  int frames0 = br0 / (sizeof(int32_t) * 2);   // stereo
  int frames1 = br1 / sizeof(int32_t);         // mono

  int frames = frames0;
  if (frames1 < frames) frames = frames1;
  if (frames > BLOCK_SIZE) frames = BLOCK_SIZE;
  if (frames <= 0) return;

  // Conversão
  for (int i = 0; i < frames; i++) {
    int32_t l = raw_i2s0[i * 2];
    int32_t r = raw_i2s0[i * 2 + 1];
    int32_t m = raw_i2s1[i];

    mic1_buf[i] = i32_to_i16(l);
    mic2_buf[i] = i32_to_i16(r);
    mic3_buf[i] = i32_to_i16(m);
  }

  // Monta cabeçalho
  frame_counter++;

  uint16_t n_samples   = (uint16_t)frames;
  uint16_t sample_rate = (uint16_t)SAMPLE_RATE;

  pkt_buf[0] = (uint8_t)NODE_ID;
  memcpy(pkt_buf + 1,  &frame_counter, 4);
  memcpy(pkt_buf + 5,  &ts_i2s0, 8);
  memcpy(pkt_buf + 13, &ts_i2s1, 8);
  memcpy(pkt_buf + 21, &n_samples, 2);
  memcpy(pkt_buf + 23, &sample_rate, 2);

  // Dados intercalados: Mic1, Mic2, Mic3
  uint8_t* p = pkt_buf + HEADER_SIZE;
  for (int i = 0; i < frames; i++) {
    memcpy(p, &mic1_buf[i], sizeof(int16_t)); p += sizeof(int16_t);
    memcpy(p, &mic2_buf[i], sizeof(int16_t)); p += sizeof(int16_t);
    memcpy(p, &mic3_buf[i], sizeof(int16_t)); p += sizeof(int16_t);
  }

  int pkt_len = HEADER_SIZE + frames * 3 * sizeof(int16_t);

  if (WiFi.status() == WL_CONNECTED) {
    udp.beginPacket(UDP_HOST_IP, UDP_PORT);
    udp.write(pkt_buf, pkt_len);
    udp.endPacket();
  }

  // Log a cada 5s
  static uint32_t last_log = 0;
  static uint32_t last_frame = 0;

  if (millis() - last_log >= 5000) {
    uint32_t elapsed = millis() - last_log;
    uint32_t fps = (elapsed > 0) ? ((frame_counter - last_frame) * 1000 / elapsed) : 0;

    Serial.printf(
      "[STREAM] Nó=%c | frame=#%lu | amostras=%d | %d bytes | fps=%lu | offset=%lld µs | heap=%u\n",
      NODE_ID,
      (unsigned long)frame_counter,
      frames,
      pkt_len,
      (unsigned long)fps,
      (long long)time_offset_us,
      ESP.getFreeHeap()
    );

    last_log = millis();
    last_frame = frame_counter;
  }
}