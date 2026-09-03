#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "img_converters.h"

const char* ssid     = "RBCET LAB";
const char* password = "";

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

httpd_handle_t stream_httpd = NULL;

// ============================================================
// SPEED SETTINGS
//   VGA (640x480) = 614 KB per raw frame  -> best detection range
//   QVGA (320x240) = 154 KB per raw frame -> 4x faster WiFi transfer
// For maximum speed set: CAM_FRAMESIZE FRAMESIZE_QVGA, 320, 240
// ============================================================
#define CAM_FRAMESIZE FRAMESIZE_QVGA
#define FRAME_WIDTH   320
#define FRAME_HEIGHT  240

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";


// ============================================================
// STREAM HANDLER
// JPEG via fmt2jpg for browser viewing (sensor already JPEG)
// ============================================================

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
  httpd_resp_set_hdr(req, "X-Framerate", "20");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

      // If sensor already JPEG, send directly; else convert
    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    bool need_free = false;

    if (fb->format == PIXFORMAT_JPEG) {
      jpg_buf = fb->buf;
      jpg_len = fb->len;
      need_free = false;
    } else {
      bool converted = fmt2jpg(
        fb->buf, fb->len, fb->width, fb->height,
        PIXFORMAT_RGB565, 12, &jpg_buf, &jpg_len
      );
      if (!converted || !jpg_buf) {
        Serial.println("JPEG conversion failed");
        esp_camera_fb_return(fb);
        res = ESP_FAIL;
        break;
      }
      need_free = true;
    }

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, jpg_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
    }

    if (need_free) free(jpg_buf);
    esp_camera_fb_return(fb);

    if (res != ESP_OK) break;
  }
  return res;
}


// ============================================================
// CAPTURE HANDLER — JPEG (reliable, good quality)
// Python decodes JPEG (~5ms) then runs model
// ============================================================

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed");
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");

  char wbuf[16], hbuf[16];
  snprintf(wbuf, sizeof(wbuf), "%u", fb->width);
  httpd_resp_set_hdr(req, "X-Frame-Width", wbuf);
  snprintf(hbuf, sizeof(hbuf), "%u", fb->height);
  httpd_resp_set_hdr(req, "X-Frame-Height", hbuf);
  httpd_resp_set_hdr(req, "Connection", "close");

  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  uint32_t len = fb->len, w = fb->width, h = fb->height;
  esp_camera_fb_return(fb);

  if (res == ESP_OK) {
    Serial.printf("[CAPTURE] Sent %u bytes JPEG (%ux%u)\n", len, w, h);
  }
  return res;
}


// ============================================================
// BENCHMARK HANDLER
// Returns ESP32 memory stats as JSON
// ============================================================

static esp_err_t bench_handler(httpd_req_t *req) {
  char json[400];
  snprintf(json, sizeof(json),
    "{"
    "\"heap_free_bytes\": %u,"
    "\"heap_total_bytes\": %u,"
    "\"psram_free_bytes\": %u,"
    "\"psram_total_bytes\": %u,"
    "\"camera_format\": \"JPEG\","
    "\"frame_size\": \"%s\","
    "\"raw_frame_bytes\": %u,"
    "\"resolution\": \"%dx%d\""
    "}",
    ESP.getFreeHeap(),
    ESP.getHeapSize(),
    ESP.getFreePsram(),
    ESP.getPsramSize(),
    CAM_FRAMESIZE == FRAMESIZE_QVGA ? "QVGA" : "VGA",
    0,
    FRAME_WIDTH,
    FRAME_HEIGHT
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Connection", "close");
  return httpd_resp_send(req, json, strlen(json));
}


// ============================================================
// INDEX HANDLER
// ============================================================

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  const char* html =
    "<html><head><title>ESP32-CAM PathIQ</title></head>"
    "<body style='text-align:center;font-family:sans-serif;background:#111;color:#fff;padding:40px;'>"
    "<h2>ESP32-CAM PathIQ AI Detection</h2>"
    "<p>Camera: <b>JPEG VGA</b> — Good quality</p>"
    "<br>"
    "<p><a href='/stream' style='color:#2A9D8F;font-size:18px;'>Live JPEG Stream</a></p>"
    "<p><a href='/capture' style='color:#E76F51;font-size:18px;'>Capture (JPEG)</a></p>"
    "<p><a href='/bench' style='color:#E9C46A;font-size:18px;'>Memory Stats</a></p>"
    "</body></html>";
  return httpd_resp_send(req, html, strlen(html));
}


// ============================================================
// SERVER SETUP
// ============================================================

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.stack_size   = 10240;
  config.task_priority = tskIDLE_PRIORITY + 5;
  config.core_id       = 1;
  config.max_uri_handlers = 5;
  config.recv_wait_timeout = 10;
  config.send_wait_timeout = 10;

  httpd_uri_t index_uri   = { .uri = "/",        .method = HTTP_GET, .handler = index_handler,  .user_ctx = NULL };
  httpd_uri_t stream_uri  = { .uri = "/stream",  .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
  httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler,.user_ctx = NULL };
  httpd_uri_t bench_uri   = { .uri = "/bench",   .method = HTTP_GET, .handler = bench_handler,  .user_ctx = NULL };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &capture_uri);
    httpd_register_uri_handler(stream_httpd, &bench_uri);
  }
}


// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  // JPEG at VGA — reliable WiFi, good quality
  // Browser /stream and Python /capture both JPEG

  if (psramFound()) {
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = CAM_FRAMESIZE;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    Serial.println("PSRAM Found — Using JPEG capture");
    Serial.printf("PSRAM Size: %d MB\n", ESP.getPsramSize() / (1024 * 1024));
    Serial.printf("JPEG frame @VGA quality 12\n");
  } else {
    // Fallback: no PSRAM = cannot hold RGB565 at VGA
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = FRAMESIZE_QVGA;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    Serial.println("PSRAM NOT Found — Falling back to JPEG");
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_framesize(s, CAM_FRAMESIZE);
    s->set_quality(s, 12); // good picture, smaller file → less lag (10=high 63=low)
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 1);
    s->set_gainceiling(s, GAINCEILING_2X);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 0);
    s->set_gain_ctrl(s, 1);
    s->set_sharpness(s, 2);
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
  }

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected!");

  Serial.print("Camera Ready! Open: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();
}


// ============================================================

void loop() {
  delay(10000);
}
