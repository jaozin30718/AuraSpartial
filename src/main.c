#include <stdio.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_system.h"
#include "esp_log.h"

// Definição dos pinos informados
#define I2S_WS 36
#define I2S_SCK 37
#define I2S_SD 35
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000

static const char *TAG = "INMP441_TEST";

i2s_chan_handle_t rx_chan;

static void i2s_init(void)
{
    ESP_LOGI(TAG, "Configurando canal I2S rx...");
    i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &rx_chan));

    ESP_LOGI(TAG, "Configurando modo standard do I2S...");
    i2s_std_config_t rx_std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK,
            .ws   = I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_SD,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    
    rx_std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &rx_std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "I2S inicializado com sucesso.");
}

void app_main(void)
{
    ESP_LOGI(TAG, "Iniciando teste do INMP441...");
    i2s_init();

    size_t bytes_read = 0;
    const int read_len = 1024;
    int32_t *read_buff = (int32_t *)calloc(1, read_len);
    assert(read_buff);

    while (1) {
        esp_err_t res = i2s_channel_read(rx_chan, read_buff, read_len, &bytes_read, portMAX_DELAY);
        
        if (res == ESP_OK) {
            int samples_read = bytes_read / sizeof(int32_t);
            
            if (samples_read > 0) {
                int32_t sample = read_buff[0] >> 8; 
                printf("Amostra: %" PRId32 "\n", sample);
            }
        } else {
            ESP_LOGE(TAG, "Erro ao ler I2S");
        }
        
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
