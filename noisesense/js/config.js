/* ==============================================================
   CONFIGURAÇÕES GLOBAIS DO GALPÃO E HEATMAP
   ============================================================== */
export const WAREHOUSE = {
  width:  2.0,   // Sua mesa tem 2 metros de largura
  height: 1.5,   // Sua mesa tem 1.5 metros de profundidade
  // Obstáculos internos (máquinas/estruturas) [x, y, w, h] em metros
  obstacles: [],
  // Paredes internas
  walls: []
};

/* ==============================================================
   PALETA DE CORES DOS NÓS
   ============================================================== */
export const NODE_COLORS = [
  { hex: '#58a6ff', rgb: '88,166,255' },
  { hex: '#f0883e', rgb: '240,136,62' },
  { hex: '#3fb950', rgb: '63,185,80' },
  { hex: '#bc8cff', rgb: '188,140,255' },
  { hex: '#39d0d8', rgb: '57,208,216' },
  { hex: '#f85149', rgb: '248,81,73' },
  { hex: '#d29922', rgb: '210,153,34' },
];

/* ==============================================================
   ESTADO GLOBAL DA APLICAÇÃO (Constantes do Heatmap Acumulativo)
   ============================================================== */
export const HEAT_GRID_RES = 0.05;  // Altíssima resolução (5cm por célula) para mesa de 2m x 1.5m
export const HEAT_COLS = Math.ceil(WAREHOUSE.width  / HEAT_GRID_RES);
export const HEAT_ROWS = Math.ceil(WAREHOUSE.height / HEAT_GRID_RES);
export const HEAT_DECAY = 0.92;     // fator de decaimento por tick (0.92 = fade suave)
export const HEAT_RADIUS = 6;       // Raio base (30cm de influência)
export const HEAT_MAX = 100.0;      // valor máximo de calor
