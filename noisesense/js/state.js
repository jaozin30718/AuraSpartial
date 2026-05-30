import { HEAT_COLS, HEAT_ROWS } from './config.js';

export const state = {
  nodes: [],           // Array de nós sensores
  selectedNode: null,  // ID do nó selecionado
  refNodeForAdd: null, // Nó de referência para adicionar filho
  simRunning: true,    // Simulação ativa
  showHeatmap: true,   // Exibir heatmap
  showDoA: true,       // Exibir setas DoA
  transientCount: 0,   // Contador de transientes detectados
  activeSources: [],   // Array de fontes ativas (Trianguladas ou Estimadas)
  tick: 0,             // Contador de ticks da simulação
  heatmapData: null,   // Cache do heatmap
  heatmapDirty: true,  // Flag para recalcular heatmap
  peakActive: 0.0,     // Alpha (transparência) da Seta e Esfera
  heatGrid: new Float32Array(HEAT_COLS * HEAT_ROWS),  // Grid acumulativo

  // WebSocket
  ws: null,

  // Canvas e Contextos
  canvas: null,
  ctx: null,
  fftCanvas: null,
  fftCtx: null,
  waveCanvas: null,
  waveCtx: null,
  doaCanvas: null,
  doaCtx: null,

  // Transformação canvas (posição mundo → pixels)
  transform: { scale: 1, offsetX: 0, offsetY: 0, padding: 40 }
};
