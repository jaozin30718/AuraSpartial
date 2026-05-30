import { state } from './state.js';
import { WAREHOUSE, NODE_COLORS } from './config.js';
import { generateIndustrialFFT } from './utils.js';
import { drawColorbar, updateTopbar, updateSidebarLeft, rebuildNodeList, updateSidebarRight, updateClock } from './ui.js';
import { resizeCanvas, drawMap, onCanvasMouseDown, onCanvasRightClick, onCanvasMouseMove, onCanvasMouseUp, onCanvasMouseLeave } from './map.js';
import { initWebSocket } from './websocket.js';

/* ==============================================================
   INICIALIZAÇÃO DO DASHBOARD
   ============================================================== */
export function init() {
  // Obter referências dos canvas e salvar no estado
  state.canvas    = document.getElementById('mapCanvas');
  state.ctx       = state.canvas.getContext('2d');
  state.fftCanvas = document.getElementById('fft-canvas');
  state.fftCtx    = state.fftCanvas.getContext('2d');
  state.waveCanvas= document.getElementById('wave-canvas');
  state.waveCtx   = state.waveCanvas.getContext('2d');
  state.doaCanvas = document.getElementById('doa-compass');
  state.doaCtx    = state.doaCanvas.getContext('2d');

  // Criar nós iniciais
  createInitialNodes();

  // Ajustar tamanho do canvas
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Eventos do mouse no canvas
  state.canvas.addEventListener('mousedown',   onCanvasMouseDown);
  state.canvas.addEventListener('contextmenu', onCanvasRightClick);
  state.canvas.addEventListener('mousemove',   onCanvasMouseMove);
  state.canvas.addEventListener('mouseup',     onCanvasMouseUp);
  state.canvas.addEventListener('mouseleave',  onCanvasMouseLeave);

  // Fechar menus ao clicar fora
  document.addEventListener('click', () => {
    const ctxMenu = document.getElementById('canvas-ctx-menu');
    if (ctxMenu) ctxMenu.classList.remove('open');
  });

  // Colorbar
  drawColorbar();

  // Iniciar WebSocket
  initWebSocket();

  // Loop da Interface (em vez da simulação)
  setInterval(() => {
    if (!state.simRunning) return;
    state.tick++;
    if (state.tick % 5 === 0) state.heatmapDirty = true;
    
    // Fade out da Seta e Esfera de Transientes
    if (state.peakActive > 0) state.peakActive = Math.max(0, state.peakActive - 0.035);
    
    updateTopbar();
    updateSidebarLeft();
    updateSidebarRight();
    drawMap();
  }, 80);

  // Clock
  setInterval(updateClock, 1000);
  updateClock();

  // Selecionar primeiro nó
  selectNode(0);
}

/* ==============================================================
   CRIAR NÓS INICIAIS DE DEMONSTRAÇÃO
   ============================================================== */
function createInitialNodes() {
  const initial = [
    { name: 'Nó A', x: 1.6, y: 0.3 }, // Canto Superior Direito
  ];
  initial.forEach((n, i) => addNode(n.name, n.x, n.y, i));
  state.heatmapDirty = true;
}

/* ==============================================================
   ADICIONAR UM NÓ AO SISTEMA
   ============================================================== */
export function addNode(name, x, y, colorIdx = null) {
  const id = state.nodes.length;
  const ci = colorIdx !== null ? colorIdx % NODE_COLORS.length : id % NODE_COLORS.length;
  const node = {
    id,
    name: name || `Nó ${id + 1}`,
    x: Math.max(0, Math.min(WAREHOUSE.width,  x)),
    y: Math.max(0, Math.min(WAREHOUSE.height, y)),
    color: NODE_COLORS[ci].hex,
    colorRgb: NODE_COLORS[ci].rgb,
    // Métricas simuladas (serão atualizadas)
    spl:      70 + Math.random() * 20,
    splPico:  0,
    rms:      0,
    doa:      Math.random() * 360,
    doaConf:  40 + Math.random() * 50,
    stability:88 + Math.random() * 10,
    frameErrors: 0,
    drift:    Math.random() * 2,
    dist:     5 + Math.random() * 10,
    intRel:   0,
    fftBands: new Array(32).fill(0),
    domFreq:  250 + Math.floor(Math.random() * 3) * 125,
    waveform: new Array(64).fill(0),
    online:   true,
    i2sOk:   true,
  };
  // Gerar dados FFT iniciais (espectro industrial típico)
  node.fftBands = generateIndustrialFFT();
  state.nodes.push(node);
  state.heatmapDirty = true;
  rebuildNodeList();
  return node;
}

/* ==============================================================
   SELECIONAR NÓ
   ============================================================== */
export function selectNode(id) {
  state.selectedNode = id;
  // Atualizar cards
  document.querySelectorAll('.node-card').forEach(c => {
    c.classList.toggle('selected', parseInt(c.dataset.id) === id);
  });
  updateSidebarRight();
  drawMap();
}

/* ==============================================================
   REMOVER NÓ
   ============================================================== */
export function removeNode(id) {
  if (state.nodes.length <= 1) {
    alert('Deve haver pelo menos 1 nó no sistema.');
    return;
  }
  state.nodes = state.nodes.filter(n => n.id !== id);
  // Renumerar IDs
  state.nodes.forEach((n, i) => n.id = i);
  if (state.selectedNode === id || state.selectedNode >= state.nodes.length) {
    state.selectedNode = 0;
  }
  state.heatmapDirty = true;
  rebuildNodeList();
  selectNode(state.selectedNode);
  drawMap();
}

/* ==============================================================
   MODAL — ADICIONAR NÓ FILHO
   ============================================================== */
export function openModal(refId) {
  state.refNodeForAdd = refId;
  const ref = state.nodes.find(n => n.id === refId);
  if (!ref) return;

  const modalRefName = document.getElementById('modal-ref-name');
  if (modalRefName) modalRefName.textContent = ref.name;

  const modalRefPos = document.getElementById('modal-ref-pos');
  if (modalRefPos) modalRefPos.textContent  = `(${ref.x.toFixed(1)}, ${ref.y.toFixed(1)})`;

  const modalDx = document.getElementById('modal-dx');
  if (modalDx) {
    modalDx.value = '';
    modalDx.focus();
  }

  const modalDy = document.getElementById('modal-dy');
  if (modalDy) modalDy.value = '';

  const modalName = document.getElementById('modal-name');
  if (modalName) modalName.value = `Nó ${state.nodes.length + 1}`;

  const overlay = document.getElementById('modal-overlay');
  if (overlay) overlay.classList.add('open');
}

export function closeModal() {
  const overlay = document.getElementById('modal-overlay');
  if (overlay) overlay.classList.remove('open');
}

export function confirmAddNode() {
  const ref = state.nodes.find(n => n.id === state.refNodeForAdd);
  if (!ref) { closeModal(); return; }

  const dxVal = document.getElementById('modal-dx')?.value;
  const dyVal = document.getElementById('modal-dy')?.value;
  const nameVal = document.getElementById('modal-name')?.value;

  const dx   = parseFloat(dxVal) || 0;
  const dy   = parseFloat(dyVal) || 0;
  const name = nameVal?.trim() || `Nó ${state.nodes.length + 1}`;

  const nx = ref.x + dx;
  const ny = ref.y + dy;

  // Validar limites do galpão
  if (nx < 0 || nx > WAREHOUSE.width || ny < 0 || ny > WAREHOUSE.height) {
    alert(`⚠ Posição (${nx.toFixed(1)}, ${ny.toFixed(1)}) fora dos limites do galpão (${WAREHOUSE.width}×${WAREHOUSE.height}m).`);
    return;
  }

  addNode(name, nx, ny);
  state.heatmapDirty = true;
  selectNode(state.nodes.length - 1);
  closeModal();
  drawMap();
}

// Fechar modal ao clicar fora
document.getElementById('modal-overlay')?.addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) closeModal();
});

/* ==============================================================
   CONTROLES DA STATUSBAR
   ============================================================== */
export function toggleSimulation() {
  state.simRunning = !state.simRunning;
  const btn = document.getElementById('btn-sim');
  if (btn) {
    btn.textContent = state.simRunning ? '⏸ Pausar' : '▶ Retomar';
    btn.classList.toggle('active', state.simRunning);
  }
  const badge = document.getElementById('sim-status');
  if (badge) badge.textContent = state.simRunning ? 'SIMULANDO' : 'PAUSADO';
}

export function toggleHeatmap() {
  state.showHeatmap = !state.showHeatmap;
  const btn = document.getElementById('btn-hm');
  if (btn) {
    btn.classList.toggle('active', state.showHeatmap);
    btn.textContent = state.showHeatmap ? '🌡 Heatmap ✓' : '🌡 Heatmap';
  }
  drawMap();
}

export function toggleDoA() {
  state.showDoA = !state.showDoA;
  const btn = document.getElementById('btn-doa');
  if (btn) btn.classList.toggle('active', state.showDoA);
  drawMap();
}

export function resetView() {
  state.heatmapDirty = true;
  resizeCanvas();
}

/* ==============================================================
   LOOP DE ANIMAÇÃO (rAF para canvas)
   ============================================================== */
export function animLoop() {
  drawMap();
  requestAnimationFrame(animLoop);
}

/* ==============================================================
   EXPOSIÇÃO PARA O ESCOPO GLOBAL (COMPATIBILIDADE HTML)
   ============================================================== */
window.toggleSimulation = toggleSimulation;
window.toggleHeatmap    = toggleHeatmap;
window.toggleDoA        = toggleDoA;
window.resetView        = resetView;
window.closeModal       = closeModal;
window.confirmAddNode   = confirmAddNode;
window.openModal        = openModal;

/* ==============================================================
   START
   ============================================================== */
window.addEventListener('load', () => {
  init();
  animLoop();
});