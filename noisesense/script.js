/* ==============================================================
   CONFIGURAÇÕES GLOBAIS DO GALPÃO
   ============================================================== */
const WAREHOUSE = {
  width:  40,  // Correção: Galpão de 40 metros para alinhar com o DSP
  height: 25,  // Correção: Galpão de 25 metros para alinhar com o DSP
  // Obstáculos internos (máquinas/estruturas) [x, y, w, h] em metros
  obstacles: [],
  // Paredes internas
  walls: []
};

/* ==============================================================
   PALETA DE CORES DOS NÓS
   ============================================================== */
const NODE_COLORS = [
  { hex: '#58a6ff', rgb: '88,166,255' },
  { hex: '#f0883e', rgb: '240,136,62' },
  { hex: '#3fb950', rgb: '63,185,80' },
  { hex: '#bc8cff', rgb: '188,140,255' },
  { hex: '#39d0d8', rgb: '57,208,216' },
  { hex: '#f85149', rgb: '248,81,73' },
  { hex: '#d29922', rgb: '210,153,34' },
];

/* ==============================================================
   ESTADO GLOBAL DA APLICAÇÃO
   ============================================================== */
// Constantes do Heatmap Acumulativo
const HEAT_GRID_RES = 1.0;   // metros por célula
const HEAT_COLS = Math.ceil(WAREHOUSE.width  / HEAT_GRID_RES);
const HEAT_ROWS = Math.ceil(WAREHOUSE.height / HEAT_GRID_RES);
const HEAT_DECAY = 0.92;     // fator de decaimento por tick (0.92 = fade suave)
const HEAT_RADIUS = 3;       // raio de influência em células
const HEAT_MAX = 100.0;      // valor máximo de calor

let state = {
  nodes: [],           // Array de nós sensores
  selectedNode: null,  // ID do nó selecionado
  refNodeForAdd: null, // Nó de referência para adicionar filho
  simRunning: true,    // Simulação ativa
  showHeatmap: true,   // Exibir heatmap
  showDoA: true,       // Exibir setas DoA
  transientCount: 0,   // Contador de transientes detectados
  sourcePos: { x: 18, y: 14 }, // Posição estimada da fonte sonora
  tick: 0,             // Contador de ticks da simulação
  heatmapData: null,   // Cache do heatmap
  heatmapDirty: true,  // Flag para recalcular heatmap
  peakActive: 0.0,     // Alpha (transparência) da Seta e Esfera
  heatGrid: new Float32Array(HEAT_COLS * HEAT_ROWS),  // Grid acumulativo
};

/* ==============================================================
   CANVAS E CONTEXTOS
   ============================================================== */
let canvas, ctx, fftCanvas, fftCtx, waveCanvas, waveCtx, doaCanvas, doaCtx;

// Transformação canvas (posição mundo → pixels)
let transform = { scale: 1, offsetX: 0, offsetY: 0, padding: 40 };

/* ==============================================================
   INICIALIZAÇÃO DO DASHBOARD
   ============================================================== */
function init() {
  // Obter referências dos canvas
  canvas    = document.getElementById('mapCanvas');
  ctx       = canvas.getContext('2d');
  fftCanvas = document.getElementById('fft-canvas');
  fftCtx    = fftCanvas.getContext('2d');
  waveCanvas= document.getElementById('wave-canvas');
  waveCtx   = waveCanvas.getContext('2d');
  doaCanvas = document.getElementById('doa-compass');
  doaCtx    = doaCanvas.getContext('2d');

  // Criar nós iniciais
  createInitialNodes();

  // Ajustar tamanho do canvas
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Eventos do mouse no canvas
  canvas.addEventListener('click',       onCanvasClick);
  canvas.addEventListener('contextmenu', onCanvasRightClick);
  canvas.addEventListener('mousemove',   onCanvasMouseMove);

  // Fechar menus ao clicar fora
  document.addEventListener('click', () => {
    document.getElementById('canvas-ctx-menu').classList.remove('open');
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

let ws;
function initWebSocket() {
  ws = new WebSocket('ws://localhost:8765');
  
  ws.onopen = () => {
    console.log('[WS] Conectado ao servidor');
    const badge = document.getElementById('sim-status');
    if(badge) {
      badge.textContent = 'AO VIVO (WS)';
      badge.style.color = 'var(--accent-green)';
    }
  };
  
  ws.onmessage = (event) => {
    if (!state.simRunning) return;
    
    try {
      const data = JSON.parse(event.data);
      
      for (const [nid, nodeData] of Object.entries(data.nodes)) {
        if (!nodeData.raw) continue;
        
        let nodeName = `Nó ${nid}`;
        let n = state.nodes.find(node => node.name === nodeName);
        if (!n) {
           let nx = 32.0, ny = 5.0; // Posição do Nó A (Top Right)
           if (nid !== "A") {
             nx = Math.random() * 20 + 10;
             ny = Math.random() * 15 + 5;
           }
           n = addNode(nodeName, nx, ny);
        }
        
        // Mapeando dB para SPL aproximado (dB + offset)
        n.spl = Math.max(40, nodeData.raw.db_L + 120); 
        n.splPico = Math.max(40, nodeData.raw.dbPk_L + 120);
        n.rms = nodeData.raw.rms_L * 1000;
        n.doa = (nodeData.raw.angle + 360) % 360;
        n.doaConf = nodeData.raw.confidence * 100;
        n.stability = 100; 
        n.online = true;
        n.domFreq = nodeData.raw.dom_freq_L;
        
        // Fontes separadas (se disponíveis)
        n.sources = nodeData.raw.sources || [];
        
        // ---- 1. RASTREAMENTO AMBIENTE (Heatmap Constante e Forte) ----
        if (nodeData.raw.ambient_x !== undefined) {
           // Multiplicador gigante (3.0) para a Aura Térmica ser muito visível ao redor do Nó
           accumulateHeatEvent(nodeData.raw.ambient_x, nodeData.raw.ambient_y, nodeData.raw.ambient_db, 3.0);
        }
        
        // Atualizar bands FFT fake baseadas na energia
        n.fftBands = n.fftBands.map((v, i) => v * 0.8 + Math.random() * 0.2);
        
        // Waveform simple update based on RMS
        for (let i = 0; i < n.waveform.length - 1; i++)
          n.waveform[i] = n.waveform[i + 1];
        n.waveform[n.waveform.length - 1] = (Math.random() - 0.5) * 2 * (n.spl - 60) / 40;
      }
      
      // ---- 2. PICOS E TRANSIENTES (Apenas Seta e Esfera, SEM HEATMAP!) ----
      if (data.events && data.events.length > 0) {
        // Posição da fonte mais intensa para o indicador
        let best = data.events.reduce((a, b) => a.intensity > b.intensity ? a : b);
        state.sourcePos = { x: best.x, y: best.y };
        state.peakActive = 1.0; // Fade In instantâneo
        
        let nA = state.nodes.find(n => n.name === 'Nó A');
        if (nA && best.angle !== undefined) nA.doa = best.angle;
        
        state.transientCount += data.events.length;
        const dot = document.getElementById('transient-dot');
        const lbl = document.getElementById('transient-label');
        if(dot) dot.classList.add('active');
        if(lbl) lbl.textContent = `⚡ ${data.events.length} fonte(s) (WS)`;
        setTimeout(() => {
          if(dot) dot.classList.remove('active');
          if(lbl) lbl.textContent = 'Sem Transiente';
        }, 600);
        const cnt = document.getElementById('transient-count');
        if(cnt) cnt.textContent = `Eventos: ${state.transientCount}`;
      }
      
      state.heatmapDirty = true;
      
    } catch (e) {
      console.error('[WS] Erro ao parsear', e);
    }
  };
  
  ws.onclose = () => {
    console.log('[WS] Desconectado. Tentando reconectar...');
    const badge = document.getElementById('sim-status');
    if(badge) {
      badge.textContent = 'OFFLINE (WS)';
      badge.style.color = 'var(--accent-red)';
    }
    setTimeout(initWebSocket, 3000);
  };
}

/* ==============================================================
   CRIAR NÓS INICIAIS DE DEMONSTRAÇÃO
   ============================================================== */
function createInitialNodes() {
  const initial = [
    { name: 'Nó A', x: 32.0, y: 5.0 }, // Canto Superior Direito (onde você circulou)
  ];
  initial.forEach((n, i) => addNode(n.name, n.x, n.y, i));
  state.heatmapDirty = true;
}

/* ==============================================================
   ADICIONAR UM NÓ AO SISTEMA
   ============================================================== */
function addNode(name, x, y, colorIdx = null) {
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
   GERAR ESPECTRO FFT SIMULADO (perfil industrial)
   ============================================================== */
function generateIndustrialFFT() {
  const bands = new Array(32).fill(0);
  // Pico em baixa frequência (ruído de motor)
  for (let i = 0; i < 32; i++) {
    const f = i / 32; // frequência normalizada
    // Componente de fundo
    let v = 0.2 + 0.15 * Math.random();
    // Pico em ~250Hz (i≈4)
    v += 0.5 * Math.exp(-Math.pow((i - 4) * 2, 2));
    // Pico em ~1kHz (i≈12)
    v += 0.3 * Math.exp(-Math.pow((i - 12) * 2, 2));
    // Harmônicos
    v += 0.2 * Math.exp(-Math.pow((i - 8)  * 3, 2));
    v += 0.15 * Math.exp(-Math.pow((i - 20) * 3, 2));
    bands[i] = Math.min(1, v + Math.random() * 0.05);
  }
  return bands;
}

/* ==============================================================
   LOOP DE SIMULAÇÃO
   ============================================================== */
function simulationTick() {
  // A simulação foi substituída pelos dados reais do WebSocket.
  // Esta função não faz mais nada.
}

/* ==============================================================
   CALCULAR POSIÇÃO DA FONTE (média ponderada por SPL)
   ============================================================== */
function estimateSource() {
  if (state.nodes.length < 2) return state.sourcePos;
  let wx = 0, wy = 0, w = 0;
  state.nodes.forEach(n => {
    const weight = Math.pow(10, n.spl / 20);
    wx += n.x * weight;
    wy += n.y * weight;
    w  += weight;
  });
  return w > 0 ? { x: wx/w, y: wy/w } : state.sourcePos;
}

/* ==============================================================
   ACUMULAR EVENTO DE CALOR NO GRID
   Agora suporta "peso" para diferenciar ambiente de impactos
   ============================================================== */
function accumulateHeatEvent(wx, wy, intensity_db, weightMultiplier = 1.0) {
  // Converter mundo → grid
  const col = Math.floor(wx / HEAT_GRID_RES);
  const row = Math.floor(wy / HEAT_GRID_RES);
  
  // CORREÇÃO: O DSP reporta ruído de fundo normal entre -85dB e -60dB. 
  // Mapeamos a rampa para começar a gerar calor a partir de -85dB.
  const heat = Math.max(0, Math.min(1.0, (intensity_db + 85) / 60)) * 25.0 * weightMultiplier;
  
  // Pintar com kernel gaussiano
  for (let dr = -HEAT_RADIUS; dr <= HEAT_RADIUS; dr++) {
    for (let dc = -HEAT_RADIUS; dc <= HEAT_RADIUS; dc++) {
      const r = row + dr;
      const c = col + dc;
      if (r < 0 || r >= HEAT_ROWS || c < 0 || c >= HEAT_COLS) continue;
      const dist2 = dr * dr + dc * dc;
      const w = Math.exp(-dist2 / (HEAT_RADIUS * 0.8));
      const idx = r * HEAT_COLS + c;
      state.heatGrid[idx] = Math.min(HEAT_MAX, state.heatGrid[idx] + heat * w);
    }
  }
}

/* ==============================================================
   DECAIMENTO DO HEATMAP (chamado a cada tick)
   ============================================================== */
function decayHeatGrid() {
  for (let i = 0; i < state.heatGrid.length; i++) {
    state.heatGrid[i] *= HEAT_DECAY;
    if (state.heatGrid[i] < 0.01) state.heatGrid[i] = 0;
  }
}

/* ==============================================================
   CONVERTER SPL → COR HEATMAP
   Mapa: azul (baixo) → ciano → verde → amarelo → laranja → vermelho
   ============================================================== */
function splToColor(spl, minSPL = 60, maxSPL = 95) {
  const t = Math.max(0, Math.min(1, (spl - minSPL) / (maxSPL - minSPL)));
  // Gradiente 5 cores
  const stops = [
    [0.00, [75,  0, 130]],   // índigo (muito baixo)
    [0.20, [0,   0, 255]],   // azul
    [0.40, [0, 220, 200]],   // ciano
    [0.60, [80, 200,  0]],   // verde
    [0.75, [255,220,  0]],   // amarelo
    [0.90, [255,100,  0]],   // laranja
    [1.00, [255,  0,  0]],   // vermelho
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0]-c0[0]) * f),
        Math.round(c0[1] + (c1[1]-c0[1]) * f),
        Math.round(c0[2] + (c1[2]-c0[2]) * f),
      ];
    }
  }
  return [255, 0, 0];
}

/* ==============================================================
   TRANSFORMAÇÃO MUNDO → CANVAS
   ============================================================== */
function getTransform() {
  const W = canvas.width;
  const H = canvas.height;
  const pad = transform.padding;
  const scaleX = (W - 2 * pad) / WAREHOUSE.width;
  const scaleY = (H - 2 * pad) / WAREHOUSE.height;
  const scale  = Math.min(scaleX, scaleY);
  const offsetX = pad + (W - 2 * pad - WAREHOUSE.width  * scale) / 2;
  const offsetY = pad + (H - 2 * pad - WAREHOUSE.height * scale) / 2;
  return { scale, offsetX, offsetY };
}

function worldToCanvas(wx, wy) {
  const t = getTransform();
  return {
    x: t.offsetX + wx * t.scale,
    y: t.offsetY + wy * t.scale,
  };
}

function canvasToWorld(cx, cy) {
  const t = getTransform();
  return {
    x: (cx - t.offsetX) / t.scale,
    y: (cy - t.offsetY) / t.scale,
  };
}

/* ==============================================================
   RESIZE DO CANVAS
   ============================================================== */
function resizeCanvas() {
  const center = document.getElementById('center');
  canvas.width  = center.clientWidth;
  canvas.height = center.clientHeight;
  // Atualizar transform global
  const t = getTransform();
  transform.scale   = t.scale;
  transform.offsetX = t.offsetX;
  transform.offsetY = t.offsetY;
  state.heatmapDirty = true;
  drawMap();
}

/* ==============================================================
   DESENHAR O MAPA COMPLETO
   ============================================================== */
function drawMap() {
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const t = getTransform();

  // Limpar
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // ---- HEATMAP ----
  if (state.showHeatmap && state.nodes.length > 0) {
    drawHeatmap(t);
  }

  // ---- GALPÃO (bordas) ----
  drawWarehouse(t);

  // ---- LINHAS DE CONEXÃO ----
  drawNodeConnections(t);

  // ---- SETAS DoA ----
  if (state.showDoA) drawDoAArrows(t);

  // ---- FONTE SONORA ESTIMADA ----
  drawSource(t);

  // ---- NÓS ----
  state.nodes.forEach(n => drawNode(n, t));

  // ---- ESCALA ----
  drawScale(t);
}

/* ---- Heatmap Acumulativo ---- */
function drawHeatmap(t) {
  // Decaimento temporal a cada frame
  decayHeatGrid();

  // Encontrar máximo para normalização
  let maxHeat = 0;
  for (let i = 0; i < state.heatGrid.length; i++) {
    if (state.heatGrid[i] > maxHeat) maxHeat = state.heatGrid[i];
  }
  if (maxHeat < 0.1) return; // Nada a desenhar

  // Coordenadas do galpão no canvas
  const wp = worldToCanvas(0, 0);
  const ep = worldToCanvas(WAREHOUSE.width, WAREHOUSE.height);
  const canvasW = ep.x - wp.x;
  const canvasH = ep.y - wp.y;

  // Tamanho de cada célula no canvas
  const cellW = canvasW / HEAT_COLS;
  const cellH = canvasH / HEAT_ROWS;

  // Criar offscreen canvas para o heatmap
  const tmp = document.createElement('canvas');
  tmp.width  = Math.ceil(canvasW);
  tmp.height = Math.ceil(canvasH);
  const tmpCtx = tmp.getContext('2d');

  for (let row = 0; row < HEAT_ROWS; row++) {
    for (let col = 0; col < HEAT_COLS; col++) {
      const val = state.heatGrid[row * HEAT_COLS + col];
      if (val < 0.05) continue;
      
      const norm = Math.min(1.0, val / Math.max(maxHeat, 1.0));
      const [r, g, b] = splToColor(norm * 35 + 60, 60, 95);
      const alpha = Math.min(0.75, norm * 0.9);
      
      tmpCtx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      tmpCtx.fillRect(
        Math.floor(col * cellW),
        Math.floor(row * cellH),
        Math.ceil(cellW) + 1,
        Math.ceil(cellH) + 1
      );
    }
  }

  // Clipar ao galpão e desenhar com blur
  ctx.save();
  ctx.beginPath();
  ctx.rect(wp.x, wp.y, canvasW, canvasH);
  ctx.clip();
  ctx.filter = 'blur(8px)';
  ctx.drawImage(tmp, wp.x, wp.y);
  ctx.filter = 'none';
  ctx.restore();
}

/* ---- Galpão ---- */
function drawWarehouse(t) {
  const wp = worldToCanvas(0, 0);
  const ep = worldToCanvas(WAREHOUSE.width, WAREHOUSE.height);

  // Fundo
  ctx.fillStyle = 'rgba(22, 27, 34, 0.6)';
  ctx.fillRect(wp.x, wp.y, ep.x - wp.x, ep.y - wp.y);

  // Borda
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth   = 2.5;
  ctx.strokeRect(wp.x, wp.y, ep.x - wp.x, ep.y - wp.y);

  // Paredes internas
  ctx.strokeStyle = 'rgba(88,166,255,0.4)';
  ctx.lineWidth   = 1.5;
  ctx.setLineDash([6, 4]);
  WAREHOUSE.walls.forEach(([x1, y1, x2, y2]) => {
    const a = worldToCanvas(x1, y1);
    const b = worldToCanvas(x2, y2);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Obstáculos (máquinas)
  WAREHOUSE.obstacles.forEach(([ox, oy, ow, oh]) => {
    const op = worldToCanvas(ox, oy);
    const oe = worldToCanvas(ox + ow, oy + oh);
    ctx.fillStyle   = 'rgba(48, 54, 61, 0.85)';
    ctx.strokeStyle = 'rgba(61,68,77,0.9)';
    ctx.lineWidth   = 1;
    const W = oe.x - op.x;
    const H = oe.y - op.y;
    ctx.fillRect(op.x, op.y, W, H);
    ctx.strokeRect(op.x, op.y, W, H);
    // Ícone máquina
    ctx.fillStyle = 'rgba(88,166,255,0.15)';
    ctx.fillRect(op.x+2, op.y+2, W-4, H-4);
  });
}

/* ---- Conexões entre nós ---- */
function drawNodeConnections(t) {
  if (state.nodes.length < 2) return;
  ctx.strokeStyle = 'rgba(88,166,255,0.12)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([3, 5]);
  for (let i = 0; i < state.nodes.length; i++) {
    for (let j = i + 1; j < state.nodes.length; j++) {
      const a = worldToCanvas(state.nodes[i].x, state.nodes[i].y);
      const b = worldToCanvas(state.nodes[j].x, state.nodes[j].y);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);
}

/* ---- Setas DoA ---- */
function drawDoAArrows(t) {
  if (state.peakActive <= 0.01) return; // Esconde no silêncio

  state.nodes.forEach(node => {
    const p   = worldToCanvas(node.x, node.y);
    // CORREÇÃO: O Canvas gira 0 graus para a direita. 
    // Subtrair 90 alinha o grau 0º com o NORTE (Cima absoluto).
    const ang = (node.doa - 90) * Math.PI / 180;
    const len = 30 + node.doaConf * 0.3;
    const ex  = p.x + Math.cos(ang) * len;
    const ey  = p.y + Math.sin(ang) * len;

    ctx.save();
    ctx.strokeStyle = node.color;
    ctx.lineWidth   = 2;
    ctx.globalAlpha = Math.min(0.8, state.peakActive); // Animação de Fade out
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    // Ponta da seta
    const headLen = 8;
    const a1 = ang + Math.PI * 0.8;
    const a2 = ang - Math.PI * 0.8;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex + Math.cos(a1) * headLen, ey + Math.sin(a1) * headLen);
    ctx.lineTo(ex + Math.cos(a2) * headLen, ey + Math.sin(a2) * headLen);
    ctx.closePath();
    ctx.fillStyle = node.color;
    ctx.fill();
    ctx.restore();
  });
}

/* ---- Fonte sonora estimada ---- */
function drawSource(t) {
  if (state.peakActive <= 0.01) return; // Esconde no silêncio

  ctx.save();
  ctx.globalAlpha = state.peakActive; // Aplica transparência em todo o indicador
  const sp = worldToCanvas(state.sourcePos.x, state.sourcePos.y);
  const now = Date.now() / 1000;

  // Ondas de propagação
  for (let i = 1; i <= 4; i++) {
    const r = (i * 18) + (now * 20) % 18;
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(240,136,62,${0.35 - i * 0.07})`;
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  }

  // Marcador central
  ctx.beginPath();
  ctx.arc(sp.x, sp.y, 8, 0, Math.PI * 2);
  ctx.fillStyle   = '#f0883e';
  ctx.shadowBlur  = 12;
  ctx.shadowColor = '#f0883e';
  ctx.fill();
  ctx.shadowBlur  = 0;

  // Rótulo
  ctx.font      = 'bold 9px Segoe UI';
  ctx.fillStyle = '#f0883e';
  ctx.textAlign = 'center';
  ctx.fillText('FONTE', sp.x, sp.y + 20);
  ctx.restore();
}

/* ---- Nó sensor ---- */
function drawNode(node, t) {
  const p   = worldToCanvas(node.x, node.y);
  const sel = state.selectedNode === node.id;
  const r   = sel ? 20 : 16;

  // Halo de seleção
  if (sel) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + 8, 0, Math.PI * 2);
    ctx.strokeStyle = node.color;
    ctx.lineWidth   = 2;
    ctx.globalAlpha = 0.4;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Círculo principal com intensidade
  const gradient = ctx.createRadialGradient(p.x, p.y, 2, p.x, p.y, r);
  gradient.addColorStop(0, node.color);
  gradient.addColorStop(1, node.color + '44');
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fillStyle   = gradient;
  ctx.shadowBlur  = sel ? 20 : 10;
  ctx.shadowColor = node.color;
  ctx.fill();
  ctx.shadowBlur  = 0;

  // Borda
  ctx.strokeStyle = sel ? '#fff' : node.color;
  ctx.lineWidth   = sel ? 2.5 : 1.5;
  ctx.stroke();

  // Microfones (3 pontos em triângulo)
  const micAngles = [270, 30, 150];
  const micR = r * 0.55;
  micAngles.forEach((deg, i) => {
    const rad = deg * Math.PI / 180;
    const mx  = p.x + Math.cos(rad) * micR;
    const my  = p.y + Math.sin(rad) * micR;
    ctx.beginPath();
    ctx.arc(mx, my, 3, 0, Math.PI * 2);
    ctx.fillStyle = ['#58a6ff','#3fb950','#f0883e'][i];
    ctx.fill();
  });

  // Rótulo
  ctx.font      = `bold ${sel?10:9}px Segoe UI`;
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  ctx.fillText(node.name, p.x, p.y - r - 8);
  ctx.font      = '9px Segoe UI';
  ctx.fillStyle = node.color;
  ctx.fillText(node.spl.toFixed(1) + ' dB', p.x, p.y - r - 18);
}

/* ---- Escala ---- */
function drawScale(t) {
  const { scale, offsetX, offsetY } = getTransform();
  const scaleLen = 10 * scale; // 10 metros em pixels
  const x0 = offsetX + 5;
  const y0 = canvas.height - 28;

  ctx.strokeStyle = 'rgba(139,148,158,0.6)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(x0, y0 - 5);
  ctx.lineTo(x0, y0);
  ctx.lineTo(x0 + scaleLen, y0);
  ctx.lineTo(x0 + scaleLen, y0 - 5);
  ctx.stroke();

  ctx.font      = '10px Segoe UI';
  ctx.fillStyle = 'rgba(139,148,158,0.8)';
  ctx.textAlign = 'center';
  ctx.fillText('10m', x0 + scaleLen / 2, y0 + 12);
}

/* ==============================================================
   COLORBAR (legenda lateral)
   ============================================================== */
function drawColorbar() {
  const cb  = document.getElementById('colorbar-gradient');
  const cbc = cb.getContext('2d');
  const h   = cb.height;

  // Gradiente vertical (top=max, bottom=min)
  const grd = cbc.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0.00, 'rgb(255,0,0)');
  grd.addColorStop(0.15, 'rgb(255,100,0)');
  grd.addColorStop(0.30, 'rgb(255,220,0)');
  grd.addColorStop(0.50, 'rgb(80,200,0)');
  grd.addColorStop(0.70, 'rgb(0,220,200)');
  grd.addColorStop(0.85, 'rgb(0,0,255)');
  grd.addColorStop(1.00, 'rgb(75,0,130)');
  cbc.fillStyle = grd;
  cbc.fillRect(0, 0, cb.width, h);
}

/* ==============================================================
   ATUALIZAR TOPBAR
   ============================================================== */
function updateTopbar() {
  const n = state.nodes.length;
  document.getElementById('ts-nodes').textContent = `${n}/${n}`;

  if (n === 0) return;
  const spls   = state.nodes.map(x => x.spl);
  const maxSPL = Math.max(...spls).toFixed(1);
  const avgSPL = (spls.reduce((a,b)=>a+b,0)/n).toFixed(1);
  const avgI2S = (state.nodes.reduce((a,b)=>a+b.stability,0)/n).toFixed(0);

  document.getElementById('ts-splmax').textContent = maxSPL + ' dB';
  document.getElementById('ts-splmed').textContent = avgSPL + ' dB';
  document.getElementById('ts-i2s').textContent    = avgI2S + '%';

  const src = state.sourcePos;
  document.getElementById('ts-source').textContent = `(${src.x.toFixed(1)}, ${src.y.toFixed(1)}) m`;

  // NR-15 classificação
  const mx = parseFloat(maxSPL);
  let nr = mx > 90 ? 'ALTO' : mx > 80 ? 'MODERADO' : 'BAIXO';
  const nrel = document.getElementById('ts-nr15');
  nrel.textContent = nr;
  nrel.style.color = mx > 90 ? 'var(--accent-red)' : mx > 80 ? 'var(--accent-yellow)' : 'var(--accent-green)';
}

/* ==============================================================
   ATUALIZAR SIDEBAR ESQUERDA (lista de nós)
   ============================================================== */
function updateSidebarLeft() {
  const list = document.getElementById('node-list');
  state.nodes.forEach((node, i) => {
    const card = list.children[i];
    if (!card) return;
    // SPL
    card.querySelector('.node-spl').textContent    = node.spl.toFixed(1) + ' dB';
    card.querySelector('.node-pos').textContent    = `(${node.x.toFixed(1)}, ${node.y.toFixed(1)}) m`;
    card.querySelector('.node-doa').textContent    = `DoA: ${node.doa.toFixed(0)}°`;
    // Barra SPL
    const fill  = card.querySelector('.bar-fill');
    const pct   = Math.max(0, Math.min(100, (node.spl - 55) / 45 * 100));
    fill.style.width      = pct + '%';
    fill.style.background = spl2color(node.spl);
    // Selecionado
    card.classList.toggle('selected', state.selectedNode === node.id);
  });
  document.getElementById('node-count-badge').textContent = `${state.nodes.length} nós`;
}

/* ==============================================================
   RECONSTRUIR LISTA DE NÓS (quando nós mudam)
   ============================================================== */
function rebuildNodeList() {
  const list = document.getElementById('node-list');
  list.innerHTML = '';
  state.nodes.forEach(node => {
    const card = document.createElement('div');
    card.className  = 'node-card';
    card.dataset.id = node.id;
    card.style.setProperty('--node-color', node.color);
    card.style.setProperty('--node-rgb',   node.colorRgb);

    card.innerHTML = `
      <div class="node-card-header">
        <div class="node-dot" style="background:${node.color};box-shadow:0 0 6px ${node.color}"></div>
        <span class="node-name">${node.name}</span>
        <span class="node-spl" style="color:${node.color}">--</span>
        <button class="node-add-btn" style="--node-color:${node.color}" 
                title="Adicionar nó filho" onclick="openModal(${node.id});event.stopPropagation()">+</button>
      </div>
      <div class="node-pos">--</div>
      <div class="node-doa">DoA: --°</div>
      <div class="node-bar-row">
        <span>I2S</span>
        <div class="bar-track">
          <div class="bar-fill" style="background:${node.color};width:0%"></div>
        </div>
        <span>92%</span>
      </div>
    `;
    card.addEventListener('click', () => selectNode(node.id));
    list.appendChild(card);
  });
}

/* ==============================================================
   ATUALIZAR SIDEBAR DIREITA (métricas do nó selecionado)
   ============================================================== */
function updateSidebarRight() {
  const node = state.nodes.find(n => n.id === state.selectedNode);
  if (!node) return;

  // SPL grande
  const splEl = document.getElementById('spl-big');
  splEl.textContent = node.spl.toFixed(1);
  splEl.style.color = spl2color(node.spl);

  document.getElementById('m-splpico').innerHTML  = node.splPico.toFixed(1) + '<span class="metric-unit">dB</span>';
  document.getElementById('m-rms').innerHTML      = node.rms.toExponential(3) + '<span class="metric-unit">amplitude</span>';
  document.getElementById('m-intrel').innerHTML   = node.intRel.toFixed(1) + '<span class="metric-unit">%</span>';
  document.getElementById('m-intrel-bar').style.width = node.intRel + '%';
  document.getElementById('m-intrel-bar').style.background = spl2color(node.spl);

  // I2S
  document.getElementById('m-stab').textContent    = node.stability.toFixed(1) + '%';
  document.getElementById('m-stab-bar').style.width = node.stability + '%';
  document.getElementById('m-errors').textContent  = node.frameErrors;
  document.getElementById('m-drift').textContent   = node.drift.toFixed(2) + ' ppm';

  // DoA
  document.getElementById('m-doa').textContent      = node.doa.toFixed(1) + '°';
  document.getElementById('m-doa-conf').textContent = node.doaConf.toFixed(0) + '%';
  document.getElementById('m-doa-bar').style.width  = node.doaConf + '%';
  drawCompass(node.doa, node.color);

  // Distância
  document.getElementById('m-dist').textContent = node.dist.toFixed(1) + ' m';

  // FFT
  drawFFT(node.fftBands, node.color);

  // Waveform
  drawWaveform(node.waveform, node.color);

  // Freq dominante
  const domBand = node.fftBands.indexOf(Math.max(...node.fftBands));
  const domFreq = Math.round(20 * Math.pow(1000, domBand / 32));
  document.getElementById('m-fdom').textContent = domFreq + ' Hz';

  // Atualizar info do nó
  document.getElementById('ni-name').textContent = node.name;
  document.getElementById('ni-pos').textContent  = `X: ${node.x.toFixed(1)}m | Y: ${node.y.toFixed(1)}m`;
  document.getElementById('ni-dot').style.background   = node.color;
  document.getElementById('ni-dot').style.boxShadow    = `0 0 6px ${node.color}`;
  document.getElementById('metrics-title').textContent  = `— ${node.name}`;
}

/* ==============================================================
   DESENHAR COMPASSO DoA
   ============================================================== */
function drawCompass(angleDeg, color) {
  const c   = doaCanvas;
  const ctx2= doaCtx;
  const cx  = c.width / 2;
  const cy  = c.height / 2;
  const r   = cx - 5;

  ctx2.clearRect(0, 0, c.width, c.height);

  // Círculo externo
  ctx2.beginPath();
  ctx2.arc(cx, cy, r, 0, Math.PI * 2);
  ctx2.strokeStyle = '#30363d';
  ctx2.lineWidth   = 1.5;
  ctx2.stroke();

  // Marcações N/S/E/W
  const dirs = [['N',0],['E',90],['S',180],['W',270]];
  ctx2.font      = '8px Segoe UI';
  ctx2.textAlign = 'center';
  ctx2.textBaseline = 'middle';
  dirs.forEach(([lbl, deg]) => {
    const rad = (deg - 90) * Math.PI / 180;
    ctx2.fillStyle = lbl === 'N' ? '#f85149' : '#484f58';
    ctx2.fillText(lbl, cx + Math.cos(rad) * (r - 8), cy + Math.sin(rad) * (r - 8));
  });

  // Agulha DoA
  // CORREÇÃO: O eixo base do Canvas já aponta para cima. 
  // Rotacionamos diretamente. O '-90' antigo fazia o Norte apontar para o Oeste.
  const rad = angleDeg * Math.PI / 180;
  ctx2.save();
  ctx2.translate(cx, cy);
  ctx2.rotate(rad);

  // Agulha principal
  const grd = ctx2.createLinearGradient(0, -r + 12, 0, 0);
  grd.addColorStop(0, color);
  grd.addColorStop(1, color + '44');
  ctx2.beginPath();
  ctx2.moveTo(0, -r + 12);
  ctx2.lineTo(-4, 5);
  ctx2.lineTo(4, 5);
  ctx2.closePath();
  ctx2.fillStyle = grd;
  ctx2.shadowBlur  = 6;
  ctx2.shadowColor = color;
  ctx2.fill();
  ctx2.shadowBlur  = 0;

  ctx2.restore();

  // Centro
  ctx2.beginPath();
  ctx2.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx2.fillStyle = color;
  ctx2.fill();
}

/* ==============================================================
   DESENHAR FFT
   ============================================================== */
function drawFFT(bands, color) {
  const c    = fftCanvas;
  const ctx2 = fftCtx;
  const W    = c.width;
  const H    = c.height;

  ctx2.clearRect(0, 0, W, H);

  // Grade
  ctx2.strokeStyle = '#21262d';
  ctx2.lineWidth   = 1;
  for (let i = 0; i < 4; i++) {
    const y = H * (i + 1) / 5;
    ctx2.beginPath();
    ctx2.moveTo(0, y); ctx2.lineTo(W, y);
    ctx2.stroke();
  }

  // Barras
  const bw   = (W - 2) / bands.length;
  const grd  = ctx2.createLinearGradient(0, H, 0, 0);
  grd.addColorStop(0, color + 'aa');
  grd.addColorStop(1, color);

  ctx2.fillStyle = grd;
  bands.forEach((v, i) => {
    const bh = v * (H - 4);
    ctx2.fillRect(2 + i * bw, H - bh, bw - 1, bh);
  });

  // Linha de topo
  ctx2.beginPath();
  ctx2.strokeStyle = color;
  ctx2.lineWidth   = 1.5;
  bands.forEach((v, i) => {
    const x = 2 + i * bw + bw / 2;
    const y = H - v * (H - 4);
    i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
  });
  ctx2.stroke();
}

/* ==============================================================
   DESENHAR WAVEFORM
   ============================================================== */
function drawWaveform(wave, color) {
  const c    = waveCanvas;
  const ctx2 = waveCtx;
  const W    = c.width;
  const H    = c.height;
  const mid  = H / 2;

  ctx2.clearRect(0, 0, W, H);

  // Linha central
  ctx2.strokeStyle = '#21262d';
  ctx2.lineWidth   = 1;
  ctx2.beginPath();
  ctx2.moveTo(0, mid); ctx2.lineTo(W, mid);
  ctx2.stroke();

  // Forma de onda
  ctx2.beginPath();
  ctx2.strokeStyle = color;
  ctx2.lineWidth   = 1.5;
  const step = W / (wave.length - 1);
  wave.forEach((v, i) => {
    const x = i * step;
    const y = mid - v * (mid - 4);
    i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
  });
  ctx2.stroke();

  // Preenchimento
  ctx2.lineTo(W, mid); ctx2.lineTo(0, mid); ctx2.closePath();
  ctx2.fillStyle = color + '22';
  ctx2.fill();
}

/* ==============================================================
   COR DO SPL (para indicadores)
   ============================================================== */
function spl2color(spl) {
  if (spl >= 90) return 'var(--accent-red)';
  if (spl >= 80) return 'var(--accent-orange)';
  if (spl >= 70) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

/* ==============================================================
   SELECIONAR NÓ
   ============================================================== */
function selectNode(id) {
  state.selectedNode = id;
  // Atualizar cards
  document.querySelectorAll('.node-card').forEach(c => {
    c.classList.toggle('selected', parseInt(c.dataset.id) === id);
  });
  updateSidebarRight();
  drawMap();
}

/* ==============================================================
   EVENTOS DO CANVAS
   ============================================================== */
function onCanvasClick(e) {
  document.getElementById('canvas-ctx-menu').classList.remove('open');
  const rect  = canvas.getBoundingClientRect();
  const cx    = e.clientX - rect.left;
  const cy    = e.clientY - rect.top;

  // Verificar se clicou em algum nó
  const hit = findNodeAt(cx, cy);
  if (hit !== null) selectNode(hit);
}

function onCanvasRightClick(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const cx   = e.clientX - rect.left;
  const cy   = e.clientY - rect.top;

  const hit = findNodeAt(cx, cy);
  const menu = document.getElementById('canvas-ctx-menu');

  if (hit !== null) {
    state.refNodeForAdd = hit;
    menu.style.left = e.clientX + 'px';
    menu.style.top  = e.clientY + 'px';
    menu.classList.add('open');

    document.getElementById('ctx-select').onclick = () => {
      selectNode(hit);
      menu.classList.remove('open');
    };
    document.getElementById('ctx-add').onclick = () => {
      openModal(hit);
      menu.classList.remove('open');
    };
    document.getElementById('ctx-remove').onclick = () => {
      removeNode(hit);
      menu.classList.remove('open');
    };
  }
}

function onCanvasMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const cx   = e.clientX - rect.left;
  const cy   = e.clientY - rect.top;
  const wPos = canvasToWorld(cx, cy);
  // Atualizar info do canvas
  const hit  = findNodeAt(cx, cy);
  canvas.style.cursor = hit !== null ? 'pointer' : 'default';
}

/* Encontrar nó próximo ao ponto (px) */
function findNodeAt(cx, cy) {
  for (const node of state.nodes) {
    const p    = worldToCanvas(node.x, node.y);
    const dx   = cx - p.x;
    const dy   = cy - p.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < 22) return node.id;
  }
  return null;
}

/* ==============================================================
   REMOVER NÓ
   ============================================================== */
function removeNode(id) {
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
function openModal(refId) {
  state.refNodeForAdd = refId;
  const ref = state.nodes.find(n => n.id === refId);
  if (!ref) return;

  document.getElementById('modal-ref-name').textContent = ref.name;
  document.getElementById('modal-ref-pos').textContent  = `(${ref.x.toFixed(1)}, ${ref.y.toFixed(1)})`;
  document.getElementById('modal-dx').value   = '';
  document.getElementById('modal-dy').value   = '';
  document.getElementById('modal-name').value = `Nó ${state.nodes.length + 1}`;
  document.getElementById('modal-overlay').classList.add('open');
  document.getElementById('modal-dx').focus();
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

function confirmAddNode() {
  const ref = state.nodes.find(n => n.id === state.refNodeForAdd);
  if (!ref) { closeModal(); return; }

  const dx   = parseFloat(document.getElementById('modal-dx').value) || 0;
  const dy   = parseFloat(document.getElementById('modal-dy').value) || 0;
  const name = document.getElementById('modal-name').value.trim() || `Nó ${state.nodes.length + 1}`;

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
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) closeModal();
});

/* ==============================================================
   CONTROLES DA STATUSBAR
   ============================================================== */
function toggleSimulation() {
  state.simRunning = !state.simRunning;
  const btn = document.getElementById('btn-sim');
  btn.textContent = state.simRunning ? '⏸ Pausar' : '▶ Retomar';
  btn.classList.toggle('active', state.simRunning);
  document.getElementById('sim-status').textContent = state.simRunning ? 'SIMULANDO' : 'PAUSADO';
}

function toggleHeatmap() {
  state.showHeatmap = !state.showHeatmap;
  const btn = document.getElementById('btn-hm');
  btn.classList.toggle('active', state.showHeatmap);
  btn.textContent = state.showHeatmap ? '🌡 Heatmap ✓' : '🌡 Heatmap';
  drawMap();
}

function toggleDoA() {
  state.showDoA = !state.showDoA;
  const btn = document.getElementById('btn-doa');
  btn.classList.toggle('active', state.showDoA);
  drawMap();
}

function resetView() {
  state.heatmapDirty = true;
  resizeCanvas();
}

/* ==============================================================
   RELÓGIO
   ============================================================== */
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString('pt-BR');
  document.getElementById('clock-date').textContent = now.toLocaleDateString('pt-BR', {
    weekday: 'short', day: '2-digit', month: 'short', year: 'numeric'
  });
}

/* ==============================================================
   LOOP DE ANIMAÇÃO (rAF para canvas)
   ============================================================== */
function animLoop() {
  drawMap();
  requestAnimationFrame(animLoop);
}

/* ==============================================================
   START
   ============================================================== */
window.addEventListener('load', () => {
  init();
  animLoop();
});