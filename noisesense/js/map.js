import { state } from './state.js';
import { WAREHOUSE, HEAT_DECAY, HEAT_MAX, HEAT_GRID_RES, HEAT_COLS, HEAT_ROWS } from './config.js';
import { worldToCanvas, canvasToWorld, getTransform, splToColor } from './utils.js';
import { updateSidebarLeft } from './ui.js';
import { selectNode, openModal, removeNode } from './script.js';

let draggingNode = null;

/* ==============================================================
   RESIZE DO CANVAS
   ============================================================== */
export function resizeCanvas() {
  if (!state.canvas) return;
  const center = document.getElementById('center');
  if (!center) return;
  state.canvas.width  = center.clientWidth;
  state.canvas.height = center.clientHeight;
  
  // Atualizar transform global
  const t = getTransform();
  state.transform.scale   = t.scale;
  state.transform.offsetX = t.offsetX;
  state.transform.offsetY = t.offsetY;
  state.heatmapDirty = true;
  drawMap();
}

/* ==============================================================
   DESENHAR O MAPA COMPLETO
   ============================================================== */
export function drawMap() {
  if (!state.ctx) return;
  const W = state.canvas.width;
  const H = state.canvas.height;
  const t = getTransform();

  // Limpar
  state.ctx.clearRect(0, 0, W, H);
  state.ctx.fillStyle = '#0d1117';
  state.ctx.fillRect(0, 0, W, H);

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
export function decayHeatGrid() {
  for (let i = 0; i < state.heatGrid.length; i++) {
    state.heatGrid[i] *= HEAT_DECAY;
    if (state.heatGrid[i] < 0.01) state.heatGrid[i] = 0;
  }
}

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
  state.ctx.save();
  state.ctx.beginPath();
  state.ctx.rect(wp.x, wp.y, canvasW, canvasH);
  state.ctx.clip();
  state.ctx.filter = 'blur(8px)';
  state.ctx.drawImage(tmp, wp.x, wp.y);
  state.ctx.filter = 'none';
  state.ctx.restore();
}

/* ---- Galpão ---- */
function drawWarehouse(t) {
  const wp = worldToCanvas(0, 0);
  const ep = worldToCanvas(WAREHOUSE.width, WAREHOUSE.height);

  // Fundo
  state.ctx.fillStyle = 'rgba(22, 27, 34, 0.6)';
  state.ctx.fillRect(wp.x, wp.y, ep.x - wp.x, ep.y - wp.y);

  // Borda
  state.ctx.strokeStyle = '#58a6ff';
  state.ctx.lineWidth   = 2.5;
  state.ctx.strokeRect(wp.x, wp.y, ep.x - wp.x, ep.y - wp.y);

  // Paredes internas
  state.ctx.strokeStyle = 'rgba(88,166,255,0.4)';
  state.ctx.lineWidth   = 1.5;
  state.ctx.setLineDash([6, 4]);
  WAREHOUSE.walls.forEach(([x1, y1, x2, y2]) => {
    const a = worldToCanvas(x1, y1);
    const b = worldToCanvas(x2, y2);
    state.ctx.beginPath();
    state.ctx.moveTo(a.x, a.y);
    state.ctx.lineTo(b.x, b.y);
    state.ctx.stroke();
  });
  state.ctx.setLineDash([]);

  // Obstáculos (máquinas)
  WAREHOUSE.obstacles.forEach(([ox, oy, ow, oh]) => {
    const op = worldToCanvas(ox, oy);
    const oe = worldToCanvas(ox + ow, oy + oh);
    state.ctx.fillStyle   = 'rgba(48, 54, 61, 0.85)';
    state.ctx.strokeStyle = 'rgba(61,68,77,0.9)';
    state.ctx.lineWidth   = 1;
    const W = oe.x - op.x;
    const H = oe.y - op.y;
    state.ctx.fillRect(op.x, op.y, W, H);
    state.ctx.strokeRect(op.x, op.y, W, H);
    // Ícone máquina
    state.ctx.fillStyle = 'rgba(88,166,255,0.15)';
    state.ctx.fillRect(op.x+2, op.y+2, W-4, H-4);
  });
}

/* ---- Conexões entre nós ---- */
function drawNodeConnections(t) {
  if (state.nodes.length < 2) return;
  state.ctx.strokeStyle = 'rgba(88,166,255,0.12)';
  state.ctx.lineWidth   = 1;
  state.ctx.setLineDash([3, 5]);
  for (let i = 0; i < state.nodes.length; i++) {
    for (let j = i + 1; j < state.nodes.length; j++) {
      const a = worldToCanvas(state.nodes[i].x, state.nodes[i].y);
      const b = worldToCanvas(state.nodes[j].x, state.nodes[j].y);
      state.ctx.beginPath();
      state.ctx.moveTo(a.x, a.y);
      state.ctx.lineTo(b.x, b.y);
      state.ctx.stroke();
    }
  }
  state.ctx.setLineDash([]);
}

/* ---- Setas DoA ---- */
function drawDoAArrows(t) {
  if (state.peakActive <= 0.01) return; // Esconde no silêncio

  state.nodes.forEach(node => {
    const p   = worldToCanvas(node.x, node.y);
    const ang = (node.doa - 90) * Math.PI / 180;
    const len = 30 + node.doaConf * 0.3;
    const ex  = p.x + Math.cos(ang) * len;
    const ey  = p.y + Math.sin(ang) * len;

    state.ctx.save();
    state.ctx.strokeStyle = node.color;
    state.ctx.lineWidth   = 2;
    state.ctx.globalAlpha = Math.min(0.8, state.peakActive); // Animação de Fade out
    state.ctx.beginPath();
    state.ctx.moveTo(p.x, p.y);
    state.ctx.lineTo(ex, ey);
    state.ctx.stroke();
    // Ponta da seta
    const headLen = 8;
    const a1 = ang + Math.PI * 0.8;
    const a2 = ang - Math.PI * 0.8;
    state.ctx.beginPath();
    state.ctx.moveTo(ex, ey);
    state.ctx.lineTo(ex + Math.cos(a1) * headLen, ey + Math.sin(a1) * headLen);
    state.ctx.lineTo(ex + Math.cos(a2) * headLen, ey + Math.sin(a2) * headLen);
    state.ctx.closePath();
    state.ctx.fillStyle = node.color;
    state.ctx.fill();
    state.ctx.restore();
  });
}

/* ---- Fonte sonora estimada ---- */
function drawSource(t) {
  if (state.peakActive <= 0.01 || state.activeSources.length === 0) return;
  
  state.ctx.save();
  state.ctx.globalAlpha = state.peakActive; // Aplica transparência em todo o indicador
  const now = Date.now() / 1000;

  state.activeSources.forEach(src => {
    const sp = worldToCanvas(src.x, src.y);
    const color = src.label === "TRIANGULADO" ? '248,81,73' : '240,136,62'; // Vermelho exato vs Laranja estimado
    
    for (let i = 1; i <= 3; i++) {
      const r = (i * 12) + (now * 20) % 12;
      state.ctx.beginPath();
      state.ctx.arc(sp.x, sp.y, r, 0, Math.PI * 2);
      state.ctx.strokeStyle = `rgba(${color},${0.35 - i * 0.1})`;
      state.ctx.lineWidth   = 1.5;
      state.ctx.stroke();
    }
    
    state.ctx.beginPath();
    state.ctx.arc(sp.x, sp.y, 6, 0, Math.PI * 2);
    state.ctx.fillStyle   = `rgb(${color})`;
    state.ctx.shadowBlur  = 12;
    state.ctx.shadowColor = `rgb(${color})`;
    state.ctx.fill();
    state.ctx.shadowBlur  = 0;

    state.ctx.font      = 'bold 9px Segoe UI';
    state.ctx.fillStyle = `rgb(${color})`;
    state.ctx.textAlign = 'center';
    state.ctx.fillText(src.label, sp.x, sp.y + 16);
  });
  
  state.ctx.restore();
}

/* ---- Nó sensor ---- */
function drawNode(node, t) {
  const p   = worldToCanvas(node.x, node.y);
  const sel = state.selectedNode === node.id;
  const r   = sel ? 20 : 16;

  // Halo de seleção
  if (sel) {
    state.ctx.beginPath();
    state.ctx.arc(p.x, p.y, r + 8, 0, Math.PI * 2);
    state.ctx.strokeStyle = node.color;
    state.ctx.lineWidth   = 2;
    state.ctx.globalAlpha = 0.4;
    state.ctx.stroke();
    state.ctx.globalAlpha = 1;
  }

  // Círculo principal com intensidade
  const gradient = state.ctx.createRadialGradient(p.x, p.y, 2, p.x, p.y, r);
  gradient.addColorStop(0, node.color);
  gradient.addColorStop(1, node.color + '44');
  state.ctx.beginPath();
  state.ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  state.ctx.fillStyle   = gradient;
  state.ctx.shadowBlur  = sel ? 20 : 10;
  state.ctx.shadowColor = node.color;
  state.ctx.fill();
  state.ctx.shadowBlur  = 0;

  // Borda
  state.ctx.strokeStyle = sel ? '#fff' : node.color;
  state.ctx.lineWidth   = sel ? 2.5 : 1.5;
  state.ctx.stroke();

  // Microfones (3 pontos em triângulo)
  const micAngles = [270, 30, 150];
  const micR = r * 0.55;
  micAngles.forEach((deg, i) => {
    const rad = deg * Math.PI / 180;
    const mx  = p.x + Math.cos(rad) * micR;
    const my  = p.y + Math.sin(rad) * micR;
    state.ctx.beginPath();
    state.ctx.arc(mx, my, 3, 0, Math.PI * 2);
    state.ctx.fillStyle = ['#58a6ff','#3fb950','#f0883e'][i];
    state.ctx.fill();
  });

  // Rótulo
  state.ctx.font      = `bold ${sel?10:9}px Segoe UI`;
  state.ctx.fillStyle = '#fff';
  state.ctx.textAlign = 'center';
  state.ctx.fillText(node.name, p.x, p.y - r - 8);
  state.ctx.font      = '9px Segoe UI';
  state.ctx.fillStyle = node.color;
  state.ctx.fillText(node.spl.toFixed(1) + ' dB', p.x, p.y - r - 18);
}

/* ---- Escala ---- */
function drawScale(t) {
  const { scale, offsetX, offsetY } = getTransform();
  const scaleLen = 0.5 * scale; // 0.5 metros em pixels
  const x0 = offsetX + 5;
  const y0 = state.canvas.height - 28;

  state.ctx.strokeStyle = 'rgba(139,148,158,0.6)';
  state.ctx.lineWidth   = 1.5;
  state.ctx.beginPath();
  state.ctx.moveTo(x0, y0 - 5);
  state.ctx.lineTo(x0, y0);
  state.ctx.lineTo(x0 + scaleLen, y0);
  state.ctx.lineTo(x0 + scaleLen, y0 - 5);
  state.ctx.stroke();

  state.ctx.font      = '10px Segoe UI';
  state.ctx.fillStyle = 'rgba(139,148,158,0.8)';
  state.ctx.textAlign = 'center';
  state.ctx.fillText('0.5m', x0 + scaleLen / 2, y0 + 12);
}

/* ==============================================================
   EVENTOS DO CANVAS
   ============================================================== */
export function onCanvasMouseDown(e) {
  if (e.button !== 0) return; // Permitir arrasto apenas com botão Esquerdo
  const menu = document.getElementById('canvas-ctx-menu');
  if (menu) menu.classList.remove('open');
  const rect  = state.canvas.getBoundingClientRect();
  const cx    = e.clientX - rect.left;
  const cy    = e.clientY - rect.top;

  const hit = findNodeAt(cx, cy);
  if (hit !== null) {
    selectNode(hit);
    draggingNode = hit; // Inicia o motor de arrasto livre
  }
}

export function onCanvasRightClick(e) {
  e.preventDefault();
  const rect = state.canvas.getBoundingClientRect();
  const cx   = e.clientX - rect.left;
  const cy   = e.clientY - rect.top;

  const hit = findNodeAt(cx, cy);
  const menu = document.getElementById('canvas-ctx-menu');

  if (hit !== null && menu) {
    state.refNodeForAdd = hit;
    menu.style.left = e.clientX + 'px';
    menu.style.top  = e.clientY + 'px';
    menu.classList.add('open');

    const selectBtn = document.getElementById('ctx-select');
    if (selectBtn) {
      selectBtn.onclick = () => {
        selectNode(hit);
        menu.classList.remove('open');
      };
    }

    const addBtn = document.getElementById('ctx-add');
    if (addBtn) {
      addBtn.onclick = () => {
        openModal(hit);
        menu.classList.remove('open');
      };
    }

    const removeBtn = document.getElementById('ctx-remove');
    if (removeBtn) {
      removeBtn.onclick = () => {
        removeNode(hit);
        menu.classList.remove('open');
      };
    }
  }
}

export function onCanvasMouseMove(e) {
  const rect = state.canvas.getBoundingClientRect();
  const cx   = e.clientX - rect.left;
  const cy   = e.clientY - rect.top;
  
  if (draggingNode !== null) {
    let wPos = canvasToWorld(cx, cy);
    let n = state.nodes.find(node => node.id === draggingNode);
    if (n) {
      n.x = Math.max(0, Math.min(WAREHOUSE.width, wPos.x));
      n.y = Math.max(0, Math.min(WAREHOUSE.height, wPos.y));
      state.heatmapDirty = true;
      updateSidebarLeft(); // Atualiza numeração em tempo real
      drawMap(); // Trava renderização cravada no mouse a 60fps
    }
  } else {
    const hit  = findNodeAt(cx, cy);
    state.canvas.style.cursor = hit !== null ? 'grab' : 'default';
  }
}

export function onCanvasMouseUp(e) {
  if (draggingNode !== null) {
    state.canvas.style.cursor = 'grab';
    draggingNode = null;
  }
}

export function onCanvasMouseLeave(e) {
  draggingNode = null;
}

/* Encontrar nó próximo ao ponto (px) */
export function findNodeAt(cx, cy) {
  for (const node of state.nodes) {
    const p    = worldToCanvas(node.x, node.y);
    const dx   = cx - p.x;
    const dy   = cy - p.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < 22) return node.id;
  }
  return null;
}
