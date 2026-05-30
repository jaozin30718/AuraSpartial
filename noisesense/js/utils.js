import { WAREHOUSE } from './config.js';
import { state } from './state.js';

/* ==============================================================
   MATEMÁTICA DE TRIANGULAÇÃO (Interseção Linear)
   ============================================================== */
export function triangulateRays(x1, y1, deg1, x2, y2, deg2) {
  const r1 = deg1 * Math.PI / 180;
  const r2 = deg2 * Math.PI / 180;
  const dx1 = Math.sin(r1), dy1 = -Math.cos(r1);
  const dx2 = Math.sin(r2), dy2 = -Math.cos(r2);

  const det = dx2 * dy1 - dx1 * dy2;
  if (Math.abs(det) < 0.01) return null; // Raios quase paralelos

  const dx = x2 - x1;
  const dy = y2 - y1;

  const t1 = (dx2 * dy - dy2 * dx) / det;
  const t2 = (dx1 * dy - dy1 * dx) / det;

  // O ponto de colisão precisa estar NA FRENTE de ambas as setas (tempo > 0)
  if (t1 > 0 && t2 > 0) return { x: x1 + t1 * dx1, y: y1 + t1 * dy1 };
  return null;
}

/* ==============================================================
   CONVERTER SPL → COR HEATMAP
   Mapa: azul (baixo) → ciano → verde → amarelo → laranja → vermelho
   ============================================================== */
export function splToColor(spl, minSPL = 60, maxSPL = 95) {
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
   COR DO SPL (para indicadores de texto e barras de status)
   ============================================================== */
export function spl2color(spl) {
  if (spl >= 90) return 'var(--accent-red)';
  if (spl >= 80) return 'var(--accent-orange)';
  if (spl >= 70) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

/* ==============================================================
   TRANSFORMAÇÃO MUNDO → CANVAS
   ============================================================== */
export function getTransform() {
  if (!state.canvas) return { scale: 1, offsetX: 0, offsetY: 0, padding: 40 };
  const W = state.canvas.width;
  const H = state.canvas.height;
  const pad = state.transform.padding;
  const scaleX = (W - 2 * pad) / WAREHOUSE.width;
  const scaleY = (H - 2 * pad) / WAREHOUSE.height;
  const scale  = Math.min(scaleX, scaleY);
  const offsetX = pad + (W - 2 * pad - WAREHOUSE.width  * scale) / 2;
  const offsetY = pad + (H - 2 * pad - WAREHOUSE.height * scale) / 2;
  return { scale, offsetX, offsetY };
}

export function worldToCanvas(wx, wy) {
  const t = getTransform();
  return {
    x: t.offsetX + wx * t.scale,
    y: t.offsetY + wy * t.scale,
  };
}

export function canvasToWorld(cx, cy) {
  const t = getTransform();
  return {
    x: (cx - t.offsetX) / t.scale,
    y: (cy - t.offsetY) / t.scale,
  };
}

/* ==============================================================
   GERAR ESPECTRO FFT SIMULADO (perfil industrial)
   ============================================================== */
export function generateIndustrialFFT() {
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
