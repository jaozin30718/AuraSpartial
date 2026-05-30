import { state } from './state.js';
import { spl2color } from './utils.js';
import { selectNode, openModal } from './script.js';

/* ==============================================================
   COLORBAR (legenda lateral)
   ============================================================== */
export function drawColorbar() {
  const cb  = document.getElementById('colorbar-gradient');
  if (!cb) return;
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
export function updateTopbar() {
  const n = state.nodes.length;
  const tsNodes = document.getElementById('ts-nodes');
  if (tsNodes) tsNodes.textContent = `${n}/${n}`;

  if (n === 0) return;
  const spls   = state.nodes.map(x => x.spl);
  const maxSPL = Math.max(...spls).toFixed(1);
  const avgSPL = (spls.reduce((a,b)=>a+b,0)/n).toFixed(1);
  const avgI2S = (state.nodes.reduce((a,b)=>a+b.stability,0)/n).toFixed(0);

  const tsSplmax = document.getElementById('ts-splmax');
  if (tsSplmax) tsSplmax.textContent = maxSPL + ' dB';

  const tsSplmed = document.getElementById('ts-splmed');
  if (tsSplmed) tsSplmed.textContent = avgSPL + ' dB';

  const tsI2s = document.getElementById('ts-i2s');
  if (tsI2s) tsI2s.textContent    = avgI2S + '%';

  const tsSource = document.getElementById('ts-source');
  if (tsSource) {
    if (state.activeSources && state.activeSources.length > 0) {
      tsSource.textContent = `${state.activeSources.length} FONTE(S) ativas`;
    } else {
      tsSource.textContent = `--`;
    }
  }

  // NR-15 classificação
  const mx = parseFloat(maxSPL);
  let nr = mx > 90 ? 'ALTO' : mx > 80 ? 'MODERADO' : 'BAIXO';
  const nrel = document.getElementById('ts-nr15');
  if (nrel) {
    nrel.textContent = nr;
    nrel.style.color = mx > 90 ? 'var(--accent-red)' : mx > 80 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  }
}

/* ==============================================================
   ATUALIZAR SIDEBAR ESQUERDA (lista de nós)
   ============================================================== */
export function updateSidebarLeft() {
  const list = document.getElementById('node-list');
  if (!list) return;
  state.nodes.forEach((node, i) => {
    const card = list.children[i];
    if (!card) return;
    // SPL
    const nSpl = card.querySelector('.node-spl');
    if (nSpl) nSpl.textContent    = node.spl.toFixed(1) + ' dB';

    const nPos = card.querySelector('.node-pos');
    if (nPos) nPos.textContent    = `(${node.x.toFixed(1)}, ${node.y.toFixed(1)}) m`;

    const nDoa = card.querySelector('.node-doa');
    if (nDoa) nDoa.textContent    = `DoA: ${node.doa.toFixed(0)}°`;

    // Barra SPL
    const fill  = card.querySelector('.bar-fill');
    if (fill) {
      const pct   = Math.max(0, Math.min(100, (node.spl - 55) / 45 * 100));
      fill.style.width      = pct + '%';
      fill.style.background = spl2color(node.spl);
    }
    // Selecionado
    card.classList.toggle('selected', state.selectedNode === node.id);
  });
  const cntBadge = document.getElementById('node-count-badge');
  if (cntBadge) cntBadge.textContent = `${state.nodes.length} nós`;
}

/* ==============================================================
   RECONSTRUIR LISTA DE NÓS (quando nós mudam)
   ============================================================== */
export function rebuildNodeList() {
  const list = document.getElementById('node-list');
  if (!list) return;
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
                title="Adicionar nó filho">+</button>
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

    // Vincula o evento ao botão programaticamente para suportar escopo de módulo
    const addBtn = card.querySelector('.node-add-btn');
    if (addBtn) {
      addBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openModal(node.id);
      });
    }

    card.addEventListener('click', () => selectNode(node.id));
    list.appendChild(card);
  });
}

/* ==============================================================
   ATUALIZAR SIDEBAR DIREITA (métricas do nó selecionado)
   ============================================================== */
export function updateSidebarRight() {
  const node = state.nodes.find(n => n.id === state.selectedNode);
  if (!node) return;

  // SPL grande
  const splEl = document.getElementById('spl-big');
  if (splEl) {
    splEl.textContent = node.spl.toFixed(1);
    splEl.style.color = spl2color(node.spl);
  }

  const mSplpico = document.getElementById('m-splpico');
  if (mSplpico) mSplpico.innerHTML  = node.splPico.toFixed(1) + '<span class="metric-unit">dB</span>';

  const mRms = document.getElementById('m-rms');
  if (mRms) mRms.innerHTML      = node.rms.toExponential(3) + '<span class="metric-unit">amplitude</span>';

  const mIntrel = document.getElementById('m-intrel');
  if (mIntrel) mIntrel.innerHTML   = node.intRel.toFixed(1) + '<span class="metric-unit">%</span>';

  const mIntrelBar = document.getElementById('m-intrel-bar');
  if (mIntrelBar) {
    mIntrelBar.style.width = node.intRel + '%';
    mIntrelBar.style.background = spl2color(node.spl);
  }

  // I2S
  const mStab = document.getElementById('m-stab');
  if (mStab) mStab.textContent    = node.stability.toFixed(1) + '%';

  const mStabBar = document.getElementById('m-stab-bar');
  if (mStabBar) mStabBar.style.width = node.stability + '%';

  const mErrors = document.getElementById('m-errors');
  if (mErrors) mErrors.textContent  = node.frameErrors;

  const mDrift = document.getElementById('m-drift');
  if (mDrift) mDrift.textContent   = node.drift.toFixed(2) + ' ppm';

  // DoA
  const mDoa = document.getElementById('m-doa');
  if (mDoa) mDoa.textContent      = node.doa.toFixed(1) + '°';

  const mDoaConf = document.getElementById('m-doa-conf');
  if (mDoaConf) mDoaConf.textContent = node.doaConf.toFixed(0) + '%';

  const mDoaBar = document.getElementById('m-doa-bar');
  if (mDoaBar) mDoaBar.style.width  = node.doaConf + '%';

  drawCompass(node.doa, node.color);

  // Distância
  const mDist = document.getElementById('m-dist');
  if (mDist) mDist.textContent = node.dist.toFixed(1) + ' m';

  // FFT
  drawFFT(node.fftBands, node.color);

  // Waveform
  drawWaveform(node.waveform, node.color);

  // Freq dominante
  const domBand = node.fftBands.indexOf(Math.max(...node.fftBands));
  const domFreq = Math.round(20 * Math.pow(1000, domBand / 32));
  const mFdom = document.getElementById('m-fdom');
  if (mFdom) mFdom.textContent = domFreq + ' Hz';

  // Atualizar info do nó
  const niName = document.getElementById('ni-name');
  if (niName) niName.textContent = node.name;

  const niPos = document.getElementById('ni-pos');
  if (niPos) niPos.textContent  = `X: ${node.x.toFixed(1)}m | Y: ${node.y.toFixed(1)}m`;

  const niDot = document.getElementById('ni-dot');
  if (niDot) {
    niDot.style.background   = node.color;
    niDot.style.boxShadow    = `0 0 6px ${node.color}`;
  }

  const metricsTitle = document.getElementById('metrics-title');
  if (metricsTitle) metricsTitle.textContent  = `— ${node.name}`;
}

/* ==============================================================
   DESENHAR COMPASSO DoA
   ============================================================== */
export function drawCompass(angleDeg, color) {
  if (!state.doaCanvas || !state.doaCtx) return;
  const c   = state.doaCanvas;
  const ctx2= state.doaCtx;
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
export function drawFFT(bands, color) {
  if (!state.fftCanvas || !state.fftCtx) return;
  const c    = state.fftCanvas;
  const ctx2 = state.fftCtx;
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
export function drawWaveform(wave, color) {
  if (!state.waveCanvas || !state.waveCtx) return;
  const c    = state.waveCanvas;
  const ctx2 = state.waveCtx;
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
   ATUALIZAR RELÓGIO
   ============================================================== */
export function updateClock() {
  const now = new Date();
  const clk = document.getElementById('clock');
  if (clk) clk.textContent = now.toLocaleTimeString('pt-BR');

  const clkDt = document.getElementById('clock-date');
  if (clkDt) {
    clkDt.textContent = now.toLocaleDateString('pt-BR', {
      weekday: 'short', day: '2-digit', month: 'short', year: 'numeric'
    });
  }
}
