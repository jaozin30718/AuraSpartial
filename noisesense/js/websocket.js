import { state } from './state.js';
import { WAREHOUSE, HEAT_GRID_RES, HEAT_ROWS, HEAT_COLS, HEAT_RADIUS, HEAT_MAX } from './config.js';
import { triangulateRays } from './utils.js';
import { addNode } from './script.js';

/* ==============================================================
   ACUMULAR EVENTO DE CALOR NO GRID
   ============================================================== */
export function accumulateHeatEvent(wx, wy, intensity_db, weightMultiplier = 1.0, rad = HEAT_RADIUS) {
  const col = Math.floor(wx / HEAT_GRID_RES);
  const row = Math.floor(wy / HEAT_GRID_RES);
  
  const heat = Math.max(0, Math.min(1.0, (intensity_db + 85) / 60)) * 25.0 * weightMultiplier;
  
  for (let dr = -rad; dr <= rad; dr++) {
    for (let dc = -rad; dc <= rad; dc++) {
      const r = row + dr;
      const c = col + dc;
      if (r < 0 || r >= HEAT_ROWS || c < 0 || c >= HEAT_COLS) continue;
      
      const dist = Math.sqrt(dr * dr + dc * dc);
      if (dist > rad) continue; // Máscara circular perfeita: corta as pontas do quadrado
      
      // Gaussiana ultra-suave (esfumaçamento orgânico)
      const w = Math.exp(-(dist * dist) / (rad * rad * 0.3));
      const idx = r * HEAT_COLS + c;
      state.heatGrid[idx] = Math.min(HEAT_MAX, state.heatGrid[idx] + heat * w);
    }
  }
}

/* ==============================================================
   INICIALIZAÇÃO DO WEBSOCKET E RECEBIMENTO DE DADOS
   ============================================================== */
export function initWebSocket() {
  state.ws = new WebSocket('ws://localhost:8765');
  
  state.ws.onopen = () => {
    console.log('[WS] Conectado ao servidor');
    const badge = document.getElementById('sim-status');
    if(badge) {
      badge.textContent = 'AO VIVO (WS)';
      badge.style.color = 'var(--accent-green)';
    }
  };
  
  state.ws.onmessage = (event) => {
    if (!state.simRunning) return;
    
    try {
      const data = JSON.parse(event.data);
      
      for (const [nid, nodeData] of Object.entries(data.nodes)) {
        if (!nodeData.raw) continue;
        
        let nodeName = `Nó ${nid}`;
        let n = state.nodes.find(node => node.name === nodeName);
        if (!n) {
            let nx = 1.6, ny = 0.3; // Posição do Nó A (Top Right)
            if (nid !== "A") {
              nx = Math.random() * 1.0 + 0.5;
              ny = Math.random() * 0.8 + 0.3;
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
        
        // ---- 1. RASTREAMENTO AMBIENTE (Heatmap Suave) ----
        if (nodeData.raw.ambient_dist !== undefined) {
           // Frontend faz o raycasting a partir da posição real e editável do Nó
           let angRad = n.doa * Math.PI / 180;
           let ax = n.x + nodeData.raw.ambient_dist * Math.sin(angRad);
           let ay = n.y - nodeData.raw.ambient_dist * Math.cos(angRad);
           accumulateHeatEvent(ax, ay, nodeData.raw.ambient_db, 0.4, 30);
        }
        
        // Atualizar bands FFT fake baseadas na energia
        n.fftBands = n.fftBands.map((v, i) => v * 0.8 + Math.random() * 0.2);
        
        // Waveform simple update based on RMS
        for (let i = 0; i < n.waveform.length - 1; i++)
          n.waveform[i] = n.waveform[i + 1];
        n.waveform[n.waveform.length - 1] = (Math.random() - 0.5) * 2 * (n.spl - 60) / 40;
      }
      
      // ---- 2. PICOS E TRANSIENTES (Seta e Esfera) ----
      if (data.events && data.events.length > 0) {
        
        // 1. Atualiza as setas de TODOS os nós que detectaram o impacto
        for (const ev of data.events) {
          let nEvent = state.nodes.find(n => n.name === `Nó ${ev.node}`);
          if (nEvent && ev.angle !== undefined) {
            nEvent.doa = ev.angle; // Cada nó aponta para o que ouviu!
          }
        }
        
        state.activeSources = [];
        state.peakActive = 1.0;
 
        // Mapeia múltiplos eventos (sons distintos) por nó
        let eventsByNode = {};
        data.events.forEach(ev => {
          if (!eventsByNode[ev.node]) eventsByNode[ev.node] = [];
          eventsByNode[ev.node].push(ev);
        });
        const nodeKeys = Object.keys(eventsByNode);
        let triangulated = false;
 
        // 2. TENTATIVA DE TRIANGULAÇÃO CRUZADA (Interseção de Raios)
        if (nodeKeys.length >= 2) {
          for (let i = 0; i < nodeKeys.length; i++) {
            for (let j = i + 1; j < nodeKeys.length; j++) {
               let n1 = state.nodes.find(n => n.name === `Nó ${nodeKeys[i]}`);
               let n2 = state.nodes.find(n => n.name === `Nó ${nodeKeys[j]}`);
               if (!n1 || !n2) continue;
 
               for (let ev1 of eventsByNode[nodeKeys[i]]) {
                 for (let ev2 of eventsByNode[nodeKeys[j]]) {
                   let pt = triangulateRays(n1.x, n1.y, ev1.angle, n2.x, n2.y, ev2.angle);
                   // Se as setas se cruzarem DENTRO da mesa
                   if (pt && pt.x >= 0 && pt.x <= WAREHOUSE.width && pt.y >= 0 && pt.y <= WAREHOUSE.height) {
                     state.activeSources.push({ x: pt.x, y: pt.y, label: "TRIANGULADO" });
                     triangulated = true;
                   }
                 }
               }
            }
          }
        }
 
        // 3. FALLBACK: ESTIMATIVA (Se houver apenas 1 nó ou raios paralelos)
        if (!triangulated) {
          for (const ev of data.events) {
            let n = state.nodes.find(nd => nd.name === `Nó ${ev.node}`);
            if (n) {
              let rad = ev.angle * Math.PI / 180;
              state.activeSources.push({
                x: Math.max(0, Math.min(WAREHOUSE.width, n.x + ev.dist * Math.sin(rad))),
                y: Math.max(0, Math.min(WAREHOUSE.height, n.y - ev.dist * Math.cos(rad))),
                label: "ESTIMADO"
              });
            }
          }
        }
        
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
  
  state.ws.onclose = () => {
    console.log('[WS] Desconectado. Tentando reconectar...');
    const badge = document.getElementById('sim-status');
    if(badge) {
      badge.textContent = 'OFFLINE (WS)';
      badge.style.color = 'var(--accent-red)';
    }
    setTimeout(initWebSocket, 3000);
  };
}
