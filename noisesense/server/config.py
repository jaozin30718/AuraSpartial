import sys
import struct

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE REDE E PORTAS
# ─────────────────────────────────────────────────────────────
UDP_IP   = "0.0.0.0"       # Endereço IP para escuta UDP. "0.0.0.0" escuta em todas as interfaces de rede ativas.
UDP_PORT = 5005            # Porta de entrada para recepção dos frames binários de áudio transmitidos pelos ESP32.
PTP_PORT = 5007            # Porta do servidor de sincronização de tempo PTP (Precision Time Protocol) para microfones.
WS_PORT  = 8765            # Porta utilizada pelo servidor WebSocket para enviar o estado analisado ao painel frontend.

# broadcast WS a cada N frames recebidos do ESP32 para reduzir uso de banda
WS_THROTTLE_FRAMES = 3   
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Reduz a frequência de atualização do front-end, economizando banda e processamento do navegador.
# ─ Se diminuir: Torna a atualização visual no painel extremamente ágil/em tempo real, mas exige mais da rede e CPU.
# ─ Recomendado: 3 ou 4 frames (cerca de 30 a 40 frames por segundo no front-end).


# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE ÁUDIO E ANÁLISE DSP
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100      
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Permite capturar frequências mais agudas (ultrassom) e melhora a precisão temporal da triangulação DoA.
# ─ Se diminuir: Reduz drasticamente o consumo de memória, CPU e banda de transmissão da rede do ESP32.
# ─ Recomendado: 44100 Hz (padrão de alta fidelidade da indústria de áudio) ou 16000 Hz (suficiente para voz/ruído geral).

ANALYSIS_SIZE = 512        
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Melhora muito a resolução espectral (detalhe das frequências no FFT), mas aumenta a latência de processamento.
# ─ Se diminuir: Minimiza o atraso temporal (resposta ágil do mapa), mas reduz a precisão de DoA e o detalhamento do FFT.
# ─ Recomendado: 512 amostras (~11.6 ms a 44.1kHz), oferecendo o melhor compromisso entre precisão espectral e baixíssima latência.

HOP_SIZE      = ANALYSIS_SIZE // 2  # Tamanho do salto de processamento overlap-add (OLA) para análise de áudio (padrão 50%)

XCORR_MAX_LAG = 32         
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Permite rastrear ângulos de DoA com microfones mais espaçados física e geometricamente.
# ─ Se diminuir: Restringe o tempo de busca da correlação cruzada, economizando CPU e filtrando reflexões espaciais distantes.
# ─ Recomendado: 32 amostras (cobre com segurança a distância máxima entre microfones a 44.1kHz).

MIC_DIST_M    = 0.1        
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Aumenta a resolução e sensibilidade angular DoA para baixas frequências, mas causa aliasing espacial nos agudos.
# ─ Se diminuir: Permite rastrear agudos sem ambiguidade (aliasing), mas reduz a sensibilidade em sons graves.
# ─ Recomendado: 0.1 metros (10 centímetros - balanço ideal para arranjos MEMS compactos).

# Filtros de Suavização Exponencial (EMA - Exponential Moving Average)
EMA_FAST = 0.3             
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: O painel responde instantaneamente a variações de SPL, mas as leituras de db ficam instáveis/trêmulas.
# ─ Se diminuir: Torna a medição de ruído muito estável e robusta, mas o medidor demorará a responder a picos rápidos de som.
# ─ Recomendado: 0.3 (resposta rápida de decibéis para monitoramento dinâmico).

EMA_SLOW = 0.1             
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Acelera a convergência do rastreador de DoA, mas a seta do compasso fica instável/oscilante no silêncio.
# ─ Se diminuir: Torna a seta DoA extremamente estável e suave, mas ela demorará a girar para uma nova fonte ativa.
# ─ Recomendado: 0.1 (suavização ideal para setas de direção acústica).


# ─────────────────────────────────────────────────────────────
# PARÂMETROS DE FILTRAGEM E TRATAMENTO DE SINAIS (DSP)
# ─────────────────────────────────────────────────────────────

# 1. Filtro Passa-Altas (HPF - High Pass Filter)
# Butterworth de alta ordem para rejeitar ruídos mecânicos de baixíssima frequência (infrassom).
HPF_CUTOFF = 80.0  
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Elimina mais ruídos de graves (ex: motores lentos, compressores), mas pode cortar frequências úteis da análise.
# ─ Se diminuir: Preserva as frequências graves originais, mas expõe a análise DoA a distorções causadas por correntes de ar.
# ─ Recomendado: 80.0 Hz (padrão da indústria de áudio para filtros de corte de graves/rumble e proteção de transdutores).

HPF_ORDER  = 4     
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Aumenta a rejeição/corte de graves abaixo da frequência de cutoff (queda de dB extremamente íngreme).
# ─ Se diminuir: Reduz a rotação e distorção de fase do filtro perto da frequência de corte, mas permite passagem de rumble mecânico.
# ─ Recomendado: 4ª ordem (Butterworth balanceado com decaimento de -24dB por oitava).

# 2. Rejeição de Componente DC (DC Block Filter)
# Filtro IIR polo-zero de alta performance para remover o offset DC introduzido pelos conversores AD/mics I2S.
DC_BLOCK_POLE = 0.995  
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Proporciona um corte de frequência de 0Hz ultra-fino, preservando quase 100% da resposta de graves a partir de 5Hz.
# ─ Se diminuir: Acelera o tempo de estabilização do filtro na inicialização, mas atenua e distorce frequências graves baixas.
# ─ Recomendado: 0.995 (padrão de telecomunicações e processamento DSP profissional de áudio).

# 3. Redutor Espectral de Ruído (Spectral Noise Reducer - Wiener Filter)
# Remove ruído contínuo e estacionário (fundo) do galpão por subtração espectral baseada na estimativa de Wiener.
NR_CALIB_TIME   = 3.0    
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Cria um perfil estatístico de ruído de fundo muito mais preciso, cobrindo variações lentas de ventiladores.
# ─ Se diminuir: Permite calibrar o redutor mais rapidamente na inicialização do servidor, mas pode distorcer a atenuação inicial.
# ─ Recomendado: 3.0 segundos (tempo ideal para capturar a assinatura acústica estacionária estável do ambiente).

NR_SMOOTH       = 0.91   
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Reduz a variação de ganho entre blocos adjacentes, eliminando tonais espúrios ("ruído musical").
# ─ Se diminuir: Torna o redutor extremamente rápido a mudanças de ambiente, mas gera ruído de processamento artificial audível.
# ─ Recomendado: 0.91 (suavização espectral balanceada e natural).

NR_OVER_SUB     = 4.0    
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Elimina agressivamente o ruído de fundo estacionário profundo, mas pode atenuar partes de sinais fracos de interesse.
# ─ Se diminuir: Reduz a agressividade, preservando transientes, mas deixa mais ruído de fundo passar para o fluxo de análise.
# ─ Recomendado: 4.0 (fornece uma forte redução de ruído de motores sem corromper transientes de impacto).

NR_NOISE_FLOOR  = 0.005  
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Reduz o artefato de "tons musicais" artificiais ao aplicar um ganho mínimo fixo nas bandas rejeitadas.
# ─ Se diminuir: Maximiza a atenuação em silêncio absoluto, mas expõe flutuações e ruído de processamento digital áspero.
# ─ Recomendado: 0.005 (-46dB de piso de atenuação — padrão de processamento de voz e cancelamento de eco industrial).

# 4. Soft Noise Gate (Portão de Ruído Suave)
# Atenua suavemente o ganho global em canais silenciados para evitar flutuações e fantasmas de DoA no silêncio.
GATE_THRESHOLD_DB = -55.0  
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Silencia o canal mais facilmente com ruídos leves, mas pode cortar palavras faladas baixas ou impactos fracos.
# ─ Se diminuir: Permite que sons muito baixos entrem no sistema, mas pode falhar em silenciar o canal quando houver estática.
# ─ Recomendado: -55.0 dBFS (limiar ideal para microfones MEMS com ruído próprio de ~ -74dBFS).

GATE_ATTACK_MS    = 5.0    
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Suaviza a abertura do gate, mas corta transientes rápidos (ex: cliques, batidas, marteladas no galpão).
# ─ Se diminuir: Abre o portão instantaneamente no início de um som, preservando transientes, mas pode gerar cliques rápidos.
# ─ Recomendado: 5.0 ms (padrão para gates rápidos de percussão e impactos industriais).

GATE_RELEASE_MS   = 80.0   
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Mantém o canal aberto por mais tempo após o fim do som, evitando cortes abruptos em reverberações de salas.
# ─ Se diminuir: Fecha o portão de ruído rapidamente após o término do impacto, mas pode criar um efeito de corte truncado.
# ─ Recomendado: 80.0 ms (tempo de decaimento natural para o ouvido humano sem engolir caudas de eco).

GATE_RATIO        = 0.05   
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Deixa passar mais som residual quando o gate está fechado, mitigando transições bruscas de áudio.
# ─ Se diminuir: Garante silenciamento absoluto quando não há atividade acústica útil (reduz ruído a zero).
# ─ Recomendado: 0.05 (atenuação de 20 vezes do nível de sinal — corte suave de -26dB).

# 5. Localizador e Separador Angular de Fontes (Narrowband DoA)
# Parâmetros para detecção de fontes acústicas discretas ativas e mapeamento espectro-angular.
SRC_MIN_DB       = -60.0  
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Exige maior energia acústica em uma frequência para classificá-la como fonte, eliminando ruídos leves.
# ─ Se diminuir: Permite detectar e separar fontes de som extremamente sutis, mas aumenta a ocorrência de falsos alarmes.
# ─ Recomendado: -60.0 dB (excelente compromisso para ambientes fabris de nível médio).

SRC_MAX_COUNT    = 3      # Número máximo de fontes distintas localizadas simultaneamente

SRC_PEAK_MIN_SEP = 15.0   
# 🎛️ EFEITOS DE ALTERAÇÃO:
# ─ Se aumentar: Impede que picos vizinhos muito próximos sejam contados como fontes separadas, fundindo-os em um só.
# ─ Se diminuir: Permite separar duas fontes com direções extremamente próximas (ex: duas máquinas lado a lado).
# ─ Recomendado: 15.0 graus (resolução espacial limite recomendada para arranjos acústicos de 3 microfones compactos).


# ─────────────────────────────────────────────────────────────
# GEOMETRIA DO GALPÃO E ANCORAGEM
# ─────────────────────────────────────────────────────────────
POS_NO_A  = (1.6, 0.3)     # Posição física XY real do Nó A na mesa de escala real 2.0m x 1.5m.
ROOM_BBOX = (-0.5, -0.5, 2.5, 2.0)  # Limites da caixa delimitadora XY do galpão para fins de triangulação de fontes.
SYNC_WINDOW_MS = 50        # Janela máxima de tempo tolerada para agrupar pacotes UDP de múltiplos nós em um evento conjunto.

BAND_RANGES = {
    'low':  (80,   400),   # Faixa acústica de Graves (ex: zumbido elétrico de transformadores e motores de indução pesados)
    'mid':  (400,  3000),  # Faixa acústica de Médios (ex: voz humana, alarmes, engrenagens e serras industriais)
    'high': (3000, 10000), # Faixa acústica de Agudos (ex: vazamentos de ar comprimido, fricção mecânica severa e sopros)
}


# ─────────────────────────────────────────────────────────────
# PROTOCOLO DE TRANSMISSÃO UDP BINÁRIO
# ─────────────────────────────────────────────────────────────
HEADER_FMT  = '<BIQQHH'    # Formato do struct: B(1) I(4) Q(8) Q(8) H(2) H(2) = 25 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 25, (
    f"HEADER_SIZE={HEADER_SIZE} bytes, esperado 25! "
    f"Verifique HEADER_FMT='{HEADER_FMT}'"
)

MAX_PACKET_SAMPLES = 2048  # Limite máximo de amostras em um único pacote UDP (segurança contra estouro de pilha e buffers)


# ─────────────────────────────────────────────────────────────
# MONITORAÇÃO E REPRODUÇÃO LOCAL
# ─────────────────────────────────────────────────────────────
LISTEN_NODE = "A"          # Identificador do nó sensor que será escutado localmente na saída de áudio física do servidor.
if "--listen" in sys.argv:
    idx = sys.argv.index("--listen")
    if idx + 1 < len(sys.argv):
        LISTEN_NODE = sys.argv[idx + 1].upper()

try:
    import sounddevice as sd
    PLAY_AUDIO = True      # Ativa a reprodução local em tempo real caso a biblioteca sounddevice esteja instalada.
except ImportError:
    print("[AVISO] sounddevice não instalado — sem reprodução local.")
    PLAY_AUDIO = False
