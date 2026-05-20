# Guia de Execução: Pipeline AuraSpatial (Simulação & Treinamento)

Este documento foi preparado para orientar uma pessoa fora da área de Inteligência Artificial a instalar e rodar todo o fluxo de ponta a ponta. O processo é dividido em três passos principais: Instalação do ambiente, Geração de Dados e Treinamento do Modelo. Acompanha a descrição de **todas as flags** disponíveis em cada script.

---

## Passo 1: Preparando o Ambiente 🛠️

O sistema precisa de diversas bibliotecas matemáticas e de processamento de áudio para funcionar, além da infraestrutura de IA. É **altamente recomendável** que este processo seja feito em um computador com placa de vídeo **NVIDIA (GPU)**.

### 1.1 Instalando o PyTorch e Ferramentas Básicas
A biblioteca principal da Inteligência Artificial é o PyTorch. Você precisa instalar a versão com suporte à placa de vídeo (CUDA). Abra o **Terminal (Prompt de Comando ou PowerShell)** e digite:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(Nota: O "cu118" refere-se à versão do CUDA. Se o seu computador tiver outro, talvez seja necessário ajustar)*.

### 1.2 Instalando as Dependências do Projeto
Todas as outras ferramentas que o projeto precisa (simulador de sala, manipuladores de áudio, Mamba) estão listadas no arquivo `requirements.txt`. Para instalar todas elas de uma vez, vá para a pasta onde os arquivos foram baixados e rode:
```bash
pip install -r synthetic_audio_pipeline/requirements.txt
```

---

## Passo 2: Geração de Dados (Simulação Acústica) 🎧

A IA precisa de milhares de áudios simulados para aprender. O script de geração de dados cria "salas virtuais", coloca pessoas falando nelas (fontes de áudio) e adiciona ruído. Ele embala tudo em arquivos compactos chamados "HDF5", que a IA lê muito rápido.

### Como rodar:
Estando no terminal, inicie o gerador de dados (exemplo de configuração comum):
```bash
python synthetic_audio_pipeline/main.py --total 10000 --output ./dataset_treino --workers 6
```

### Lista Completa de Flags (Configurações) de Geração
Todas as chaves que você pode colocar após `python synthetic_audio_pipeline/main.py` para controlar a simulação:

- `--total`: O número total de amostras de áudio a serem geradas (Padrão: 10000).
- `--workers`: Quantos núcleos do processador trabalharão juntos para gerar o som. Aumentar isso acelera a criação (Padrão: 6).
- `--output`: A pasta onde o dataset gerado será salvo (Padrão: `./dataset`).
- `--sources`: A pasta onde o sistema deve procurar as gravações originais de vozes de pessoas que serão inseridas nas salas virtuais (Padrão: `./audio_sources`).
- `--noise`: A pasta onde o sistema deve procurar as gravações de ruído ambiente e barulhos que serão inseridos nas salas (Padrão: `./noise_sources`).
- `--sample-rate`: A qualidade do áudio em Hz. 16000 significa que o áudio terá qualidade de gravação telefônica/rádio padrão de voz, que é o ideal para a IA (Padrão: 16000).
- `--duration`: A duração de cada áudio gerado, em segundos (Padrão: 4.0).
- `--format`: O formato na qual o banco de dados será agrupado. Pode ser `hdf5` ou `webdataset` (Padrão: `hdf5`).
- `--seed`: O número inicial do sorteador de aleatoriedade (para que você possa gerar dados perfeitamente iguais no futuro se quiser). Se `None`, ele é totalmente aleatório a cada vez (Padrão: `None`).
- `--validate`: Faz o script gerar apenas 1 (uma) única amostra de áudio e fechar, apenas para verificar se tudo está funcionando e imprimir estatísticas (Não salva nada no disco).
- `--verbose`: O terminal vai imprimir todos os mínimos detalhes matemáticos e cada passo minúsculo que está acontecendo. Útil para descobrir onde um erro aconteceu.

---

## Passo 3: Treinamento da IA 🧠

Depois que os dados foram gerados, é hora de ensinar a IA a limpar o áudio (BSS) e encontrar de onde o som vem (DoA). O treinamento acontece de forma progressiva (chamado "Curriculum Learning"): a IA primeiro estuda os dados por conta própria (Fase 1: JEPA) e depois resolve as tarefas reais (Fase 2: Multitask). Existe ainda a Fase 3, onde a IA treina apenas "adaptadores" LoRA numa sala específica.

### Como rodar:
Estando no terminal, inicie o treinamento usando as configurações sugeridas (fase JEPA, batch 64, atualizando pesos a cada 2 passos, e continuando o último treino caso tenha caído):
```bash
python aura_training/train.py --phase jepa --data-dir ./dataset_treino --batch-size 64 --accumulate-grad-batches 2 --resume-last
```

### Lista Completa de Flags (Configurações) de Treinamento
Todas as chaves que você pode colocar após `python aura_training/train.py` para controlar o treinamento:

- `--phase`: Escolhe em qual fase a inteligência artificial vai rodar. Pode ser `jepa` (aprende acústica básica sozinha), `multitask` (aprende a separar e localizar a voz) ou `lora_finetune` (aprende um local específico via adaptadores) (Padrão: `jepa`).
- `--checkpoint`: Se você quiser continuar o treinamento de onde parou, passe o caminho para um arquivo `.ckpt` (Padrão: `None`).
- `--resume-last`: Se ativado, o sistema procura automaticamente pelo último treinamento que foi interrompido (um arquivo chamado `last.ckpt` na pasta de saída) e continua do ponto exato onde parou. Muito útil para quedas de energia ou fechamentos acidentais.
- `--batch-size`: Quantos áudios a IA deve tentar resolver ao mesmo tempo na placa de vídeo. Se você ver o erro **"CUDA out of memory"**, diminua este número (Padrão: 4).
- `--accumulate-grad-batches`: Quantos passos a IA deve acumular antes de atualizar o cérebro. Ideal se você usar um batch-size muito pequeno e precisar compensar para a IA aprender direito (Padrão: 1).
- `--max-epochs`: O número máximo de "ciclos de ensino". A cada epoch, ela revisa todos os áudios que você gerou (Padrão: 100).
- `--lr`: O *Learning Rate* ou Taxa de Aprendizado. Quanto os "neurônios" da inteligência são alterados de cada vez. Valores altos causam esquecimento em massa; valores baixos causam demora para aprender (Padrão: 0.0005).
- `--devices`: Quantas placas de vídeo devem ser usadas (Padrão: 1).
- `--strategy`: Como o processamento deve ser dividido entre as placas de vídeo. Normalmente você deixa em `"auto"` (Padrão: `"auto"`).
- `--precision`: O tipo de matemática executada. `16-mixed` usa precisão curta economizando metade da memória. `32` gasta mais memória, mas nunca estoura a precisão (Padrão: `"16-mixed"`).
- `--seed`: Semente para fixar a ordem em que os arquivos são lidos, ajudando a IA ser previsível (Padrão: 42).
- `--profile`: Liga um monitor interno super complexo para saber quanto tempo exato a placa de vídeo gasta lendo e gravando coisas (Padrão: Desativado).
- `--detect-anomaly`: Faz o PyTorch conferir cada mínima alteração de número e travar com alerta se achar que um número ficou corrompido (Infinito ou "NaN"). Atrasará muito o treinamento. (Padrão: Desativado).
- `--compile`: Liga o modo "turbinado" (torna o primeiro passo do treinamento demorado enquanto o PyTorch tenta criar atalhos de computador, mas o resto do treino rodará até 20% mais rápido). (Padrão: Desativado).
- `--data-dir`: Onde os arquivos do banco de dados (que você criou no Passo 2) estão (Padrão: Pasta `dataset` padrão do projeto).
- `--output-dir`: Onde ele deve salvar o "cérebro" treinado da IA (.ckpt) (Padrão: `./outputs`).
- `--wandb-project`: Se você usar o sistema Weights & Biases (um painel em nuvem para ver os gráficos de evolução da inteligência), isso define o nome do painel lá (Padrão: `"aura-spatial"`).
- `--no-wandb`: Adicionar isso faz a IA ignorar a conexão de nuvem e parar de tentar se conectar à internet. As estatísticas de aprendizado viram textos CSV offline.
- `--no-checkpointing`: O "Gradient Checkpointing" é um artifício que troca espaço em disco por esforço de cálculo, economizando muita memória VRAM. Se você por `--no-checkpointing`, o programa gasta muito mais memória, mas roda pouca coisa mais rápido.

---

## 🆘 Dicas Rápidas para Solução de Problemas

1. **"CUDA out of memory" durante o Treinamento:**
   Significa que sua placa de vídeo ficou sem espaço. Cancele o treino (Ctrl+C) e comece de novo usando um `--batch-size` menor.

2. **O gerador de dados (Passo 2) parou e travou no meio do caminho:**
   O processo gera arquivos enormes na memória. Certifique-se de que o seu disco rígido não encheu (o disco C: ou D:). Os áudios geram gigabytes rapidamente.

3. **Demora absurda para começar o treinamento na Fase 1:**
   Se você usou a flag `--compile`, tenha paciência, o sistema pode parecer travado enquanto analisa o modelo. Se você quiser rodar sem esperas longas de compilação, apenas não inclua a flag.
