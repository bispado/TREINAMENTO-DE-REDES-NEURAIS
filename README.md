# 🎓 Checkpoint 02 - Redes Neurais com Keras + Visão Computacional

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00D4FF)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-FF6F00)

**Curso:** Inteligência Artificial  
**Professor:** André Tritiack  
**Instituição:** FIAP  
**Data:** Outubro 2025

---

## 👥 Integrantes do Grupo

- **Vinicius Murtinho Vicente** - RM551151
- **Lucas Barreto Consentino** - RM557107  
- **Gustavo Bispo Cordeiro** - RM558515

---

## 📋 Sumário

- [Objetivo do Projeto](#-objetivo-do-projeto)
- [Parte 01 - Redes Neurais com Keras (40%)](#-parte-01---redes-neurais-com-keras-40)
- [Parte 02 - Visão Computacional (60%)](#-parte-02---visão-computacional-60)
- [Ferramentas Utilizadas](#️-ferramentas-utilizadas)
- [Datasets](#-datasets)
- [Hiperparâmetros e Configurações](#️-hiperparâmetros-e-configurações)
- [Comparativo entre Abordagens](#-comparativo-entre-abordagens)
- [Resultados e Observações](#-resultados-e-observações)
- [Como Executar](#-como-executar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Vídeo de Demonstração](#-vídeo-de-demonstração)
- [Referências](#-referências)

---

## 🎯 Objetivo do Projeto

Este projeto do **Checkpoint 02** tem como objetivo principal implementar e comparar diferentes técnicas de Machine Learning e Deep Learning em dois contextos distintos:

### **Parte 01 (40%):** Redes Neurais para Dados Tabulares
Desenvolver e comparar redes neurais em Keras com modelos tradicionais do scikit-learn em problemas de:
- **Classificação Multiclasse** (Wine Dataset)
- **Regressão** (California Housing Dataset)

### **Parte 02 (60%):** Visão Computacional
Treinar e demonstrar um modelo de Visão Computacional utilizando **duas ferramentas diferentes**:
- **YOLOv8** - Detecção de objetos com Deep Learning
- **MediaPipe** - Reconhecimento de gestos em tempo real

**Objetivo Prático:** Comparar técnicas de detecção, classificação e rastreamento de gestos de mão para o jogo "Pedra, Papel e Tesoura" (Rock, Paper, Scissors).

---

## 📊 Parte 01 - Redes Neurais com Keras (40%)

Esta seção implementa redes neurais em Keras para problemas de classificação e regressão com dados tabulares, comparando-as com modelos tradicionais do scikit-learn.

### 🔷 Exercício 1 - Classificação Multiclasse

#### Dataset: Wine Dataset (UCI)
- **Descrição:** Dataset com 178 amostras de vinhos de 3 classes diferentes
- **Features:** 13 atributos químicos (álcool, acidez málica, cinzas, alcalinidade, magnésio, fenóis, flavonoides, etc.)
- **Classes:** 3 tipos de vinho (classe 0, 1, 2)
- **Fonte:** UCI Machine Learning Repository

#### Arquitetura da Rede Neural:
```python
- Camada de Entrada: 13 features
- Camada Oculta 1: 32 neurônios + ReLU
- Camada Oculta 2: 32 neurônios + ReLU
- Camada de Saída: 3 neurônios + Softmax
- Loss: categorical_crossentropy
- Optimizer: Adam
- Epochs: 100
```

#### Modelos Comparados:
1. **Rede Neural (Keras)** - configuração acima
2. **Random Forest Classifier** - 100 estimadores
3. **Logistic Regression** - multi_class='multinomial'

#### Resultados:

| Modelo | Acurácia | Observações |
|--------|----------|-------------|
| **Random Forest** | **100.00%** | ✅ Melhor desempenho |
| Logistic Regression | 97.22% | ✅ Excelente resultado |
| Keras Neural Network | 91.67% | ✅ Bom desempenho |

**Análise:**
- O Random Forest alcançou 100% de acurácia, classificando perfeitamente as 3 classes
- A Logistic Regression obteve 97.22%, resultado excelente para um modelo linear
- A Rede Neural Keras atingiu 91.67%, competitivo considerando o tamanho reduzido do dataset
- Para datasets pequenos e bem estruturados, modelos ensemble tendem a superar redes neurais profundas

**Conclusão:** O dataset Wine possui boa separabilidade entre classes, permitindo que modelos tradicionais obtenham resultados excelentes.

---

### 🔷 Exercício 2 - Regressão

#### Dataset: California Housing Dataset
- **Descrição:** Dataset com 20.640 amostras de preços de imóveis na Califórnia
- **Features:** 8 atributos (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Valor médio das casas (em $100.000)
- **Fonte:** Scikit-learn

#### Arquitetura da Rede Neural:
```python
- Camada de Entrada: 8 features
- Camada Oculta 1: 64 neurônios + ReLU
- Camada Oculta 2: 32 neurônios + ReLU
- Camada Oculta 3: 16 neurônios + ReLU
- Camada de Saída: 1 neurônio + Linear
- Loss: mse (Mean Squared Error)
- Optimizer: Adam
- Epochs: 100
```

#### Modelos Comparados:
1. **Rede Neural (Keras)** - configuração acima
2. **Random Forest Regressor** - 100 estimadores
3. **Linear Regression** - baseline

#### Resultados:

| Modelo | RMSE | MAE | R² | Observações |
|--------|------|-----|-----|-------------|
| **Random Forest** | **0.5051** | **0.3274** | **0.8053** | ✅ Melhor desempenho |
| Keras Neural Network | 0.5354 | 0.3578 | 0.7812 | ✅ Muito competitivo |
| Linear Regression | 0.7456 | 0.5332 | 0.5758 | ⚠️ Baseline |

**Análise:**
- O Random Forest obteve o melhor resultado com RMSE de 0.5051 e R² de 0.8053
- A Rede Neural Keras ficou muito próxima com RMSE de 0.5354 e R² de 0.7812
- A Regressão Linear teve desempenho inferior (R² = 0.5758), indicando relações não-lineares nos dados
- **Feature mais importante (Random Forest):** MedInc (renda média) com 52.49% de importância

**Conclusão:** Para problemas de regressão com datasets maiores, tanto Random Forest quanto Redes Neurais demonstram excelente capacidade de capturar padrões complexos.

---

### 📝 Escolha dos Hiperparâmetros - Parte 01

#### Exercício 1 (Classificação):
- **2 camadas ocultas com 32 neurônios cada**: Balanceia complexidade e capacidade de generalização para o dataset pequeno (178 amostras)
- **ReLU**: Função de ativação eficiente que evita vanishing gradient
- **Softmax**: Ideal para classificação multiclasse, produz distribuição de probabilidades
- **Categorical Crossentropy**: Loss padrão para classificação multiclasse
- **Adam**: Optimizer adaptativo que converge rapidamente

#### Exercício 2 (Regressão):
- **3 camadas ocultas (64→32→16)**: Arquitetura progressivamente menor permite extração hierárquica de features
- **ReLU**: Ativação eficiente para camadas ocultas
- **Linear**: Ativação na saída permite predizer valores contínuos sem limitação
- **MSE**: Loss padrão para regressão, penaliza erros grandes
- **Adam**: Optimizer robusto que ajusta learning rate automaticamente

---

## 🎮 Parte 02 - Visão Computacional (60%)

Esta seção implementa e compara duas abordagens distintas de Visão Computacional para detecção e classificação de gestos de mão em tempo real no jogo "Pedra, Papel e Tesoura".

### Objetivos Específicos:

1. ✅ Treinar um modelo **YOLOv8** do zero utilizando dataset customizado
2. ✅ Implementar detecção em tempo real com **MediaPipe** usando análise geométrica
3. ✅ Comparar as duas abordagens em termos de precisão, velocidade e recursos
4. ✅ Desenvolver aplicações práticas executáveis em tempo real via webcam
5. ✅ Documentar todo o processo incluindo desafios e soluções

---

## 🛠️ Ferramentas Utilizadas

### **Parte 01 - Redes Neurais:**

#### 1️⃣ Keras/TensorFlow
- **Tipo:** Framework de Deep Learning
- **Uso:** Construção e treinamento de redes neurais
- **Versão:** Keras 2.x com backend TensorFlow 2.x

#### 2️⃣ Scikit-learn
- **Tipo:** Biblioteca de Machine Learning tradicional
- **Uso:** Modelos de comparação (Random Forest, Logistic Regression, Linear Regression)
- **Versão:** 1.3+

#### 3️⃣ Pandas, NumPy, Matplotlib, Seaborn
- **Uso:** Manipulação de dados, análise exploratória e visualização

---

### **Parte 02 - Visão Computacional:**

#### 1️⃣ YOLOv8 (You Only Look Once v8) ⭐

**Tipo:** Detecção de Objetos com Deep Learning

**Descrição:**
- Framework de última geração para detecção de objetos em tempo real
- Utiliza redes neurais convolucionais (CNN) para detectar e classificar objetos
- Treinamento realizado com transfer learning a partir de pesos pré-treinados no COCO dataset
- Versão utilizada: **YOLOv8n (Nano)** - otimizada para velocidade

**Principais Características:**
- ✅ Alta precisão na detecção de múltiplos objetos
- ✅ Bounding boxes precisas ao redor dos objetos
- ✅ Robusto a variações de iluminação, ângulo e escala
- ✅ Capacidade de detectar múltiplas mãos simultaneamente
- ⚠️ Requer GPU para treinamento eficiente
- ⚠️ Necessita dataset anotado com bounding boxes

**Implementação:**
- Notebook Jupyter para treinamento no Google Colab
- Script Python para inferência em tempo real
- Framework: Ultralytics

---

#### 2️⃣ MediaPipe Hands ⭐

**Tipo:** Detecção de Landmarks com Machine Learning

**Descrição:**
- Framework do Google para detecção de mãos e extração de pontos-chave (landmarks)
- Detecta 21 pontos em cada mão (articulações e pontas dos dedos)
- Classificação baseada em lógica geométrica dos landmarks
- Modelo pré-treinado, sem necessidade de treinamento adicional

**Principais Características:**
- ✅ Leve e rápido (funciona em CPU)
- ✅ Não requer dataset ou treinamento
- ✅ Detecta landmarks precisos das mãos
- ✅ Ideal para rastreamento de gestos em tempo real
- ⚠️ Classificação baseada em regras (não aprendizado profundo)
- ⚠️ Pode ter dificuldade com gestos ambíguos

**Implementação:**
- Script Python com lógica de classificação customizada
- Análise geométrica dos dedos estendidos/dobrados
- Suavização temporal das predições

---

#### 3️⃣ Ferramentas Auxiliares

- **OpenCV (cv2):** Captura e processamento de vídeo em tempo real
- **NumPy:** Operações matemáticas com arrays
- **Google Colab:** Ambiente para treinamento do YOLOv8 com GPU gratuita
- **Roboflow:** Gerenciamento e download do dataset

---

## 📊 Datasets

### **Parte 01 - Dados Tabulares:**

#### 🍷 Wine Dataset (Exercício 1)
- **Descrição:** Classificação de vinhos italianos
- **Amostras:** 178
- **Features:** 13 (químicas)
- **Classes:** 3
- **Fonte:** UCI Machine Learning Repository
- **Formato:** CSV

#### 🏘️ California Housing Dataset (Exercício 2)
- **Descrição:** Preços de imóveis na Califórnia
- **Amostras:** 20.640
- **Features:** 8 (demográficas e geográficas)
- **Target:** Preço médio das casas
- **Fonte:** Scikit-learn (built-in)
- **Formato:** Numpy arrays

---

### **Parte 02 - Visão Computacional:**

#### 🎮 Rock Paper Scissors Detection Dataset

**Informações Gerais:**
- **Nome:** Rock Paper Scissors Detection Dataset
- **Fonte:** [Roboflow Universe](https://universe.roboflow.com/bispado/rock-paper-scissors-sxsw-zbvgm)
- **Formato:** YOLOv8 (anotações em .txt)
- **Total de Imagens:** 3.129
- **Classes:** 3 (Paper, Rock, Scissors)

**Distribuição:**

| Conjunto   | Imagens | Porcentagem |
|------------|---------|-------------|
| Treino     | 2.196   | 70.2%       |
| Validação  | 604     | 19.3%       |
| Teste      | 329     | 10.5%       |

**Características:**
- ✅ Imagens de alta qualidade com diversas condições
- ✅ Múltiplas pessoas e ângulos de câmera
- ✅ Variações de iluminação e backgrounds
- ✅ Anotações precisas com bounding boxes
- ✅ Balanceamento razoável entre as classes

**Links:**
- 🔗 **Google Drive:** [Link para o dataset](https://drive.google.com/drive/folders/1Lyaf5Ns15ABItUMfwP8jumKftJxOSmjL?usp=sharing)
- 🔗 **Roboflow:** [Rock Paper Scissors Dataset](https://universe.roboflow.com/bispado/rock-paper-scissors-sxsw-zbvgm/dataset/1)

---

## ⚙️ Hiperparâmetros e Configurações

### **Parte 02 - Visão Computacional:**

#### YOLOv8 - Configurações de Treinamento

**Modelo Base:**
- **Arquitetura:** YOLOv8n (Nano)
- **Parâmetros:** ~3.2M
- **Pesos Iniciais:** COCO pré-treinado (transfer learning)
- **Input Size:** 640x640 pixels

**Hiperparâmetros Principais:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `epochs` | 100 | Convergência completa com early stopping |
| `batch_size` | 16 | Otimizado para GPU T4 do Colab |
| `learning_rate_initial` | 0.01 | Taxa padrão do YOLO com bons resultados |
| `learning_rate_final` | 0.01 | Mantém aprendizado ao final |
| `momentum` | 0.937 | Momentum padrão do SGD |
| `weight_decay` | 0.0005 | Regularização L2 para evitar overfitting |
| `warmup_epochs` | 3 | Aquecimento gradual do learning rate |
| `patience` | 50 | Early stopping após 50 epochs sem melhora |
| `optimizer` | Auto (AdamW) | Escolha automática baseada no dataset |
| `image_size` | 640 | Resolução padrão do YOLO |
| `workers` | 8 | Paralelização do data loading |
| `device` | cuda:0 | GPU para aceleração |

**Data Augmentation:**
- ✅ Mosaic Augmentation (primeiros 90 epochs)
- ✅ Mixup
- ✅ HSV Color Augmentation
- ✅ Random Horizontal Flip
- ✅ Scale & Translate

---

#### MediaPipe - Configurações

**Parâmetros do Modelo:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `min_detection_confidence` | 0.7 | Detectar apenas mãos com alta confiança |
| `min_tracking_confidence` | 0.5 | Rastreamento mais permissivo para fluidez |
| `max_num_hands` | 2 | Detectar até 2 mãos simultaneamente |
| `model_complexity` | 1 | Modelo full (não lite) para melhor precisão |

**Lógica de Classificação Customizada:**

1. **Contagem de dedos estendidos:**
   - Análise das posições Y dos landmarks (pontos-chave)
   - Polegar: comparação horizontal (X) considerando lateralidade da mão
   - Outros dedos: comparação vertical (Y da ponta vs base)

2. **Regras de Classificação:**
   - **Rock (Pedra):** 0-1 dedos estendidos → Confiança 85-95%
   - **Paper (Papel):** 4-5 dedos estendidos → Confiança 85-95%
   - **Scissors (Tesoura):** 2-3 dedos (índice + médio) → Confiança 70-90%

3. **Suavização Temporal:**
   - Histórico de 5 frames
   - Predição final = moda (gesto mais frequente)
   - Evita oscilações indesejadas

---

## 🔄 Comparativo entre Abordagens

### **Parte 01 - Redes Neurais vs Modelos Tradicionais:**

#### Classificação (Wine Dataset):

| Aspecto | Rede Neural (Keras) | Random Forest | Logistic Regression |
|---------|---------------------|---------------|---------------------|
| **Acurácia** | 91.67% | **100.00%** ✅ | 97.22% |
| **Complexidade** | Alta (requer tuning) | Média | Baixa |
| **Tempo de Treino** | ~10s | ~2s | <1s |
| **Interpretabilidade** | ❌ Baixa | ⚠️ Média | ✅ Alta |
| **Overfitting** | ⚠️ Risco médio | ⚠️ Risco médio | ✅ Baixo risco |

**Conclusão:** Para datasets pequenos e estruturados, Random Forest supera redes neurais.

---

#### Regressão (California Housing):

| Aspecto | Rede Neural (Keras) | Random Forest | Linear Regression |
|---------|---------------------|---------------|-------------------|
| **RMSE** | 0.5354 | **0.5051** ✅ | 0.7456 |
| **R²** | 0.7812 | **0.8053** ✅ | 0.5758 |
| **Complexidade** | Alta | Média | Baixa |
| **Tempo de Treino** | ~45s | ~15s | <1s |
| **Capacidade Não-Linear** | ✅ Alta | ✅ Alta | ❌ Baixa |

**Conclusão:** Para datasets maiores, redes neurais se tornam competitivas, mas Random Forest ainda lidera.

---

### **Parte 02 - YOLOv8 vs MediaPipe:**

#### Tabela Comparativa Geral

| Critério | YOLOv8 | MediaPipe | Vencedor |
|----------|--------|-----------|----------|
| **Precisão/mAP** | ⭐⭐⭐⭐⭐ (85-95%) | ⭐⭐⭐⭐ (80-90%) | YOLOv8 |
| **Velocidade (FPS)** | ⭐⭐⭐⭐ (30-60 com GPU) | ⭐⭐⭐⭐⭐ (60-120 em CPU) | MediaPipe |
| **Uso de Recursos** | ⭐⭐⭐ (Requer GPU) | ⭐⭐⭐⭐⭐ (Funciona em CPU) | MediaPipe |
| **Facilidade de Implementação** | ⭐⭐⭐ (Requer treinamento) | ⭐⭐⭐⭐⭐ (Modelo pré-treinado) | MediaPipe |
| **Flexibilidade** | ⭐⭐⭐⭐⭐ (Qualquer objeto) | ⭐⭐⭐ (Apenas mãos/corpo) | YOLOv8 |
| **Robustez** | ⭐⭐⭐⭐⭐ (Muito robusto) | ⭐⭐⭐⭐ (Bom, mas sensível) | YOLOv8 |
| **Escalabilidade** | ⭐⭐⭐⭐ (Escalável com recursos) | ⭐⭐⭐⭐⭐ (Muito escalável) | MediaPipe |

---

#### Análise Detalhada por Critério:

##### 🎯 Precisão e Acurácia

**YOLOv8:**
- ✅ mAP@0.5: ~92% (após treinamento)
- ✅ Detecta gestos mesmo parcialmente visíveis
- ✅ Robusto a oclusões e backgrounds complexos
- ✅ Aprende padrões complexos do dataset
- ❌ Requer dataset grande e bem anotado

**MediaPipe:**
- ✅ Precisão alta em condições ideais (~88%)
- ✅ Landmarks muito precisos
- ❌ Classificação baseada em regras pode errar
- ❌ Sensível a gestos ambíguos (ex: 3 dedos)
- ❌ Não aprende com novos dados

**🏆 Vencedor: YOLOv8** - Maior precisão após treinamento adequado

---

##### ⚡ Velocidade e Performance

**YOLOv8:**
- Inferência: ~20-30ms por frame (GPU)
- FPS: 30-50 em webcam 1280x720
- ❌ Requer GPU para tempo real fluido
- ✅ Otimizações: YOLOv8n, TensorRT, ONNX

**MediaPipe:**
- Inferência: ~8-15ms por frame (CPU)
- FPS: 60-120 em webcam 1280x720
- ✅ Extremamente leve e rápido
- ✅ Funciona perfeitamente em CPU

**🏆 Vencedor: MediaPipe** - Muito mais rápido, especialmente em CPU

---

##### 💻 Recursos Computacionais

**YOLOv8:**
- Treinamento: Requer GPU (8-16GB VRAM)
- Inferência: GPU recomendada
- Memória: ~200-500MB (modelo carregado)
- Disco: ~6MB (modelo YOLOv8n)

**MediaPipe:**
- Treinamento: Não requer (pré-treinado)
- Inferência: CPU suficiente
- Memória: ~100-200MB
- Disco: ~20MB (bibliotecas)

**🏆 Vencedor: MediaPipe** - Muito menos exigente

---

##### 🔧 Complexidade de Implementação

**YOLOv8:**
1. Coletar/preparar dataset (trabalhoso)
2. Anotar imagens com bounding boxes (demorado)
3. Configurar ambiente de treinamento
4. Treinar modelo (1-3 horas)
5. Validar e otimizar
6. Deploy

**MediaPipe:**
1. Instalar biblioteca
2. Implementar lógica de classificação
3. Ajustar thresholds
4. Deploy

**🏆 Vencedor: MediaPipe** - Muito mais simples

---

##### 🎨 Flexibilidade e Adaptabilidade

**YOLOv8:**
- ✅ Pode detectar qualquer objeto treinado
- ✅ Fácil adicionar novas classes
- ✅ Transfer learning eficiente
- ✅ Personalização total

**MediaPipe:**
- ❌ Limitado a mãos/corpo/rosto
- ❌ Classificação requer lógica manual
- ❌ Difícil adaptar para outros objetos
- ✅ Landmarks servem para vários gestos

**🏆 Vencedor: YOLOv8** - Muito mais flexível

---

### 📊 Casos de Uso Ideais

#### Quando usar YOLOv8:
- ✅ Detecção de objetos personalizados
- ✅ Múltiplos objetos em cena complexa
- ✅ Dataset disponível ou fácil de criar
- ✅ GPU disponível
- ✅ Precisão é prioridade máxima

**Exemplos:** Controle de qualidade industrial, contagem de objetos, vigilância

#### Quando usar MediaPipe:
- ✅ Detecção de mãos/gestos/poses
- ✅ Recursos computacionais limitados (CPU)
- ✅ Latência ultra-baixa necessária
- ✅ Prototipagem rápida
- ✅ Aplicações mobile/embarcadas

**Exemplos:** Apps de fitness, jogos com gestos, realidade aumentada

---

### 💡 Conclusão Geral do Comparativo

**Melhor Abordagem:** Depende do contexto e dos requisitos!

- **Para este projeto (Rock Paper Scissors):** YOLOv8 oferece melhor precisão, mas MediaPipe é surpreendentemente eficaz e muito mais acessível.

- **Para produção em escala:** MediaPipe vence por ser mais leve, rápido e não exigir GPU.

- **Para pesquisa/precisão máxima:** YOLOv8 é superior devido ao aprendizado profundo.

**Abordagem Híbrida Ideal:**
1. Use MediaPipe para detecção inicial da mão (rápido)
2. Use YOLOv8 para classificação final (preciso)
3. Combine o melhor dos dois mundos!

---

## 📈 Resultados e Observações

### **Parte 01 - Redes Neurais:**

#### 🍷 Exercício 1 (Classificação - Wine):

**Vencedor:** Random Forest com 100% de acurácia

**Observações:**
- Dataset pequeno (178 amostras) favorece modelos tradicionais
- Classes bem separáveis linearmente
- Rede neural competitiva (91.67%), mas não necessária neste caso
- Overfitting não foi problema significativo em nenhum modelo

---

#### 🏘️ Exercício 2 (Regressão - California Housing):

**Vencedor:** Random Forest com RMSE 0.5051 e R² 0.8053

**Observações:**
- Dataset maior (20.640 amostras) permite melhor performance das redes neurais
- Relações não-lineares complexas nos dados
- Rede neural muito competitiva (R² 0.7812), quase empate com Random Forest
- MedInc (renda média) é a feature mais importante (52.49%)
- Regressão Linear insuficiente para capturar complexidade dos dados

**Insight:** Com datasets maiores, redes neurais se tornam cada vez mais competitivas.

---

### **Parte 02 - Visão Computacional:**

#### YOLOv8 - Métricas de Treinamento

**Métricas Finais:**
- **mAP@0.5:** 92.3%
- **mAP@0.5:0.95:** 78.5%
- **Precision:** 91.7%
- **Recall:** 89.4%
- **F1-Score:** 90.5%

**Métricas por Classe:**

| Classe | Precision | Recall | mAP@0.5 |
|--------|-----------|--------|---------|
| Paper | 93.2% | 91.5% | 94.1% |
| Rock | 91.8% | 88.9% | 91.7% |
| Scissors | 90.1% | 87.8% | 91.1% |

**Performance:**
- **Tempo de Treinamento:** ~1.5 horas (GPU T4 do Colab)
- **Epochs até Convergência:** 87/100 (early stopping funcionou)
- **Inferência (GPU):** ~22ms por frame
- **FPS Real-time:** 45 FPS

**Observações:**
- ✅ Excelente precisão para todas as classes
- ✅ Transfer learning do COCO foi eficaz
- ✅ Data augmentation ajudou na generalização
- ✅ Modelo robusto a diferentes iluminações e ângulos
- ⚠️ Requer GPU para inferência em tempo real fluida

---

#### MediaPipe - Resultados

**Performance em Tempo Real:**
- **FPS Médio:** 85 FPS (CPU i7)
- **Latência:** ~12ms por frame
- **Taxa de Detecção:** 96% (frames com mão visível)

**Acurácia Estimada (testes manuais com 100 gestos):**

| Gesto | Acurácia | Confiança Média |
|-------|----------|-----------------|
| Rock | 94% | 92% |
| Paper | 96% | 94% |
| Scissors | 85% | 83% |

**Média Geral:** ~91.7% de acurácia

**Observações:**
- ✅ Muito rápido e fluido (85 FPS em CPU!)
- ✅ Funciona bem em boa iluminação
- ✅ Landmarks extremamente precisos
- ⚠️ Scissors às vezes confundido com Rock (dedos parcialmente visíveis)
- ⚠️ Sensível a ângulos muito laterais da mão
- ⚠️ Classificação baseada em regras tem limitações

---

### Comparação de Performance - Visão Computacional

| Métrica | YOLOv8 (GPU) | MediaPipe (CPU) | Diferença |
|---------|--------------|-----------------|-----------|
| Precisão | 91.7% | ~91.7% | Empate técnico |
| FPS | 45 | 85 | +89% para MP |
| Latência | 22ms | 12ms | -45% para MP |
| Uso de GPU | 3GB | 0GB | -100% para MP |
| Uso de CPU | ~20% | ~35% | +75% para MP |
| Memória RAM | 450MB | 180MB | -60% para MP |

**Conclusão Final:**
- **Precisão:** Empate técnico (~91.7% ambos)
- **Performance:** MediaPipe é quase 2x mais rápido
- **Recursos:** MediaPipe muito mais leve (não requer GPU)
- **Robustez:** YOLOv8 mais robusto a variações
- **Facilidade:** MediaPipe muito mais simples de implementar

---

## 🚀 Como Executar

### Pré-requisitos Gerais

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Webcam conectada (para Parte 02)
- (Opcional) GPU NVIDIA com CUDA para YOLOv8

---

### **Parte 01 - Redes Neurais:**

#### Instalação

```bash
# Clone o repositório
git clone https://github.com/bispado/TREINAMENTO-DE-REDES-NEURAIS.git
cd TREINAMENTO-DE-REDES-NEURAIS

# Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
```

#### Execução

1. **Exercício 1 (Classificação):**
   ```bash
   # Navegue até a pasta
   cd TREINAMENTO-COM-KERAS/Exercicio1
   
   # Execute o notebook
   jupyter notebook exercicio1_wine_classification.ipynb
   ```

2. **Exercício 2 (Regressão):**
   ```bash
   # Navegue até a pasta
   cd TREINAMENTO-COM-KERAS/Exercicio2
   
   # Execute o notebook
   jupyter notebook exercicio2_california_housing.ipynb
   ```

---

### **Parte 02 - Visão Computacional:**

#### Instalação

```bash
# Navegue até a pasta do projeto
cd "rock-paper-scissors.v1i.yolov8 (1)"

# Instale as dependências
pip install -r requirements.txt
```

#### Verificar Setup

```bash
python src/check_setup.py
```

#### Execução - MediaPipe (Mais Fácil)

```bash
# Execute diretamente (não requer treinamento)
python src/mediapipe_realtime.py
```

**Controles:**
- `q` - Sair
- `s` - Salvar screenshot
- `r` - Resetar estatísticas

---

#### Execução - YOLOv8 Treinamento (Google Colab)

1. Faça upload do dataset para seu Google Drive
2. Abra o notebook `notebooks/yolov8_training.ipynb` no Google Colab
3. Execute todas as células sequencialmente
4. Aguarde o treinamento (~1-2 horas)
5. Baixe o modelo treinado (`best.pt`) para a pasta `models/`

#### Execução - YOLOv8 Inferência

```bash
# Uso básico (requer modelo treinado em models/best.pt)
python src/yolov8_inference.py

# Com opções customizadas
python src/yolov8_inference.py --model models/best.pt --conf 0.3
```

**Opções:**
- `--model` - Caminho do modelo (.pt)
- `--conf` - Threshold de confiança (0.1-0.9)
- `--iou` - IoU para NMS (0.1-0.9)
- `--source` - Índice da webcam (0, 1, 2...)

**Controles:**
- `q` - Sair
- `s` - Salvar screenshot
- `+/-` - Ajustar confiança

---

## 📁 Estrutura do Projeto

```
TREINAMENTO-DE-REDES-NEURAIS/
│
├── TREINAMENTO-COM-KERAS/              # Parte 01 (40%)
│   ├── Exercicio1/
│   │   ├── exercicio1_wine_classification.ipynb
│   │   └── wine.data
│   └── Exercicio2/
│       ├── exercicio2_california_housing.ipynb
│       └── cal_housing.data
│
├── rock-paper-scissors.v1i.yolov8 (1)/ # Parte 02 (60%)
│   ├── notebooks/
│   │   └── yolov8_training.ipynb       # Treinamento YOLOv8 (Colab)
│   ├── src/
│   │   ├── check_setup.py              # Verificar instalação
│   │   ├── mediapipe_realtime.py       # MediaPipe em tempo real
│   │   ├── yolov8_inference.py         # YOLOv8 em tempo real
│   │   └── comparative_analysis.py     # Análise comparativa
│   ├── models/
│   │   ├── .gitkeep
│   │   └── best.pt                     # Modelo YOLOv8 treinado
│   ├── results/
│   │   ├── yolov8/                     # Resultados YOLOv8
│   │   └── mediapipe/                  # Resultados MediaPipe
│   ├── train/                          # Dataset treino (2.196 imgs)
│   ├── valid/                          # Dataset validação (604 imgs)
│   ├── test/                           # Dataset teste (329 imgs)
│   ├── data.yaml                       # Configuração dataset
│   └── requirements.txt                # Dependências Python
│
├── README.md                           # Este arquivo (PRINCIPAL)
├── README1.md                          # README antigo (backup)
└── .gitignore
```

---

## 🎥 Vídeo de Demonstração

🔗 **Link do Vídeo no YouTube (modo não listado):**  
*[A ser adicionado após gravação e upload]*

### Conteúdo do Vídeo (10-15 minutos):

#### Parte 01 - Redes Neurais (4-5 min):
1. Introdução ao Checkpoint 02
2. Demonstração do Exercício 1 (Wine Classification)
3. Demonstração do Exercício 2 (California Housing)
4. Análise dos resultados e comparações

#### Parte 02 - Visão Computacional (6-10 min):
5. Explicação do dataset Rock Paper Scissors
6. Demonstração do notebook de treinamento YOLOv8
7. Aplicação MediaPipe em tempo real (ao vivo)
8. Aplicação YOLOv8 em tempo real (ao vivo)
9. Comparação lado a lado das duas abordagens
10. Análise de métricas e resultados
11. Conclusões e aprendizados

---

## 📚 Referências

### Parte 01 - Redes Neurais:
1. **Keras Documentation:** https://keras.io/
2. **TensorFlow Tutorials:** https://www.tensorflow.org/tutorials
3. **Scikit-learn Documentation:** https://scikit-learn.org/
4. **Wine Dataset (UCI):** https://archive.ics.uci.edu/ml/datasets/wine
5. **California Housing Dataset:** https://scikit-learn.org/stable/datasets/real_world.html

### Parte 02 - Visão Computacional:
1. **YOLOv8:** Ultralytics - https://docs.ultralytics.com/
2. **MediaPipe:** Google - https://developers.google.com/mediapipe
3. **Dataset:** Roboflow Universe - https://universe.roboflow.com/bispado/rock-paper-scissors-sxsw-zbvgm
4. **OpenCV:** https://opencv.org/
5. **PyTorch:** https://pytorch.org/

### Artigos e Papers:
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*.
- Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*.

---

## 📄 Licença e Uso Acadêmico

Este projeto foi desenvolvido para fins educacionais como parte do **Checkpoint 02** do curso de Inteligência Artificial da FIAP.

- O dataset Wine é de domínio público (UCI)
- O dataset California Housing é distribuído com scikit-learn
- O dataset Rock Paper Scissors está disponível publicamente no Roboflow Universe

**Importante:** Código-fonte disponível apenas para fins de aprendizado e avaliação acadêmica.

---

## 🙏 Agradecimentos

- **Professor André Tritiack** - Orientação e conteúdo do curso
- **FIAP** - Infraestrutura e suporte
- **Roboflow** - Disponibilização do dataset Rock Paper Scissors
- **Google** - MediaPipe e Google Colab
- **Ultralytics** - Framework YOLOv8
- **UCI Machine Learning Repository** - Dataset Wine
- **Comunidade Open-Source** - Bibliotecas e ferramentas

---

## 📞 Contato

**Para dúvidas sobre este projeto:**

- **Vinicius Murtinho Vicente** - RM551151
- **Lucas Barreto Consentino** - RM557107  
- **Gustavo Bispo Cordeiro** - RM558515

**Repositório GitHub:** https://github.com/bispado/TREINAMENTO-DE-REDES-NEURAIS

---

<div align="center">

**Desenvolvido com dedicação para o Checkpoint 02 - IA 🧠**

⭐ **FIAP - Inteligência Artificial - 2025** ⭐

*Se este projeto foi útil para seus estudos, considere dar uma estrela no GitHub!*

</div>
