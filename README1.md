# ATIVIDADE – TREINAMENTO DE REDES NEURAIS COM KERAS (DADOS TABULARES)

Esta atividade corresponde a 40% da nota do Checkpoint 2.

## Integrantes do Grupo
- **Vinicius Murtinho Vicente** - RM551151
- **Lucas Barreto Consentino** - RM557107  
- **Gustavo Bispo Cordeiro** - RM558515

---

## Exercício 1 – Classificação Multiclasse

### Dataset: Wine Dataset (UCI)

O exercício implementa uma rede neural em Keras para classificar vinhos em 3 classes diferentes, comparando os resultados com modelos tradicionais do scikit-learn.

### Objetivos:
1. **Treinar uma rede neural em Keras** com a configuração especificada:
   - 2 camadas ocultas com 32 neurônios cada
   - Função de ativação ReLU nas camadas ocultas
   - Camada de saída com 3 neurônios e função de ativação Softmax
   - Função de perda: categorical_crossentropy
   - Otimizador: Adam

2. **Comparar com modelos do scikit-learn**:
   - RandomForestClassifier
   - LogisticRegression

3. **Avaliar e discutir os resultados**:
   - Métricas de acurácia
   - Classification reports
   - Matrizes de confusão
   - Análise comparativa dos modelos

---

## Exercício 2 – Regressão

### Dataset: California Housing Dataset (Scikit-learn)

O exercício implementa uma rede neural em Keras para prever o valor médio das casas na Califórnia, comparando os resultados com modelos de regressão do scikit-learn.

### Objetivos:
1. **Treinar uma rede neural em Keras** com a configuração especificada:
   - 3 camadas ocultas com 64, 32 e 16 neurônios
   - Função de ativação ReLU nas camadas ocultas
   - Camada de saída com 1 neurônio e função de ativação Linear
   - Função de perda: mse
   - Otimizador: Adam

2. **Comparar com modelos do scikit-learn**:
   - LinearRegression
   - RandomForestRegressor

3. **Avaliar e discutir os resultados**:
   - Métricas de erro (RMSE, MAE)
   - Coeficiente de determinação (R²)
   - Visualizações das predições
   - Análise comparativa dos modelos

---

## Arquivos do Projeto:
- `exercicio1_wine_classification.ipynb` - Notebook do Exercício 1 (Classificação)
- `exercicio2_california_housing.ipynb` - Notebook do Exercício 2 (Regressão)
- `wine.data` - Dataset Wine (UCI)
- `wine.names` - Descrição do dataset Wine

## Como Executar:

### Dependências:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
```

### Execução:
1. **Exercício 1**: Execute o notebook `exercicio1_wine_classification.ipynb` célula por célula
2. **Exercício 2**: Execute o notebook `exercicio2_california_housing.ipynb` célula por célula

### Resultados Incluem:
- Análise exploratória dos dados
- Treinamento e avaliação dos modelos
- Comparação visual dos resultados
- Discussão sobre qual modelo teve melhor desempenho
- Métricas detalhadas (acurácia para classificação, RMSE/MAE para regressão)

## Resultados Obtidos:

### Exercício 1 (Classificação - Wine Dataset):

**Métricas de Acurácia:**
| Modelo | Acurácia | Observações |
|--------|----------|-------------|
| **Random Forest** | **100.00%** | Melhor desempenho |
| Logistic Regression | 97.22% | Excelente resultado |
| Keras Neural Network | 91.67% | Bom desempenho |

**Análise:**
- O Random Forest alcançou 100% de acurácia, demonstrando perfeita classificação das 3 classes de vinho
- A Logistic Regression teve excelente desempenho com 97.22%
- A Rede Neural Keras obteve 91.67%, resultado competitivo considerando o tamanho do dataset (178 amostras)
- O dataset Wine possui boa separabilidade entre as classes, permitindo que modelos tradicionais obtenham resultados excelentes

**Conclusão:** Para datasets pequenos e bem estruturados como o Wine, modelos ensemble (Random Forest) tendem a superar redes neurais profundas.

---

### Exercício 2 (Regressão - California Housing Dataset):

**Métricas de Erro:**
| Modelo | RMSE | MAE | R² | Observações |
|--------|------|-----|-----|-------------|
| **Random Forest** | **0.5051** | **0.3274** | **0.8053** | Melhor desempenho |
| Keras Neural Network | 0.5354 | 0.3578 | 0.7812 | Muito competitivo |
| Linear Regression | 0.7456 | 0.5332 | 0.5758 | Baseline |

**Análise:**
- O Random Forest obteve o melhor resultado com RMSE de 0.5051 e R² de 0.8053
- A Rede Neural Keras ficou muito próxima com RMSE de 0.5354 e R² de 0.7812
- A Regressão Linear teve desempenho inferior (R² = 0.5758), indicando relações não-lineares nos dados
- **Feature mais importante (Random Forest):** MedInc (renda média) com 52.49% de importância

**Conclusão:** Para problemas de regressão com datasets maiores (20.640 amostras), tanto Random Forest quanto Redes Neurais demonstram excelente capacidade de capturar padrões complexos, com Random Forest tendo leve vantagem.

---

## Insights Gerais:

1. **Random Forest** se destacou em ambos os exercícios, demonstrando robustez e eficácia
2. **Keras Neural Networks** mostraram desempenho competitivo, especialmente no dataset maior
3. **Modelos tradicionais** (Logistic Regression, Linear Regression) servem como bons baselines
4. A escolha do modelo deve considerar: tamanho do dataset, interpretabilidade necessária e recursos computacionais disponíveis
