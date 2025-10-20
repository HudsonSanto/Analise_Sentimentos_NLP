# Análise de Sentimentos em Tweets usando NLP

## Visão Geral
Este projeto implementa um pipeline completo de **Processamento de Linguagem Natural (NLP)** para classificação de sentimentos em tweets, utilizando um dataset de aproximadamente **100.000 postagens** coletadas do Twitter entre 01/08/2018 e 20/10/2018. Cada tweet é rotulado em três classes:

- **0**: Negativo  
- **1**: Positivo  
- **2**: Neutro  

O foco técnico reside na engenharia de features textuais, otimização de hiperparâmetros e avaliação de modelos de classificação multiclasse, com ênfase em métricas balanceadas para detecção de ambiguidades em linguagem informal.

O notebook principal `Analise_Sentimento.ipynb` executa o fluxo end-to-end, incluindo pré-processamento, modelagem e geração de predições para submissão.

---

## Estrutura do Dataset

| Coluna       | Descrição                                                                 |
|-------------|---------------------------------------------------------------------------|
| `id`        | Identificador único do tweet (int64)                                      |
| `tweet_text`| Texto da postagem (str; principal feature para extração de features)     |
| `tweet_date`| Timestamp da publicação (datetime)                                        |
| `sentiment` | Rótulo alvo: 0 (negativo), 1 (positivo), 2 (neutro) (int)               |
| `query_used`| Filtro de busca original (ex.: :), :(, #fato)                             |

- **Dimensões**: Treino (~100K amostras); Teste (~28K amostras sem rótulos)  
- **Distribuição de Classes (Treino)**: Aproximadamente balanceada (~33% por classe)

---

## Objetivo Técnico
Construir e otimizar um classificador multiclasse para predizer o sentimento de tweets com base no texto bruto, maximizando **F1-score macro** em cenários de texto ruidoso (emojis, abreviações, multilíngue parcial).  

Aplicações incluem análise de sentimento em tempo real para monitoramento de redes sociais e detecção de churn.

---

## Metodologia

### 1. Análise de Consistência e Exploração
- **Verificação de integridade**: Duplicatas, valores ausentes e outliers via Pandas  
- **EDA**: Distribuição de classes, comprimento médio de tweets, análise de frequência de termos (WordCloud)

### 2. Pré-processamento e Engenharia de Features
- **Limpeza**: URLs, menções (@), hashtags (#), acentos (unidecode), stopwords em português (NLTK)  
- **Tokenização e Lematização**: NLTK WordNetLemmatizer para normalização  
- **Vetorização**: TF-IDF (`TfidfVectorizer`) com n-grams (1-2)  
- **Divisão**: Train-test split 80/20 com estratificação por classe

### 3. Modelagem e Avaliação
- **Modelo Principal**: Regressão Logística multiclasse (`solver='saga'`, `multi_class='multinomial'`)  
- **Otimização**: Optuna (penalty, C, tol, l1_ratio, max_iter; 100 trials com TPE sampler)  
- **Métricas**: Accuracy, Precision, Recall, F1-score (macro/micro/weighted), Matriz de Confusão (mlxtend), ROC-AUC multiclasse  
- **Ensemble Testado**: Random Forest, XGBoost, LightGBM, CatBoost (baseline comparativo)  
- **Predições**: Treinamento completo no treino; transformação no teste; exportação para CSV

---

## Bibliotecas e Dependências Técnicas
- **Processamento de Texto**: NLTK, SpaCy (opcional para NER), Gensim (Word2Vec/Doc2Vec exploratório)  
- **ML/Feature Engineering**: Scikit-learn (Pipeline, CountVectorizer/TfidfVectorizer, métricas)  
- **Otimização e Visualização**: Optuna, Matplotlib, Seaborn, Plotly  
- **Outros**: Pandas, NumPy, Unidecode, MLxtend  
- **Ambiente**: Python 3.11+, testado em Google Colab com GPU T4

---

## Resultados e Análise
- **Melhores Hiperparâmetros (Optuna)**:  
  `penalty='l2'`, `C=1.39`, `tol=8.65e-4`, `max_iter=269`, `warm_start=True`
- **Importância de Features**: penalty domina (~67%)

### Relatório de Classificação (Teste, n=28.500)

| Classe       | Precision | Recall | F1-Score | Suporte |
|-------------|----------|-------|----------|---------|
| Negativo (0)| 0.75     | 0.72  | 0.74     | 9.559   |
| Positivo (1)| 0.71     | 0.72  | 0.71     | 9.555   |
| Neutro (2)  | 0.93     | 0.94  | 0.93     | 9.386   |
| **Macro Avg**| 0.79    | 0.80  | 0.80     | 28.500  |
| **Weighted Avg**|0.79  | 0.79  | 0.79     | 28.500  |

- **Accuracy Geral**: 79%  
- **Insights**: Alto desempenho em neutros; desafios em positivos por polaridade implícita  
- **Baseline (Random Forest)**: ~76% accuracy  

**Predições exportadas**: `predictions.csv` (índice por `id`, coluna `sentiment_predict`)

---

## Execução

### Instalação
```bash
pip install pandas numpy seaborn matplotlib nltk spacy unidecode wordcloud scikit-learn gensim optuna xgboost lightgbm catboost mlxtend
# Gere requirements.txt
pip freeze > requirements.txt
