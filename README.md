# Tech Challenge Fase 4 - Predição de Ações com Deep Learning (LSTM)

Este projeto é a entrega final do Tech Challenge da Fase 4 da Pós-Graduação em Machine Learning Engineering. O objetivo é criar um modelo preditivo utilizando redes neurais Long Short-Term Memory (LSTM) para prever o valor de fechamento das ações da Petrobras (PETR4.SA) e realizar o deploy do modelo através de uma API conteinerizada.

## 🚀 Tecnologias Utilizadas
* **Linguagem:** Python 3.10
* **Deep Learning:** PyTorch
* **Manipulação de Dados:** Pandas, NumPy, Scikit-learn, yfinance
* **API:** FastAPI, Uvicorn
* **Deploy:** Docker

## 🧠 Arquitetura do Modelo
Foi desenvolvida uma rede neural LSTM univariável focada no preço de fechamento (`Close`). A escolha de um modelo mais enxuto (sem a variável de Volume) demonstrou maior robustez e menor propensão a ruídos, resultando num modelo mais estável e aderente à tendência de mercado, com um ótimo baseline de MAPE (Mean Absolute Percentage Error).

O modelo utiliza uma janela de tempo (Sliding Window) de 60 dias para prever o valor do dia seguinte (D+1). Os artefatos (`.pth` e `.pkl`) são gerados localmente e carregados na memória pela API para garantir estabilidade, sem dependência de requisições externas na inicialização.

## 📁 Estrutura do Projeto
* `data_handler.py`: Script responsável pela extração (yfinance) e pré-processamento (normalização e janelamento).
* `model.py`: Definição da arquitetura da rede neural LSTM em PyTorch (com Dropout de 10%).
* `train.py`: Orquestrador de treinamento, criação de mini-batches (DataLoader) e salvamento dos artefatos.
* `evaluate.py`: Script para validação do modelo com dados de teste e plotagem de gráficos com as métricas (MAE, RMSE, MAPE).
* `main.py`: Aplicação FastAPI para servir o modelo em produção.
* `Dockerfile`: Receita de conteinerização do projeto.
* `requirements.txt`: Dependências e versões do projeto.

## ⚙️ Como Executar o Projeto (Via Docker)

Certifique-se de que possui o **Docker Desktop** instalado e a correr na sua máquina.

**1. Construir a imagem Docker:**
```bash
docker build -t api-petr4-lstm .