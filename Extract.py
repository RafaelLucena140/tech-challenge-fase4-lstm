import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Definindo os parâmetros de busca
symbol = 'PETR4.SA'
start_date = '2018-01-01'
end_date = '2026-05-15' # Data atualizada para pegar o histórico mais recente

# 2. Coletando os dados
print(f"Baixando dados para {symbol}...")
df = yf.download(symbol, start=start_date, end=end_date)

# 3. Visualizando as primeiras linhas para garantir que a importação funcionou
print(df.head())

# Vamos isolar a coluna 'Close' (Fechamento) que é o valor que o modelo LSTM vai predizer
df_close = df[['Close']]

# Plot rápido para visualizar o histórico de preços
df_close.plot(figsize=(10, 5), title=f'Histórico de Preço de Fechamento - {symbol}')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.show()