import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from model import StockLSTM
from data_handler import prepare_data

SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = '2026-05-15'
SEQ_LENGTH = 60

print("Preparando os dados de teste...")
X, y, scaler = prepare_data(SYMBOL, START_DATE, END_DATE, SEQ_LENGTH)

train_size = int(len(X) * 0.8)
X_test = X[train_size:]
y_test = y[train_size:]

print("Carregando o modelo LSTM...")
# input_size=1 para carregar os pesos corretamente
model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
model.load_state_dict(torch.load('lstm_petr4_model.pth'))
model.eval()

with torch.no_grad():
    predictions = model(X_test)

predictions_np = predictions.numpy()
y_test_np = y_test.numpy().reshape(-1, 1)

predictions_real = scaler.inverse_transform(predictions_np)
y_test_real = scaler.inverse_transform(y_test_np)

mae = mean_absolute_error(y_test_real, predictions_real)
rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
mape = mean_absolute_percentage_error(y_test_real, predictions_real) * 100

print("\n--- RESULTADOS DA AVALIAÇÃO (APENAS PREÇO) ---")
print(f"MAE (Erro Absoluto Médio): R$ {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): R$ {rmse:.2f}")
print(f"MAPE (Erro Percentual Absoluto Médio): {mape:.2f}%")
print("----------------------------------------------")

plt.figure(figsize=(14, 6))
plt.plot(y_test_real, color='blue', label='Preço Real (PETR4)')
plt.plot(predictions_real, color='red', linestyle='dashed', label='Previsão do Modelo (LSTM)')
plt.title('Previsão de Preços das Ações - Petrobras (Apenas Fechamento)')
plt.xlabel('Dias (Período de Teste)')
plt.ylabel('Preço de Fechamento (R$)')
plt.legend()
plt.grid(True)
plt.show()