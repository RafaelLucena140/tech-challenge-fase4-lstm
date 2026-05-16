import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import StockLSTM
from data_handler import prepare_data
import joblib

# 1. Parâmetros do Modelo
SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = '2026-05-15'
SEQ_LENGTH = 60
BATCH_SIZE = 32

# 2. Preparação de Dados
print("A preparar os dados...")
X, y, scaler = prepare_data(SYMBOL, START_DATE, END_DATE, SEQ_LENGTH)

# 3. Guardar o Scaler para a API (NOVO PASSO)
joblib.dump(scaler, 'scaler_petr4.pkl')
print("Scaler guardado com sucesso como 'scaler_petr4.pkl'")

# 4. Divisão de Treino e Teste
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

# 5. Inicialização do Modelo
model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Ciclo de Treino
epochs = 80
print("\nA iniciar o treino (Apenas Preço de Fecho)...")

for epoch in range(epochs):
    model.train()
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            val_loss = criterion(test_preds, y_test)
        print(f'Época [{epoch+1}/{epochs}], Loss Treino: {loss.item():.6f}, Loss Validação: {val_loss.item():.6f}')

# 7. Guardar o Modelo Treinado
torch.save(model.state_dict(), 'lstm_petr4_model.pth')
print("\nModelo guardado com sucesso como 'lstm_petr4_model.pth'!")