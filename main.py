from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from model import StockLSTM

# Inicialização da aplicação FastAPI
app = FastAPI(
    title="API de Previsão de Ações - PETR4",
    description="API para prever o preço de fechamento das ações da Petrobras usando Deep Learning (LSTM).",
    version="1.0"
)

# Variáveis globais para armazenar o modelo e o normalizador em memória
model = None
scaler = None

# Definir o formato dos dados que o usuário deve enviar (Payload)
class StockInput(BaseModel):
    historical_prices: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "historical_prices": [38.5] * 60  # Exemplo fictício preenchido com 60 dias
            }
        }

@app.on_event("startup")
def load_model_and_scaler():
    """Carrega os artefatos locais sem depender da internet."""
    global model, scaler
    print("A carregar o modelo e o scaler para a memória...")
    
    try:
        # 1. Carrega o scaler diretamente do ficheiro salvo pelo train.py
        scaler = joblib.load('scaler_petr4.pkl')
        
        # 2. Carrega o modelo LSTM (Arquitetura simples: input_size=1)
        model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load('lstm_petr4_model.pth'))
        model.eval() # Modo de inferência
        
        print("API pronta para receber requisições!")
    except FileNotFoundError as e:
        print(f"ERRO: Ficheiro não encontrado durante a inicialização: {e}")
        print("Certifique-se de que correu o 'train.py' para gerar o scaler_petr4.pkl e o lstm_petr4_model.pth.")

@app.post("/predict")
def predict_price(data: StockInput):
    """Endpoint principal para fazer previsões."""
    
    # Validação de segurança: garantir que recebemos os 60 dias de histórico
    if len(data.historical_prices) != 60:
        raise HTTPException(
            status_code=400, 
            detail=f"A API requer exatamente 60 dias de histórico. Recebidos: {len(data.historical_prices)}."
        )
    
    try:
        # 1. Converter a lista de entrada num array NumPy e formatar para coluna
        input_array = np.array(data.historical_prices).reshape(-1, 1)
        
        # 2. Normalizar os dados usando o scaler carregado
        scaled_input = scaler.transform(input_array)
        
        # 3. Converter para Tensor do PyTorch com o formato [batch_size, seq_length, input_size]
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)
        
        # 4. Fazer a previsão
        with torch.no_grad():
            prediction = model(tensor_input)
        
        # 5. Desnormalizar o valor previsto (voltar para R$)
        pred_value = prediction.numpy()
        real_prediction = scaler.inverse_transform(pred_value)
        
        return {
            "predicted_price_brl": round(float(real_prediction[0][0]), 2),
            "message": "Previsão realizada com sucesso!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))