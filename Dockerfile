# 1. Escolher a imagem base oficial do Python (versão slim para ser mais leve)
FROM python:3.10-slim

# 2. Definir o diretório de trabalho dentro do contentor
WORKDIR /app

# 3. Copiar apenas o ficheiro de dependências primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# 4. Instalar as bibliotecas necessárias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o resto dos ficheiros do projeto (scripts e o modelo .pth) para o contentor
COPY . .

# 6. Expor a porta que o FastAPI vai utilizar
EXPOSE 8000

# 7. Comando para arrancar a API quando o contentor iniciar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]