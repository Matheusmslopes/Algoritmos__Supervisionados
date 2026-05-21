from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instância do FastAPI
app = FastAPI()

# Criar uma classe para validar os dados do request
class request_body(BaseModel):
    horas_estudo: float

# Carregar o modelo para realizar a predição
modelo_regressao = joblib.load('./modelo_regressao.pkl')

@app.post('/predict')
def predict(data: request_body):
    # Preparar os dados para predição
    input_feature = [[data.horas_estudo]]

    # Realziar a predição
    y_pred = modelo_regressao.predict(input_feature)[0].astype(int)

    return {'pontuacao_teste': y_pred.tolist()}