from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS # Import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Caminhos para o modelo e dados
MODEL_PATH = 'notebooks/models/fiscal_stability_model.joblib'
FEATURED_DATA_PATH = 'notebooks/data/04_features/featured_data.csv'

# Carregar o modelo treinado
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo carregado com sucesso de {MODEL_PATH}")
except FileNotFoundError:
    print(f"Erro: Modelo não encontrado em {MODEL_PATH}. Certifique-se de que o modelo foi treinado e salvo corretamente.")
    model = None

# Carregar dados para obter os nomes das features
try:
    df_features = pd.read_csv(FEATURED_DATA_PATH)
    # Excluir 'country_id' e 'year' se existirem, pois não são features para o modelo
    feature_names = [col for col in df_features.columns if col not in ['country_id', 'year']]
    print(f"Nomes das features carregados com sucesso de {FEATURED_DATA_PATH}")
except FileNotFoundError:
    print(f"Erro: Arquivo de features não encontrado em {FEATURED_DATA_PATH}. Certifique-se de que os dados foram processados.")
    feature_names = []

@app.route('/', methods=['GET'])
def home():
    return "API de Predição de Estabilidade Fiscal está funcionando!"

@app.route('/features', methods=['GET'])
def get_feature_names():
    if not feature_names:
        return jsonify({"error": "Nomes das features não carregados."}), 500
    return jsonify({"feature_names": feature_names})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not feature_names:
        return jsonify({"error": "Modelo ou nomes das features não carregados."}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Nenhum dado JSON fornecido."}), 400

        # Converter os dados de entrada para um DataFrame do pandas
        # Garantir que a ordem das colunas é a mesma que a do treinamento do modelo
        input_df = pd.DataFrame([data])
        
        # Reordenar as colunas para corresponder à ordem esperada pelo modelo
        # Preencher com NaN e depois com 0 se alguma feature estiver faltando
        input_df = input_df.reindex(columns=feature_names, fill_value=0.0)

        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction[0].item()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Para desenvolvimento, use app.run()
    # Em produção, use Gunicorn ou similar
    app.run(host='0.0.0.0', port=5000, debug=True)
