import os
import random
import mlflow
import numpy as np
import random as python_random # Renomeado para evitar conflito com o módulo 'random' se usado diretamente
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input # Importa Input diretamente, que é mais idiomático para a primeira camada

import pandas as pd
# import matplotlib.pyplot as plt # Removido, pois não é usado no código fornecido
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def reset_seeds() -> None:
  """
  Reseta as seeds para garantir a reprodutibilidade dos resultados
  em TensorFlow, NumPy e no módulo padrão 'random'.
  """
  os.environ['PYTHONHASHSEED'] = str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  python_random.seed(42) # Usando a variável renomeada para o módulo random


def read_data():
    """
    Lê o dataset de saúde fetal de um repositório GitHub.
    Separa as features (X) e o target (y).
    """
    url = 'raw.githubusercontent.com'
    username = 'renansantosmendes'
    repository = 'lectures-cdas-2023'
    file_name = 'fetal_health_reduced.csv'
    data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return X, y

def process_data(X, y):
    """
    Processa os dados:
    1. Normaliza as features usando StandardScaler.
    2. Divide os dados em conjuntos de treino e teste (70/30).
    3. Ajusta os rótulos do target (y) para que comecem de 0 (0, 1, 2),
       necessário para 'sparse_categorical_crossentropy' no Keras.
    """
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    # Adicionado 'stratify=y' para garantir que a distribuição das classes
    # seja mantida nos conjuntos de treino e teste, o que é importante para classificação.
    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    y_train = y_train - 1 # Ajusta os rótulos de 1,2,3 para 0,1,2
    y_test = y_test - 1   # Ajusta os rótulos de 1,2,3 para 0,1,2
    return  X_train, X_test, y_train, y_test


def create_model(input_shape: int):
    """
    Cria e compila um modelo de rede neural sequencial.
    A camada de entrada é configurada com base no 'input_shape' fornecido.
    O modelo usa duas camadas Dense com ativação ReLU e uma camada de saída
    com ativação softmax para 3 classes.
    """
    reset_seeds() # Garante que a inicialização dos pesos do modelo seja reprodutível
    model = Sequential()
    # Usa Input como a primeira camada para definir o formato de entrada
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=3, activation='softmax')) # 3 classes de saída (0, 1, 2)

    model.compile(loss='sparse_categorical_crossentropy', # Adequado para rótulos inteiros
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def config_mlflow():
    """
    Configura o MLflow para rastreamento de experimentos.
    Define o usuário, senha e URI de rastreamento.
    Ativa o autologging para TensorFlow/Keras, registrando modelos,
    exemplos de entrada e assinaturas de modelo automaticamente.
    """
    # ATENÇÃO: Armazenar credenciais diretamente no código não é seguro para produção.
    # Considere usar variáveis de ambiente ou um gerenciador de segredos.
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow')

    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)

def train_model(model: keras.Model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Treina o modelo de Machine Learning e registra o experimento no MLflow.
    Usa o conjunto de teste para validação durante o treinamento.
    """
    with mlflow.start_run(run_name='experiment_mlops_ead') as run:
      model.fit(X_train,
                y_train,
                epochs=50,
                validation_data=(X_test, y_test), # Usando o conjunto de teste para validação
                verbose=1) # verbose=1 mostra o progresso do treinamento


if __name__ == "__main__":
    # 1. Leitura dos dados
    X_raw, y_raw = read_data() # Renomeado para evitar conflito com X_df dentro de process_data

    # 2. Processamento dos dados
    X_train, X_test, y_train, y_test = process_data(X_raw, y_raw) # Passa as variáveis corretas

    # 3. Criação do modelo
    # Obtém o número de features para a camada de entrada do modelo
    input_features_shape = X_train.shape[1]
    model = create_model(input_features_shape) # Passa o shape correto para a função

    # 4. Configuração do MLflow
    config_mlflow()

    # 5. Treinamento do modelo
    # Passa todos os dados necessários para a função de treinamento
    train_model(model, X_train, y_train, X_test, y_test)

    print("\nTreinamento concluído e métricas logadas no MLflow.")
