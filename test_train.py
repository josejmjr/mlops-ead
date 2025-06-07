import pandas as pd
import pytest
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split  # Importar para usar no teste de train_model

# Importa as funções do módulo 'train' (assumindo que o código do Canvas está em train.py)
from train import (read_data,
                   process_data,  # Adicionado para usar no teste de train_model
                   create_model,
                   train_model)


@pytest.fixture
def sample_data():
    """
    A fixture function that returns a sample dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing sample data with three columns: 'feature1',
         'feature2', and 'fetal_health'.
    """
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'fetal_health': [1, 1, 2, 3, 2, 1, 3, 2, 1, 3]  # Mais dados para uma divisão mais significativa
    })
    return data


def test_read_data():
    """
    This function tests the `read_data` function. It checks whether the returned data is not
     empty for both features (X) and labels (y).

    Parameters:
    None

    Returns:
    None
    """
    X, y = read_data()

    assert not X.empty
    assert not y.empty


def test_create_model():
    """
    Tests the `create_model` function to ensure it returns a valid Keras Sequential model.
    It checks the number of layers, if the model is trainable, and its type.
    """
    X_raw, _ = read_data()  # Lê os dados brutos
    # process_data é chamada para obter o X_train e o input_shape correto
    X_train, _, _, _ = process_data(X_raw, pd.Series([1] * len(X_raw)))  # y fictício para process_data

    # create_model espera o input_shape (número de features), não o DataFrame X completo
    model = create_model(X_train.shape[1])

    assert len(model.layers) > 2
    assert model.trainable
    assert isinstance(model, Sequential)


def test_train_model(sample_data):
    """
    Tests the `train_model` function. It prepares sample data, creates a model,
    and then calls `train_model` to ensure the training process runs and
    populates the model's history with loss and accuracy metrics.
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health']

    # Ajusta os rótulos para 0, 1, 2 conforme a função process_data faz
    y_adjusted = y - 1

    # Divide os dados de amostra em treino e teste, como a função process_data faria
    # Usando stratify para garantir a proporção das classes, mesmo com dados pequenos
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_adjusted,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y_adjusted)

    # create_model espera o input_shape (número de features)
    model = create_model(X_train.shape[1])

    # train_model espera model, X_train, y_train, X_test, y_test
    # Removido o parâmetro 'is_train=False' pois não existe na função train_model do Canvas
    train_model(model, X_train, y_train, X_test, y_test)

    # Verifica se o histórico de treinamento foi preenchido
    assert 'loss' in model.history.history
    assert 'accuracy' in model.history.history
    assert 'val_loss' in model.history.history
    assert 'val_accuracy' in model.history.history

    # Verifica se os valores finais não são NaN (indicando que o treinamento ocorreu)
    assert not pd.isna(model.history.history['loss'][-1])
    assert not pd.isna(model.history.history['val_loss'][-1])
