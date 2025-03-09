# Atvidade-04
# Criado para depositar o código desta atividade, referente a questão 01.

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def get_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()
    
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop(columns=["timestamp"], inplace=True)
    return df

def preprocess_data(df):
    df["price_change"] = df["price"].diff()
    df["target"] = (df["price_change"] > 0).astype(int)
    df.dropna(inplace=True)
    
    df["SMA_5"] = df["price"].rolling(window=5).mean()
    df["SMA_10"] = df["price"].rolling(window=10).mean()
    df["volatility"] = df["price"].pct_change().rolling(window=5).std()
    df.dropna(inplace=True)
    
    features = ["price", "SMA_5", "SMA_10", "volatility"]
    X = df[features]
    y = df["target"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    plt.figure(figsize=(8,5))
    plt.barh(feature_names, importance, color='skyblue')
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.title("Importância das Variáveis no Modelo")
    plt.show()

def main():
    print("Coletando dados...")
    df = get_crypto_data()
    df.to_csv("crypto_data.csv", index=False)
    print("Dados salvos em crypto_data.csv")
    
    print("Processando dados...")
    X, y = preprocess_data(df)
    
    print("Treinando modelo...")
    model, accuracy = train_model(X, y)
    print(f"Acurácia do modelo: {accuracy:.2f}")
    
    print("Exibindo importância das variáveis...")
    plot_feature_importance(model, X.columns)

if __name__ == "__main__":
    main()
