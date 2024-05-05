import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title('Previsão Inicial de Custo para a Franquia')

dados = pd.read_csv("slr12.csv", sep=";")

x =  dados[['FrqAnual']]
y =  dados['CusInic']

modelo = LinearRegression().fit(x,y)

col1, col2 = st.columns(2)

with col1:
    st.header("dados")
    st.table(dados.head(10))

with col2:
    st.header("grafico de Dispersão")
    fig, ax = plt.subplots()
    ax.scatter(x, y, color= 'blue')
    ax.plot(x, modelo.predict(x), color="red")
    st.pyplot(fig)

st.header("Valor Anual de Franquia:")
novo_valor = st.number_input("Insira Nova Valor", min_value=1.0, max_value= 99999999.0, value=1500.0, step=0.01)
processar = st.button("Processar")

if processar:
    dados_novo_valor =  pd.DataFrame([[novo_valor]], columns=["FrqAnual"])
    prev = modelo.predict(dados_novo_valor)
    st.header(f"Previsão de Custo Inicial R$: {prev[0]:.2f}")
