import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
yf.pdr_override()

input_ativo = st.sidebar.text_input('Digite o ativo que quer prever', 'CPLE6')

def baixar_por_intevalo(ativo=f'{input_ativo}.SA', inicio='2015-01-01', fim=datetime.date.today()):
    ativo = ativo
    inicio = inicio
    fim = fim
    df_baixada = yf.download(ativo, inicio, fim)
    return df_baixada

def baixar_por_ano(ativo=f'{input_ativo}.SA', anos='9y'):
    ativo = ativo
    ticket = yf.Ticker(ativo)
    df_baixada = ticket.history(period=anos)
    return df_baixada

df_baixada = baixar_por_intevalo()

df = df_baixada[:]
ultima_linha = df_baixada[-1:]

df['mm9'] = df['Close'].rolling(9).mean()
df['mm21'] = df['Close'].rolling(21).mean()
df['Close'] = df['Close'].shift(-1)
df = df.dropna()

total = len(df)
treino = total - 700
teste = total - 15
st.sidebar.subheader("Dados da análise")
linhas = f'Dados da previsão:\n - Dados de treino 0:{treino}\n - Dados de teste {treino}:{teste}\n - Dados para validação (15 linhas) {teste}:{total}\n'
st.sidebar.warning(linhas)
df = df.reset_index()
x_features = df.drop(['Date', 'Adj Close', 'Close'], axis=1)
x_features_list = list(x_features.columns)
y_labels = df['Close']

from sklearn.feature_selection import SelectKBest
k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(x_features, y_labels)
k_best_features_score = k_best_features.scores_
melhores = zip(x_features_list, k_best_features_score)
melhores_ordenados = list(reversed(sorted(melhores, key=lambda x: x[1])))
melhores_variaveis = dict(melhores_ordenados[:15])

#st.sidebar.write(f'Melhores: {melhores_variaveis}')
x_features = x_features.drop('Volume', axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_features)
x_features_normalizado = scaler.fit_transform(x_features)
x_features_normal = pd.DataFrame(x_features_normalizado, columns=list(x_features.columns))

x_train = x_features_normal[0:treino]
x_test = x_features_normal[treino:teste]

y_train = y_labels[0:treino]
y_test = y_labels[treino:teste]

st.subheader(f'Previsão dos ultimos 15 dias com um modelo que aprende com dados desde 2015 até a ultima cotação.')
st.write(f'Ainda não criei validações, então se o gráfico ficar estranho (além do normal) tente após as 10:40 que é o horário que geralmente o yfinance atualiza')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr = LinearRegression()
lr.fit(x_train, y_train)
y_predito = lr.predict(x_test)

coeficiente = r2_score(y_test, y_predito)


# Previsão
previsao = x_features_normal[teste:total]
dia = df['Date'][teste:total]
real = df['Close'][teste:total]

y_pred = lr.predict(previsao)

df2 = pd.DataFrame({'Data': dia, 'Cotacao': real, 'Previsto': y_pred})
df2['Cotacao'] = df2['Cotacao'].shift(+1)
# Nessa linha acima, estamos devolvendo as cotações ao valor verdadeiro, isto é, desfazemos o que fizemos acima

df2 = df2.dropna()
df2.round(2)

df2['Erro'] = df2['Cotacao'] - df2['Previsto']
df2.round(2)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(16, 8))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_tick_params(rotation=30)

ax.set_title(f'Previsão dos ultimos 15 dias com coeficiente de {round(coeficiente * 100, 2)}%- By J. Brutus',
             fontsize=16)
ax.set_ylabel('Preço do ativo em R$', fontsize=14)
ax.plot(df2['Data'], df2['Cotacao'], marker='o', label='Cotação Real', color='blue')
ax.plot(df2['Data'], df2['Previsto'], marker='o', label='Cotação Prevista', color='red')

plt.grid()
plt.legend()
plt.show()


df_baixada = df_baixada.reset_index()
df_baixada['mm7'] = df_baixada['Close'].rolling(7).mean().round(2)
df_baixada['mm21'] = df_baixada['Close'].rolling(21).mean().round(2)
x_df_baixada = df_baixada.drop(['Date', 'Close', 'Adj Close', 'Volume'], axis=1)

scaler.fit(x_df_baixada)
x_df_baixada_norm = scaler.fit_transform(x_df_baixada)

dados_df_baixada = pd.DataFrame(x_df_baixada_norm, columns=[list(x_df_baixada.columns)])

prever_proximo_dia = dados_df_baixada[-1:]

y_de_amanha = lr.predict(prever_proximo_dia)

df_y_de_amanha = pd.DataFrame(y_de_amanha, columns=['Preco de Amanha'], )



fig, ax = plt.subplots(figsize=(16, 8))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_tick_params(rotation=30)

ax.set_title(
    f'Previsor de Preço Estrelinha\n\n- Precisão de {round(coeficiente * 100, 2)}% - By J. Brutus',
    fontsize=16)
ax.set_ylabel('Preço do ativo em R$', fontsize=14)
ax.plot(df2['Data'], df2['Cotacao'], marker='o', label='Cotação Real', color='blue')
ax.plot(df2['Data'], df2['Previsto'], marker='o', label='Cotação Prevista', color='red')

# Aqui colocamos nossa previsão de preço de amanhã:
ax.scatter(x=datetime.date.today(), y=y_de_amanha, color='green', marker='X', label='Previsão do próximo dia')

plt.grid()
plt.legend()
plt.show()

st.pyplot(fig)
st.subheader('Tabela com dados das previsões dos ultimos 15 dias')
df2.set_index(df2['Data'])
df2.round(2)
st.dataframe(df2)
