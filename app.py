import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Заголовок
st.title("📊 Прогноз нагрузки трафика")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("hour.csv")
    df = df[['hr', 'weekday', 'cnt']]
    
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['lag_1'] = df['cnt'].shift(1)
    df['lag_24'] = df['cnt'].shift(24)
    
    df = df.dropna()
    return df

df = load_data()

X = df[['hr', 'weekday', 'is_weekend', 'lag_1', 'lag_24']]
y = df['cnt']

# Обучение модели
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# График
st.subheader("📈 Нагрузка (последние 200 часов)")
st.line_chart(df['cnt'].tail(200))

st.info("Модель учитывает время и прошлые значения нагрузки")

# Ввод
st.subheader("⚙️ Введите параметры")

col1, col2 = st.columns(2)

with col1:
    hr = st.slider("Час", 0, 23, 12)
    weekday = st.slider("День недели (0=Пн, 6=Вс)", 0, 6, 3)

with col2:
    lag_1 = st.number_input("Нагрузка за прошлый час", value=100.0)
    lag_24 = st.number_input("Нагрузка за прошлые сутки", value=100.0)

is_weekend = 1 if weekday >= 5 else 0

# Предсказание
if st.button("🔮 Предсказать"):
    data = np.array([[hr, weekday, is_weekend, lag_1, lag_24]])
    pred = model.predict(data)[0]

    st.success(f"Прогноз: {pred:.2f}")