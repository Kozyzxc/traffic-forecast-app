import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Прогноз нагрузки трафика")

# =========================
# 📂 ЗАГРУЗКА ДАННЫХ
# =========================
uploaded_file = st.file_uploader("📂 Загрузите CSV файл (или используем дефолтный)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл загружен пользователем")
else:
    df = pd.read_csv("hour.csv")
    st.info("Используется стандартный dataset (hour.csv)")

# =========================
# 🧹 ОБРАБОТКА ДАННЫХ
# =========================
df = df[['hr', 'weekday', 'cnt']]

df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['lag_1'] = df['cnt'].shift(1)
df['lag_24'] = df['cnt'].shift(24)

df = df.dropna()

# =========================
# 🤖 МОДЕЛЬ
# =========================
X = df[['hr', 'weekday', 'is_weekend', 'lag_1', 'lag_24']]
y = df['cnt']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# 📈 ГРАФИК
# =========================
st.subheader("📈 Нагрузка (последние 200 часов)")
st.line_chart(df['cnt'].tail(200))

st.info("Модель учитывает время и прошлые значения нагрузки")

# =========================
# ⚙️ ВВОД ДАННЫХ
# =========================
st.subheader("⚙️ Введите параметры")

col1, col2 = st.columns(2)

with col1:
    hr = st.slider("Час", 0, 23, 12)
    weekday = st.slider("День недели (0=Пн, 6=Вс)", 0, 6, 3)

with col2:
    lag_1 = st.number_input("Нагрузка за прошлый час", value=100.0)
    lag_24 = st.number_input("Нагрузка за прошлые сутки", value=100.0)

is_weekend = 1 if weekday >= 5 else 0

# =========================
# 🔮 ПРЕДСКАЗАНИЕ
# =========================
if st.button("🔮 Предсказать"):
    data = np.array([[hr, weekday, is_weekend, lag_1, lag_24]])
    pred = model.predict(data)[0]

    st.success(f"Прогноз нагрузки: {pred:.2f}")