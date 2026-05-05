import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Универсальный прогноз нагрузки (ML)")

# =========================
# 📂 ЗАГРУЗКА ДАННЫХ
# =========================
uploaded_file = st.file_uploader("📂 Загрузите CSV файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл загружен пользователем")
else:
    df = pd.read_csv("hour.csv")
    st.info("Используется стандартный dataset")

# =========================
# 🧠 ПОИСК КОЛОНОК
# =========================
col_map = {
    "hr": ["hr", "hour", "Hour", "time_hour"],
    "weekday": ["weekday", "day", "day_of_week", "week_day"],
    "cnt": ["cnt", "count", "traffic", "value", "demand"]
}

def find_column(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None

hr_col = find_column(df, col_map["hr"])
weekday_col = find_column(df, col_map["weekday"])
cnt_col = find_column(df, col_map["cnt"])

if hr_col is None or weekday_col is None or cnt_col is None:
    st.error("❌ Нужные колонки не найдены! Требуются hr, weekday, cnt (или аналоги)")
    st.stop()

# =========================
# 🔄 ПЕРЕИМЕНОВАНИЕ
# =========================
df = df[[hr_col, weekday_col, cnt_col]]

df.rename(columns={
    hr_col: "hr",
    weekday_col: "weekday",
    cnt_col: "cnt"
}, inplace=True)

# =========================
# 🧹 FEATURE ENGINEERING
# =========================
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['lag_1'] = df['cnt'].shift(1)
df['lag_24'] = df['cnt'].shift(24)
df = df.dropna()

# =========================
# 🤖 MODEL
# =========================
X = df[['hr', 'weekday', 'is_weekend', 'lag_1', 'lag_24']]
y = df['cnt']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# 📊 GRAPH
# =========================
st.subheader("📈 Нагрузка")
st.line_chart(df['cnt'].tail(200))

# =========================
# ⚙️ INPUT
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
# 🔮 PREDICTION
# =========================
if st.button("🔮 Предсказать"):
    data = np.array([[hr, weekday, is_weekend, lag_1, lag_24]])
    pred = model.predict(data)[0]

    st.success(f"Прогноз нагрузки: {pred:.2f}")