import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Универсальная система прогнозирования (ML)")

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
# 👀 ПРОСМОТР ДАННЫХ
# =========================
st.subheader("📄 Предпросмотр данных")
st.dataframe(df.head())

# =========================
# 🧠 ВЫБОР КОЛОНОК (БЕЗ ОГРАНИЧЕНИЙ)
# =========================
st.subheader("🧠 Выберите колонки для модели")

columns = df.columns.tolist()

target_col = st.selectbox("🎯 Целевая переменная (что предсказываем)", columns)

feature_cols = st.multiselect(
    "📌 Признаки (что используем для прогноза)",
    [col for col in columns if col != target_col]
)

if len(feature_cols) == 0:
    st.warning("Выберите хотя бы 1 признак")
    st.stop()

# =========================
# 🧹 УДАЛЕНИЕ ПУСТЫХ ДАННЫХ
# =========================
df = df[feature_cols + [target_col]].dropna()

# =========================
# 🤖 МОДЕЛЬ
# =========================
X = df[feature_cols]
y = df[target_col]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.success("Модель обучена!")

# =========================
# 📊 ГРАФИК
# =========================
st.subheader("📈 Данные (целевая переменная)")
st.line_chart(df[target_col].tail(200))

# =========================
# ⚙️ ВВОД ДАННЫХ ДЛЯ ПРОГНОЗА
# =========================
st.subheader("⚙️ Ввод для предсказания")

input_data = []

cols1 = st.columns(min(len(feature_cols), 3))

for i, col in enumerate(feature_cols):
    with cols1[i % len(cols1)]:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data.append(val)

# =========================
# 🔮 ПРЕДСКАЗАНИЕ
# =========================
if st.button("🔮 Предсказать"):
    data = np.array([input_data])
    pred = model.predict(data)[0]

    st.success(f"📊 Прогноз: {pred:.2f}")