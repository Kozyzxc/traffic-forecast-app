import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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
# 🧠 ВЫБОР КОЛОНОК
# =========================
st.subheader("🧠 Выберите колонки для модели")

columns = df.columns.tolist()

target_col = st.selectbox("🎯 Целевая переменная", columns)

feature_cols = st.multiselect(
    "📌 Признаки",
    [col for col in columns if col != target_col]
)

if len(feature_cols) == 0:
    st.warning("Выберите хотя бы 1 признак")
    st.stop()

# =========================
# 🧹 ПОДГОТОВКА ДАННЫХ
# =========================
df = df[feature_cols + [target_col]].dropna()

# 🔥 обработка категориальных данных
df = pd.get_dummies(df)

# =========================
# 📦 X и y
# =========================
X = df.drop(target_col, axis=1)
y = df[target_col]

# =========================
# ✂️ TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🤖 МОДЕЛЬ
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 📊 ОЦЕНКА МОДЕЛИ
# =========================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.subheader("📊 Метрики модели")
st.write(f"📉 MAE: {mae:.2f}")
st.write(f"📈 R²: {r2:.2f}")

# =========================
# 📈 ГРАФИК
# =========================
st.subheader("📈 Целевая переменная")
st.line_chart(df[target_col].tail(200))

# =========================
# ⚙️ ВВОД ДЛЯ ПРЕДСКАЗАНИЯ
# =========================
st.subheader("⚙️ Ввод для предсказания")

input_data = []

cols1 = st.columns(3)

for i, col in enumerate(X.columns):
    with cols1[i % 3]:
        val = st.number_input(col, value=float(df[col].mean()))
        input_data.append(val)

# =========================
# 🔮 ПРЕДСКАЗАНИЕ
# =========================
if st.button("🔮 Предсказать"):
    data = np.array([input_data])
    pred = model.predict(data)[0]

    st.success(f"📊 Прогноз: {pred:.2f}")