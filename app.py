import streamlit as st
import pandas as pd
import joblib

# Загружаем модель, препроцессор и label_encoders, сохранённые после обучения
catboost_model = joblib.load('catboost_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title("Прогноз сердечных заболеваний")

# Ввод данных пользователя
age = st.number_input("Возраст", min_value=1, max_value=120, value=50)
sex = st.selectbox("Пол", options=['M', 'F'])
chest_pain = st.selectbox("Тип боли в груди", options=['TA', 'ATA', 'NAP', 'ASY'])
resting_bp = st.number_input("Давление в покое (RestingBP)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Холестерин", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Уровень сахара натощак (FastingBS)", options=[0, 1])
resting_ecg = st.selectbox("ЭКГ в покое", options=['Normal', 'ST', 'LVH'])
max_hr = st.number_input("Максимальный ЧСС (MaxHR)", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Стенокардия при нагрузке", options=['Y', 'N'])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
st_slope = st.selectbox("ST Slope", options=['Up', 'Flat', 'Down'])

# Кодируем бинарный пол вручную
sex_encoded = 0 if sex == 'M' else 1

# Кодируем категориальные признаки с помощью сохранённых LabelEncoder
chest_pain_encoded = label_encoders['ChestPainType'].transform([chest_pain])[0]
resting_ecg_encoded = label_encoders['RestingECG'].transform([resting_ecg])[0]
exercise_angina_encoded = label_encoders['ExerciseAngina'].transform([exercise_angina])[0]
st_slope_encoded = label_encoders['ST_Slope'].transform([st_slope])[0]

# Собираем все признаки в словарь
input_data = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex': sex_encoded,
    'ChestPainType': chest_pain_encoded,
    'RestingECG': resting_ecg_encoded,
    'ExerciseAngina': exercise_angina_encoded,
    'ST_Slope': st_slope_encoded
}

# Превращаем в DataFrame
X_user = pd.DataFrame([input_data])

# Применяем препроцессор
X_processed = preprocessor.transform(X_user)

# Кнопка для предсказания
if st.button("Предсказать риск сердечного заболевания"):
    prediction = catboost_model.predict(X_processed)
    proba = catboost_model.predict_proba(X_processed)[0,1]

    if prediction[0] == 1:
        st.error(f"Риск сердечного заболевания высокий (вероятность: {proba:.2f})")
    else:
        st.success(f"Риск сердечного заболевания низкий (вероятность: {proba:.2f})")

