import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Диагностика сердца",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load('model.joblib')
    except FileNotFoundError:
        st.error("Файл модели model.joblib не найден")
        st.stop()

model = load_model()

st.title("Диагностика заболевания сердца")
st.markdown("Введите данные пациента для оценки риска сердечного заболевания.")
st.warning("Данное приложение предназначено только для образовательных целей и не является медицинским диагнозом.")
st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Данные пациента")

    age = st.number_input("Возраст", min_value=29, max_value=77, value=55, step=1)
    sex = st.selectbox("Пол", ['Женский', 'Мужской'])
    chest_pain_type = st.selectbox("Тип боли в груди", [
        'Атипичная стенокардия', 'Бессимптомно', 'Неангинозная боль', 'Типичная стенокардия'
    ])
    resting_bp = st.number_input("Давление в покое (мм рт.ст.)", min_value=94, max_value=200, value=130, step=5)
    cholesterol = st.number_input("Холестерин (мг/дл)", min_value=126, max_value=564, value=245, step=5)
    fasting_blood_sugar = st.selectbox("Сахар натощак", ['Выше 120 мг/дл', 'Норма'])

with col_right:
    st.subheader("Клинические показатели")

    rest_ecg = st.selectbox("ЭКГ в покое", ['Аномалия ST-T', 'Гипертрофия ЛЖ', 'Норма'])
    max_heart_rate = st.number_input("Макс. ЧСС при нагрузке", min_value=71, max_value=202, value=150, step=5)
    exercise_angina = st.selectbox("Стенокардия при нагрузке", ['Да', 'Нет'])
    oldpeak = st.slider("Депрессия ST (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    slope = st.selectbox("Наклон сегмента ST", ['Восходящий', 'Нисходящий', 'Плоский'])
    num_major_vessels = st.number_input("Крупные сосуды (флюороскопия)", min_value=0, max_value=4, value=0, step=1)
    thalassemia = st.selectbox("Талассемия", [
        'Неизвестно', 'Нормально', 'Обратимый дефект', 'Фиксированный дефект'
    ])

st.divider()

if st.button("Оценить риск", type="primary", use_container_width=True):
    try:
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain_type,
            'resting_bp': resting_bp,
            'cholesterol': cholesterol,
            'fasting_blood_sugar': fasting_blood_sugar,
            'rest_ecg': rest_ecg,
            'max_heart_rate': max_heart_rate,
            'exercise_angina': exercise_angina,
            'oldpeak': oldpeak,
            'slope': slope,
            'num_major_vessels': num_major_vessels,
            'thalassemia': thalassemia,
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("Высокий риск заболевания сердца")
        else:
            st.success("Низкий риск заболевания сердца")

        m1, m2, m3 = st.columns(3)
        m1.metric("Вероятность заболевания", f"{probability:.1%}")
        m2.metric("Давление", f"{resting_bp} мм рт.ст.")
        m3.metric("Холестерин", f"{cholesterol} мг/дл")

        st.subheader("Шкала риска")
        st.progress(float(min(probability, 1.0)))

        if probability > 0.7:
            st.error("Высокий риск — рекомендуется консультация кардиолога")
        elif probability > 0.4:
            st.warning("Умеренный риск — рекомендуется дополнительное обследование")
        else:
            st.success("Низкий риск — продолжайте наблюдение")

        warnings_list = []
        if age > 55:
            warnings_list.append("Возраст старше 55 лет")
        if cholesterol > 240:
            warnings_list.append("Повышенный холестерин (> 240 мг/дл)")
        if resting_bp > 140:
            warnings_list.append("Повышенное давление (> 140 мм рт.ст.)")
        if exercise_angina == 'Да':
            warnings_list.append("Стенокардия при нагрузке")
        if num_major_vessels > 0:
            warnings_list.append(f"Сужение {num_major_vessels} крупных сосудов")

        if warnings_list:
            with st.expander("Выявленные факторы риска"):
                for w in warnings_list:
                    st.write(f"• {w}")

        with st.expander("Введённые данные пациента"):
            display_df = input_data.T.reset_index()
            display_df.columns = ["Параметр", "Значение"]
            display_df["Значение"] = display_df["Значение"].astype(str)
            st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")