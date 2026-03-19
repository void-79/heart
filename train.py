import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Heart Disease Dataset (UCI Cleveland)
# Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# UCI: https://archive.ics.uci.edu/ml/datasets/heart+disease
url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
df = pd.read_csv(url)

print("=" * 55)
print("Предсказание заболевания сердца")
print("=" * 55)
print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nПропуски:\n{df.isnull().sum().to_string()}")

# Переименуем колонки
df.columns = [
    'age', 'sex', 'chest_pain_type', 'resting_bp',
    'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
    'max_heart_rate', 'exercise_angina', 'oldpeak',
    'slope', 'num_major_vessels', 'thalassemia', 'target'
]

# Kaggle-версия имеет инвертированный target:
# target=1 = здоров, target=0 = болен
# После инверсии: target=1 = болезнь, target=0 = здоров
df['target'] = 1 - df['target']

# Маппинги категориальных признаков (верифицированы по UCI)
df['sex'] = df['sex'].map({1: 'Мужской', 0: 'Женский'})
df['chest_pain_type'] = df['chest_pain_type'].map({
    0: 'Бессимптомно',
    1: 'Атипичная стенокардия',
    2: 'Неангинозная боль',
    3: 'Типичная стенокардия'
})
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
    1: 'Выше 120 мг/дл',
    0: 'Норма'
})
df['rest_ecg'] = df['rest_ecg'].map({
    0: 'Норма',
    1: 'Гипертрофия ЛЖ',
    2: 'Аномалия ST-T'
})
df['exercise_angina'] = df['exercise_angina'].map({
    1: 'Да',
    0: 'Нет'
})
df['slope'] = df['slope'].map({
    0: 'Нисходящий',
    1: 'Плоский',
    2: 'Восходящий'
})
df['thalassemia'] = df['thalassemia'].map({
    0: 'Неизвестно',
    1: 'Фиксированный дефект',
    2: 'Нормально',
    3: 'Обратимый дефект'
})

print(f"\nРаспределение target (после инверсии):")
print(df['target'].value_counts().to_string())
print(f"1 = болезнь: {df['target'].mean():.1%}")

X = df.drop('target', axis=1)
y = df['target']

num_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate',
                'oldpeak', 'num_major_vessels']
cat_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar',
                'rest_ecg', 'exercise_angina', 'slope', 'thalassemia']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Проверка переобучения
y_train_pred = pipeline.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 55}")
print("Результаты")
print("=" * 55)
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy:  {test_acc:.3f}")
print(f"ROC-AUC:        {roc_auc_score(y_test, y_proba):.3f}")
if train_acc - test_acc > 0.15:
    print("ВНИМАНИЕ: Возможное переобучение")
print(f"\n{classification_report(y_test, y_pred, target_names=['Здоров', 'Болезнь'])}")

joblib.dump(pipeline, 'model.joblib')
print(f"Модель сохранена: model.joblib ({os.path.getsize('model.joblib') / 1024:.0f} KB)")

# Smoke-test
print(f"\n{'=' * 55}")
print("SMOKE-TEST")
print("=" * 55)

loaded = joblib.load('model.joblib')

healthy = pd.DataFrame([{
    'age': 30, 'sex': 'Женский', 'chest_pain_type': 'Неангинозная боль',
    'resting_bp': 110, 'cholesterol': 180, 'fasting_blood_sugar': 'Норма',
    'rest_ecg': 'Норма', 'max_heart_rate': 180, 'exercise_angina': 'Нет',
    'oldpeak': 0.0, 'slope': 'Восходящий', 'num_major_vessels': 0,
    'thalassemia': 'Нормально'
}])

prob_h = loaded.predict_proba(healthy)[0][1]
print(f"\nЗдоровый пациент (30 лет, женщина, нормальные показатели):")
print(f"Вероятность болезни: {prob_h:.1%}")
print("OK" if prob_h < 0.3 else "ПРОВЕРИТЬ")

sick = pd.DataFrame([{
    'age': 65, 'sex': 'Мужской', 'chest_pain_type': 'Бессимптомно',
    'resting_bp': 170, 'cholesterol': 350, 'fasting_blood_sugar': 'Выше 120 мг/дл',
    'rest_ecg': 'Гипертрофия ЛЖ', 'max_heart_rate': 100, 'exercise_angina': 'Да',
    'oldpeak': 3.0, 'slope': 'Нисходящий', 'num_major_vessels': 3,
    'thalassemia': 'Обратимый дефект'
}])

prob_s = loaded.predict_proba(sick)[0][1]
print(f"\nБольной пациент (65 лет, мужчина, факторы риска):")
print(f"Вероятность болезни: {prob_s:.1%}")
print("OK" if prob_s > 0.7 else "ПРОВЕРИТЬ")

# Категории OneHotEncoder
print(f"\n{'=' * 55}")
print("Категории OneHotEncoder:")
print("=" * 55)
cat_encoder = loaded.named_steps['preprocessor'].named_transformers_['cat']
for i, feature in enumerate(cat_features):
    print(f"{feature}: {list(cat_encoder.categories_[i])}")