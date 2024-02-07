import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from joblib import dump
from utils import prepare_dataset, load_data

# Загрузка данных
train, val, test = load_data()

train_prepared = prepare_dataset(train)

# Экспорт подготовленных данных для предсказания
train_prepared.to_csv('data/train_prepared.csv', index=False)
prepare_dataset(val).to_csv('data/val_prepared.csv', index=False)
prepare_dataset(test, split_virus=False).to_csv('data/test_prepared.csv', index=False)

# Подготовка данных для обучения
X_train = train_prepared.drop(['is_virus'], axis=1)
y_train = train_prepared['is_virus']

# Подготовка сетки параметров для оптимизации
params = {
    'n_estimators': [10, 25, 50, 100],
    'max_depth': [3, 8, 13],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None] 
}

# Инициализация модели
model = RandomForestClassifier(random_state=42)

# Поиск наилучшей комбинации гиперпараметров
grid = GridSearchCV(estimator=model, param_grid=params, scoring=make_scorer(f1_score), n_jobs=1, cv=3, verbose=2)

# Обучение модели
grid.fit(X_train, y_train)

# Сохранение оптимальной модели
model_optimized = grid.best_estimator_

# Сохранение модели в файл
dump(model_optimized, 'model/model.jolib')


