import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import load

# Загрузка подготовленных данных
val_prepared = pd.read_csv('data/val_prepared.csv')

# Загрузка модели из файла
model = load('model/model.jolib')

# Подготовка данных для предсказания
X_val = val_prepared.drop(['is_virus'], axis=1)
y_val = val_prepared['is_virus']

# Предсказание
prediction = model.predict(X_val)

# Получение необходимых метрик качества
tn, fp, fn, tp = confusion_matrix(y_val, prediction).ravel()
accuracy = accuracy_score(y_val, prediction)
precision = precision_score(y_val, prediction)
recall = recall_score(y_val, prediction)
f1 = f1_score(y_val, prediction)

# Запись в файл
with open('report/validation.txt', 'w') as file:
    file.write(f"True positive: {tp}\n")
    file.write(f"False positive: {fp}\n")
    file.write(f"False negative: {fn}\n")
    file.write(f"True negative: {tn}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1: {f1:.4f}\n")

