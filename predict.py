import pandas as pd
import shap
from joblib import load
from utils import add_missing_features

# Загрузка подготовленных данных
test_prepared = pd.read_csv('data/test_prepared.csv')
train_prepared = pd.read_csv('data/train_prepared.csv')

# Загрузка модели
model = load('model/model.jolib')

# Подготовка данных для предсказаний
train_prepared = train_prepared.drop('is_virus', axis=1)
X_test = add_missing_features(train_prepared, test_prepared)[train_prepared.columns]

# Предсказания модели
predictions = model.predict(X_test)

# Расчет SHAP значений для объяснения предсказаний на основе значимости признаков
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)[1]

# Сохранение предсказаний в файл
with open('report/prediction.txt', 'w') as file:
    file.write("prediction\n") 
    for pred in predictions:
        file.write(f"{pred}\n")

# Сохранение объяснений в файл
with open('report/explain.txt', 'w') as f:
    for i, pred in enumerate(predictions):
        if pred == 1:
            top_feature_idx = shap_values[i].argmax()
            top_feature_name = X_test.columns[top_feature_idx]
            top_feature_value = shap_values[i][top_feature_idx]
            reason = f"Файл зловредный, так как признак {top_feature_name} стат. значимо влияет на значение класса (SHAP = ({top_feature_value:.2f}))"
            f.write(f"{i}: {reason}\n")
        else:
            f.write(f"{i}: \n")