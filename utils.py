import pandas as pd
from collections import Counter
from typing import List, Tuple

def get_top_libraries(data: pd.DataFrame, split_virus: bool = True) -> List[str]:
    """
    Возвращает список из 10 наиболее часто встречающихся библиотек среди вирусных и не вирусных файлов.

    Parameters
    ----------
    data : pd.DataFrame: Датафрейм
    split_virus : bool, optional: Флаг, указывающий на необходимость разделения библиотек на вирусные и не вирусные.

    Returns
    -------
    List[str]: Список из 10 наиболее часто встречающихся библиотек
    """
    if split_virus: # Блок if-else, так как в тестовом наборе отсутствует столбец is_virus
        virus_libs = data[data['is_virus'] == 1]['libs'].sum()
        non_virus_libs = data[data['is_virus'] == 0]['libs'].sum()
    else:
        libs = data['libs'].sum()
        virus_libs = libs
        non_virus_libs = []

    virus_libs_counter = Counter(virus_libs)
    non_virus_libs_counter = Counter(non_virus_libs)

    most_common_virus_libs = virus_libs_counter.most_common(10)
    most_common_non_virus_libs = non_virus_libs_counter.most_common(10)

    top_libs = [lib[0] for lib in most_common_virus_libs + most_common_non_virus_libs]

    return top_libs

def prepare_dataset(data: pd.DataFrame, split_virus: bool = True) -> pd.DataFrame:
    """
    Подготавливает датафрейм для анализа, преобразуя список библиотек в признаки и добавляя дополнительные признаки.
    Создает бинарные признаки для топ библиотек, общее количество библиотек и удаляет ненужные колонки.

    Parameters
    ----------
    data : pd.DataFrame
        Исходный датафрейм.
    split_virus : bool, optional
        Если True, функция учитывает разделение на вирусные и не вирусные библиотеки при определении топ библиотек.
        По умолчанию True. Если False, топ библиотек определяется без разделения на вирусные и не вирусные.

    Returns
    -------
    pd.DataFrame
        Преобразованный датафрейм с добавленными бинарными признаками для топ библиотек и общим количеством библиотек.
        Ненужные колонки ('filename', 'libs') удаляются.
    """
    prepared_data = data.copy(deep=True)

    prepared_data['libs'] = prepared_data['libs'].str.split(',') # Преобразуем строку с библиотеками в список
    prepared_data['total_libs'] = prepared_data['libs'].apply(len) # Создаем новый признак с количеством используемых библиотек
    
    try: # Блок try-except, так как в тестовом наборе отсутствует столбец filename
        prepared_data = prepared_data.drop(['filename'], axis=1) 
    except:
        pass

    top_libs = get_top_libraries(prepared_data, split_virus=split_virus) # Получаем наиболее используемые библиотеки

    for lib in set(top_libs): # Создаем новые бинарные признаки с наличием библиотеки
        prepared_data[f"lib_{lib}"] = prepared_data['libs'].apply(lambda x: 1 if lib in x else 0)

    prepared_data = prepared_data.drop(['libs'], axis=1) # Удаляем ненужный признак

    return prepared_data

def add_missing_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Дополняет тестовый датафрейм отсутствующими признаками из обучающего датафрейма.
    Эта функция проходит по всем признакам обучающего датафрейма и добавляет отсутствующие признаки в тестовый датафрейм,
    инициализируя их нулевыми значениями. Это обеспечивает совместимость признакового пространства между обучающим и тестовым
    датафреймами, что необходимо для корректного применения обученной модели.

    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающий датафрейм, содержащий полный набор признаков.
    test_df : pd.DataFrame
        Тестовый датафрейм, который может не содержать часть признаков из обучающего датафрейма.

    Returns
    -------
    pd.DataFrame
        Модифицированный тестовый датафрейм с добавленными отсутствующими признаками, инициализированными нулевыми значениями.
    """
    for feature in train_df.columns:
        if feature not in test_df.columns:
            test_df[feature] = 0
    return test_df

def load_data() -> Tuple[pd.DataFrame]:
    """
    Загружает тренировочные, валидационные и тестовые данные из соответствующих TSV-файлов.

    Функция читает данные из файлов 'train.tsv', 'val.tsv' и 'test.tsv', предполагая, что каждый файл
    содержит данные в формате TSV (табулированные значения). Возвращаемые датафреймы содержат данные
    из этих файлов без какой-либо дополнительной обработки.

    Returns
    -------
    Tuple[pd.DataFrame]
        Кортеж из трех объектов pd.DataFrame: тренировочного датафрейма, валидационного датафрейма и тестового датафрейма,
        загруженных из соответствующих файлов.
    """
    train = pd.read_csv('data/train.tsv', sep='\t')
    val = pd.read_csv('data/val.tsv', sep='\t')
    test = pd.read_csv('data/test.tsv', sep='\t')
    
    return train, val, test