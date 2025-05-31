### OVERALL CODE
#либы
#база
#!pip install streamlit
#!pip install ydata-profiling

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
#from ydata_profiling import ProfileReport
import inspect
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
st.set_page_config(page_title="HSE | ТИИПА", layout="wide")

st.title("© HSE | ТИИПА")
st.markdown("**Fast-touch classification-analytical tool**")

st.markdown("---")

# Информация о команде
with st.expander("ℹО команде разработчиков"):
    st.markdown("""
    **Команда проекта:**
    - Константин Ильященко — Team Leader
    - Вадим Казаков — Master of Machine Learning
    - Андрей Ширшов — Business Developer
    - Татьяна Черных — Designer
    """)

# Инструкция по использованию
with st.expander("Инструкция по использованию сервиса", expanded=True):
    st.markdown("""
    1. *Выберите нужную тему* анализа
    2. *Подготовьте данные* с переменной для предсказания и переменными для анализа
    3. *Загрузите данные* и выберите подходящие переменные
    4. *Запустите анализ* — выберите модели машинного обучения и доверьтесь нам. *Дайте алгоритму время для обработки.*
    5. *Изучите результаты* — метрики предсказания и визуализации.
    6. *Выберите* лучшую модель и *примените* её на совершенно новых данных.
    """)

st.markdown("---")

#st.title("Загрузи файл сюда, друг")
#uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])
st.title("Данные для анализа")
uploaded_file = st.file_uploader("Выберите файл (CSV или Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Определяем тип файла по расширению
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.csv'):
        # Настройки для CSV файлов
        sep_sign = st.selectbox(
            "Выберите разделитель",
            (";", ",", " ", "|"), index=0)

        decimal_sign = st.selectbox(
            "Выберите отделитель дробной части",
            (".", ","), index=1)

        df = pd.read_csv(uploaded_file, sep=sep_sign, decimal=decimal_sign)

    elif file_name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)

    st.write("Загруженный набор данных:")
    st.dataframe(df.head())
    backup_df = df.copy()
 
#if uploaded_file is not None:
  #seps = [";", ","]
  #decimals = [".", ","]
  #sep_sign = ";"
  #sep_sign = st.selectbox(
   # "Выберите разделитель",
   # (";", ",", " ", "/"))
  #decimal_sign = ","
  #decimal_sign = st.selectbox(
    #"Выберите отделитель дробной части",
    #(".", ","))
  #df = pd.read_csv(uploaded_file, sep = sep_sign, decimal = decimal_sign)
  #st.write("Твой датасет:")
  #st.dataframe(df.head())
  #backup_df = df.copy()


#чтение данных
#def read_data(df):
  #return df
#def handle_missing_values(df, option):

    #df_processed = df.copy()
    
    #if option == "Дропнуть":
        #df_processed = df_processed.dropna()
    #elif option == "Заменить на ноль":
        #for col in df_processed.columns:
            #if pd.api.types.is_numeric_dtype(df_processed[col]):
                #df_processed[col] = df_processed[col].fillna(0)
            #else:
                #df_processed[col] = df_processed[col].fillna("0")
    #elif option == "Заменить на медиану":
        #for col in df_processed.columns:
            #if pd.api.types.is_numeric_dtype(df_processed[col]):
                #median_val = df_processed[col].median()
                #df_processed[col] = df_processed[col].fillna(median_val)
            #else:
                #df_processed[col] = df_processed[col].fillna("0")
    
    #return df_processed
# Основной код Streamlit
def handle_missing_values(df, option, categorical_method="mode"):
    df_processed = df.copy()

    # Для каждого столбца применяем соответствующую обработку
    for col in df_processed.columns:
        # Для числовых данных
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if option == "Удалить строки с пропусками":
                df_processed = df_processed.dropna(subset=[col])
            elif option == "Заменить на ноль":
                df_processed[col] = df_processed[col].fillna(0)
            elif option == "Заменить на медиану":
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            elif option == "Заменить на среднее":
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif option == "Интерполяция":
                df_processed[col] = df_processed[col].interpolate()

        # Для категориальных данных
        else:
            if option == "Удалить строки с пропусками":
                df_processed = df_processed.dropna(subset=[col])
            elif option == "Заменить на моду":
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            elif option == "Создать новую категорию 'Unknown'":
                df_processed[col] = df_processed[col].fillna("Unknown")
            elif option == "Экстраполяция (последнее значение)":
                df_processed[col] = df_processed[col].fillna(method='fill')

    return df_processed
st.title("Обработка пропущенных значений в датафрейме")

# Предполагаем, что датафрейм уже загружен в переменную df
if 'df' not in globals():
    st.warning("Датафрейм не найден в памяти. Загрузите данные.")
    st.stop()
# Информация о пропусках
st.subheader("Статистика пропущенных значений")
missing_stats = df.isna().sum()
missing_percent = (missing_stats / len(df)) * 100
missing_df = pd.DataFrame({
    "Колонка": missing_stats.index,
    "Количество пропусков": missing_stats.values,
    "Процент пропусков": missing_percent.values.round(2)
})
st.dataframe(missing_df)

# Выбор метода обработки
st.markdown("---")
st.subheader("Настройки обработки пропусков")

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Метод обработки пропусков:",
        options=[
            "Удалить строки с пропусками",
            "Заменить на ноль",
            "Заменить на медиану",
            "Заменить на среднее",
            "Заменить на моду",
            "Создать новую категорию 'Unknown'",
            "Интерполяция",
            "Экстраполяция (последнее значение)"
        ],
        index=2,
        help="Выберите метод обработки пропущенных значений"
    )

with col2:
    show_details = st.checkbox("Показать пояснения методов", value=True)

if show_details:
    with st.expander("📚 Пояснения методов обработки"):
        st.markdown("""
        - **Удалить строки с пропусками** - полное удаление строк, содержащих пропуски
        - **Заменить на ноль** - замена всех пропусков на 0 (для числовых данных)
        - **Заменить на медиану** - замена пропусков медианным значением (устойчиво к выбросам)
        - **Заменить на среднее** - замена пропусков средним значением (может искажаться выбросами)
        - **Заменить на моду** - замена пропусков наиболее частым значением (для категориальных данных)
        - **Создать новую категорию 'Unknown'** - для категориальных данных, сохраняет информацию о пропуске
        - **Интерполяция** - линейная интерполяция значений (для временных рядов)
        - **Экстраполяция** - заполнение последним известным значением (метод 'forward fill')
        """)

# Обработка данных
if st.checkbox("Обработать пропущенные значения"): #, type="primary"):
    with st.spinner("Обработка данных..."):
        df = handle_missing_values(df, method)
        st.success("Обработка завершена!")

        # Вывод результатов
        st.markdown("---")
        st.subheader("Результаты обработки")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Исходное количество строк", len(backup_df))
        with col2:
            st.metric("Количество строк после обработки", len(df))

        st.dataframe(df.head(10))
# Выпадающий список для выбора метода обработки пропусков
#missing_values_option = st.selectbox(
    #"Что делать с пропусками?",
    #options=["Заменить на ноль", "Дропнуть", "Заменить на медиану"],
    #index=0  # По умолчанию выбран "Заменить на ноль"
#)

# Обработка датафрейма
#df = handle_missing_values(df, missing_values_option)

# Вывод информации о результате
#st.subheader("Результат обработки")
#st.write(f"Выбранный метод: {missing_values_option}")
#st.write(f"Исходное количество строк: {len(backup_df)}")
#st.write(f"Количество строк после обработки: {len(df)}")
#df = st.session_state.df
# Показать обработанный датафрейм
st.dataframe(df)

#Посмотреть данные перед работой
st.title("Анализ данных")

st.write(df.describe())

# Анализ данных
st.title("Визуальный анализ данных")
st.markdown("---")

with st.expander("Основные статистические показатели данных", expanded=True):
    st.subheader("Описательная статистика")

    # Улучшенное отображение describe()
    desc = df.describe().T
    desc['missing'] = df.isna().sum()
    desc['missing_percent'] = (desc['missing'] / len(df)).round(2)
    desc['dtype'] = df.dtypes
    desc['unique'] = df.nunique()

    # Форматируем вывод
    st.dataframe(
        desc.style.format({
            'mean': '{:.2f}',
            'std': '{:.2f}',
            'min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'max': '{:.2f}',
            'missing_percent': '{:.0%}'
        }),
        use_container_width=True
    )

st.markdown("---")
st.header("Анализ распределения")

# Выбор типа графика
chart_type = st.radio(
    "Тип визуализации распределения:",
    ["Гистограмма", "Ящик с усами (Boxplot)", "Плотность распределения"],
    horizontal=True
)

col1, col2 = st.columns([3, 1])
with col1:
    selected_col = st.selectbox(
        "Выберите переменную для анализа:",
        options=df.select_dtypes(include=['number']).columns,
        index=0
    )

with col2:
    if chart_type == "Гистограмма":
        bins = st.slider(
            "Количество интервалов:",
            min_value=5,
            max_value=50,
            value=int(np.sqrt(len(df)) if len(df) > 0 else 20)
        )

# Отображение выбранного графика
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 5))

if chart_type == "Гистограмма":
    sns.histplot(df[selected_col], bins=bins, kde=True, ax=ax)
    ax.set_title(f'Распределение {selected_col}')
    ax.set_xlabel(selected_col)
    ax.set_ylabel('Частота')
elif chart_type == "Ящик с усами (Boxplot)":
    sns.boxplot(x=df[selected_col], ax=ax)
    ax.set_title(f'Boxplot для {selected_col}')
elif chart_type == "Плотность распределения":
    sns.kdeplot(df[selected_col], ax=ax, fill=True)
    ax.set_title(f'Плотность распределения {selected_col}')
    ax.set_xlabel(selected_col)
    ax.set_ylabel('Плотность')

st.pyplot(fig)

st.markdown("---")
    # 1. Гистограмма для выбранной колонки
#st.subheader("Гистограмма")
#selected_col = st.selectbox(
#        "Выберите колонку для гистограммы",
#        options=df.columns,
#        index=2
#    )
#hist_values = np.histogram(df[selected_col], bins=int(round( len(df[selected_col])**0.5 ,0)))
    
    # Преобразуем в DataFrame для st.bar_chart
#hist_df = pd.DataFrame({
        #'bin_left': hist_values[1][:-1],
        #'bin_right': hist_values[1][1:],
        #'count': hist_values[0]
    #})
#hist_df['bin'] = hist_df.apply(lambda x: f"{x['bin_left']:.2f}-{x['bin_right']:.2f}", axis=1)
    
# Отображаем барчарт
#st.bar_chart(hist_df.set_index('bin')['count'])
    
    # 2. Scatter plot с настройками
st.subheader("Scatter plot")
col1, col2, col3, col4 = st.columns(4)
    
with col1:
  x_axis = st.selectbox(
            "Ось X",
            options=df.columns,
            index=0
        )
with col2:
  y_axis = st.selectbox(
            "Ось Y",
            options=df.columns,
            index=1 if len(df.columns) > 1 else 0
        )
    
with col3:
  color_col = st.selectbox(
            "Цвет",
            options=["None"] + list(df.columns),
            index=0
        )
with col4:
  size_col = st.selectbox(
            "Размер",
            options=["None"] + list(df.columns),
            index=0
        )    
if (color_col == "None") & (size_col == "None"):
    st.scatter_chart(df, x=x_axis, y=y_axis)
elif (size_col == "None") & (color_col != "None"):
    st.scatter_chart(df, x=x_axis, y=y_axis, color=color_col)
elif (size_col != "None") & (color_col == "None"):
    st.scatter_chart(df, x=x_axis, y=y_axis, size = size_col)
elif (size_col != "None") & (color_col != "None"):
    st.scatter_chart(df, x=x_axis, y=y_axis, color = color_col, size = size_col)

#st.subheader("Подготовка данных для моделирования")
#    
#    # 1. Выбор таргетной переменной (y)
#target_col = st.selectbox(
#        "Выберите таргетную переменную (y)",
#        options=df.columns,
#        index=0,  # Дефолтное значение - первая колонка
#        key="target_select"
#    )
    
#    # 2. Выбор признаков (X) - исключаем таргет из возможных признаков
#available_features = [col for col in df.columns if col != target_col]
    
#selected_features = st.multiselect(
#        "Выберите признаки для модели (X)",
#        options=available_features,
#        default=available_features,  # По умолчанию выбираем все доступные признаки
#        key="features_select"
#    )
#    
#    # Проверка, что выбраны хотя бы один признак
#if not selected_features:
#  st.error("Пожалуйста, выберите хотя бы один признак для модели")
#      
#if selected_features:    
#    # Формируем X и y
#  y = df[target_col]
#  X = df[selected_features]
#    
#    # 3. Разбиение на train/test
#  X_train, X_test, y_train, y_test = train_test_split(
#        X, y, 
#        test_size=0.2, 
#        random_state=42
#    )
# Подготовка данных к предсказанию
st.subheader("Подготовка данных для моделирования")
st.markdown("---")

# 1. Выбор таргетной переменной (y)
target_col = st.selectbox(
    "Выберите таргетную переменную (y)",
    options=df.columns,
    index=0,
    key="target_select"
)

# 2. Выбор ID переменных (для идентификации наблюдений)
id_cols = st.multiselect(
    "Выберите переменные-идентификаторы (не будут использоваться в модели)",
    options=[col for col in df.columns if col != target_col],
    key="id_cols"
)

# 3. Выбор признаков (X)
available_features = [col for col in df.columns if col != target_col and col not in id_cols]

selected_features = st.multiselect(
    "Выберите признаки для модели (X)",
    options=available_features,
    default=available_features,
    key="features_select"
)

if not selected_features:
    st.error("Пожалуйста, выберите хотя бы один признак для модели")
    st.stop()

# 4. Настройки предобработки данных
with st.expander("Настройки предобработки данных", expanded=True):
    st.subheader("Нормализация данных")
    numeric_cols = [col for col in selected_features if pd.api.types.is_numeric_dtype(df[col])]
    norm_cols = st.multiselect(
        "Выберите признаки для нормализации (StandardScaler)",
        options=numeric_cols,
        help="Применяет стандартную нормализацию (z-score)"
    )

    st.subheader("Логарифмическое преобразование")
    log_cols = st.multiselect(
        "Выберите признаки для логарифмирования",
        options=numeric_cols,
        help="Применяет log(x+1) преобразование для правосторонне-скошенных данных"
    )

    st.subheader("Обработка категориальных данных")
    categorical_cols = [col for col in selected_features if not pd.api.types.is_numeric_dtype(df[col])]
    dummy_cols = st.multiselect(
        "Выберите категориальные признаки для one-hot кодирования",
        options=categorical_cols,
        help="Создаст dummy-переменные (n-1) для выбранных категориальных признаков"
    )

    st.subheader("Балансировка классов")
    if pd.api.types.is_numeric_dtype(df[target_col]):
        st.warning("Балансировка классов доступна только для категориального таргета")
    else:
        balance_method = st.selectbox(
            "Метод балансировки классов",
            options=["Нет", "Random Oversampling", "SMOTE", "Random Undersampling"],
            index=0,
            help="Выберите метод для балансировки классов в обучающей выборке"
        )

# 5. Параметры разбиения
test_size = st.slider(
    "Размер тестовой выборки (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=5
)

random_state = st.number_input(
    "Random state для воспроизводимости",
    min_value=0,
    value=42
)
    
    # Выводим информацию о разбиении
#  st.success("Данные успешно подготовлены!")
#  st.write(f"Выбран таргет: {target_col}")
#  st.write(f"Выбрано признаков: {len(selected_features)}")
#  st.write(f"Размер обучающей выборки: {X_train.shape[0]}")
#  st.write(f"Размер тестовой выборки: {X_test.shape[0]}")
def preprocess_data(data, target_col, id_cols, features, norm_cols, log_cols, dummy_cols,
                   balance_method, test_size, random_state):

    # Выделяем признаки и таргет
    X = data[features]
    y = data[target_col]
    ids = data[id_cols] if id_cols else None

    # Разбиваем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size/100,
        random_state=random_state,
        stratify=y if not pd.api.types.is_numeric_dtype(y) else None
    )

    # Сохраняем ID для train и test
    if ids is not None:
        train_ids = ids.loc[X_train.index]
        test_ids = ids.loc[X_test.index]
    else:
        train_ids, test_ids = None, None

    # 1. Логарифмирование
    for col in log_cols:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    # 2. One-hot кодирование
    if dummy_cols:
        encoder = OneHotEncoder(drop='first', sparse=False)
        train_encoded = encoder.fit_transform(X_train[dummy_cols])
        test_encoded = encoder.transform(X_test[dummy_cols])

        encoded_cols = encoder.get_feature_names_out(dummy_cols)
        train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)

        X_train = X_train.drop(dummy_cols, axis=1).join(train_encoded_df)
        X_test = X_test.drop(dummy_cols, axis=1).join(test_encoded_df)

    # 3. Нормализация
    if norm_cols:
        scaler = StandardScaler()
        X_train[norm_cols] = scaler.fit_transform(X_train[norm_cols])
        X_test[norm_cols] = scaler.transform(X_test[norm_cols])

    # 4. Балансировка классов
    if balance_method != "Нет" and not pd.api.types.is_numeric_dtype(y_train):
        if balance_method == "Random Oversampling":
            sampler = RandomOverSampler(random_state=random_state)
        elif balance_method == "SMOTE":
            sampler = SMOTE(random_state=random_state)
        elif balance_method == "Random Undersampling":
            sampler = RandomUnderSampler(random_state=random_state)

        X_train, y_train = sampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, train_ids, test_ids

# Кнопка для запуска обработки
if st.checkbox("Подготовить данные"):
    with st.spinner("Обработка данных..."):
        # Полная обработка данных с разбиением
        X_train, X_test, y_train, y_test, train_ids, test_ids = preprocess_data(
            data=df,
            target_col=target_col,
            id_cols=id_cols,
            features=selected_features,
            norm_cols=norm_cols,
            log_cols=log_cols,
            dummy_cols=dummy_cols,
            balance_method=balance_method,
            test_size=test_size,
            random_state=random_state
        )

        # Сохраняем в session_state
        #st.session_state.update({
            #'X_train': X_train,
            #'X_test': X_test,
            #'y_train': y_train,
            #'y_test': y_test,
            #'train_ids': train_ids,
            #'test_ids': test_ids
        #})

        st.success("Данные успешно подготовлены!")

        # Вывод результатов
        st.markdown("---")
        st.subheader("Результаты предобработки")
        st.header("Детальный анализ")
        st.subheader(f"Анализ по классам ({target_col})")
        class_stats = df.groupby(target_col,as_index=False).mean().merge(pd.DataFrame(df[target_col].value_counts()).reset_index(), how='left')
        st.dataframe(class_stats)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Обучающая выборка", f"{len(X_train)} наблюдений")
            st.metric("Признаков после обработки", X_train.shape[1])
            if hasattr(y_train, 'value_counts'):
                st.write("Распределение классов (train):")
                st.dataframe(y_train.value_counts(normalize=True))

        with col2:
            st.metric("Тестовая выборка", f"{len(X_test)} наблюдений")
            st.metric("Размер теста", f"{test_size}%")
            if hasattr(y_test, 'value_counts'):
                st.write("Распределение классов (test):")
                st.dataframe(y_test.value_counts(normalize=True))

        # Пример данных
        st.markdown("---")
        st.subheader("Пример данных")

        tab1, tab2 = st.tabs(["Обучающая выборка", "Тестовая выборка"])
        with tab1:
            display_df = X_train.head(5).copy()
            display_df[target_col] = y_train.head(5).values
            if train_ids is not None:
                display_df = pd.concat([train_ids.head(5), display_df], axis=1)
            st.dataframe(display_df)

        with tab2:
            display_df = X_test.head(5).copy()
            display_df[target_col] = y_test.head(5).values
            if test_ids is not None:
                display_df = pd.concat([test_ids.head(5), display_df], axis=1)
            st.dataframe(display_df)

# Для использования в ML моделях
#if 'X_train' in st.session_state:
    #X_train = st.session_state.X_train
    #X_test = st.session_state.X_test
    #y_train = st.session_state.y_train
    #y_test = st.session_state.y_test
if 'X_train' not in globals():
    st.warning("Подготовьте, пожалуйста, данные")
    st.stop()
else:
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    #st.write(cm)
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(cm, annot=True, cbar=False, fmt = "d",  ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    #plt.tight_layout()
    #st.pyplot(fig)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5, str(cm[i, j]),
                    ha='center', va='center', color='white')
    
    st.pyplot(fig, clear_figure=True)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, use_cv=False, cv_folds=5):
    if use_cv:
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        st.write(f"Кросс-валидация (Accuracy): Среднее = {scores.mean():.4f}, Std = {scores.std():.4f}")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    metrics = {
        'Accuracy': [
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred)
        ],
        'Precision': [
            precision_score(y_train, y_train_pred, average='weighted'),
            precision_score(y_test, y_test_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_train, y_train_pred, average='weighted'),
            recall_score(y_test, y_test_pred, average='weighted')
        ],
        'F1': [
            f1_score(y_train, y_train_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        ]
    }

    metrics_df = pd.DataFrame(metrics, index=['Train', 'Test'])

    st.subheader(f"Матрицы ошибок для {model_name}")
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion_matrix(y_train, y_train_pred, f'Train\n{model_name}')
    with col2:
        plot_confusion_matrix(y_test, y_test_pred, f'Test\n{model_name}')

    return model, metrics_df, y_train_pred, y_test_pred  
models = {}
#logreп
def logistic_regression(X_train, X_test, y_train, y_test, use_cv=False, cv_folds=5):
  logreg = LogisticRegression(
    multi_class='multinomial',
    max_iter = 1000
)
  logreg.fit(X_train, y_train)
  return evaluate_model(logreg, X_train, X_test, y_train, y_test, "Logistic Regression", use_cv=use_cv, cv_folds=cv_folds)

#tree
def tree(X_train, X_test, y_train, y_test, max_depth_target = 10, min_samples_split_target = 10, use_cv=False, cv_folds=5):
  tree = DecisionTreeClassifier(
    max_depth = max_depth_target,             # Максимальная глубина
    min_samples_split = min_samples_split_target,     # Минимальное число образцов для разделения
)
  tree.fit(X_train, y_train)
  return evaluate_model(tree, X_train, X_test, y_train, y_test, "Decision Tree", use_cv=use_cv, cv_folds=cv_folds)

#forest
def random_forest(X_train, X_test, y_train, y_test, estimators_target = 50, max_depth_target = 10, min_samples_split_target = 10, use_cv=False, cv_folds=5):
  random_forest = RandomForestClassifier(
    n_estimators = estimators_target,  # Число деревьев
    max_features='sqrt',
    max_depth = max_depth_target,
    min_samples_split = min_samples_split_target
)
  random_forest.fit(X_train, y_train)
  return evaluate_model(random_forest, X_train, X_test, y_train, y_test, "Random Forest", use_cv=use_cv, cv_folds=cv_folds)


#xgboost
def xgboost(X_train, X_test, y_train, y_test, learning_rate=0.01, n_estimators=50, max_depth=10, use_cv=False, cv_folds=5):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Создание и обучение модели
    xgboost_model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        eval_metric='mlogloss',  # Для многоклассовой классификации
        use_label_encoder=False  # Чтобы избежать предупреждений
    )

    if use_cv:
        scores = cross_val_score(xgboost_model, X_train, y_train_encoded, cv=cv_folds, scoring='accuracy')
        st.write(f"Кросс-валидация (Accuracy): Среднее = {scores.mean():.4f}, Std = {scores.std():.4f}")

    xgboost_model.fit(X_train, y_train_encoded)

    # Предсказания
    y_train_pred = le.inverse_transform(xgboost_model.predict(X_train))
    y_test_pred = le.inverse_transform(xgboost_model.predict(X_test))

    # Расчет метрик
    metrics = {
        'Accuracy': [
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred)
        ],
        'Precision': [
            precision_score(y_train, y_train_pred, average='weighted'),
            precision_score(y_test, y_test_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_train, y_train_pred, average='weighted'),
            recall_score(y_test, y_test_pred, average='weighted')
        ],
        'F1': [
            f1_score(y_train, y_train_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        ],
    }

    metrics_df = pd.DataFrame(metrics, index=['Train', 'Test'])

    # Визуализация
    st.subheader("Матрицы ошибок для XGBoost")
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion_matrix(y_train, y_train_pred, 'Train\nXGBoost')
    with col2:
        plot_confusion_matrix(y_test, y_test_pred, 'Test\nXGBoost')

    # Важность признаков
    #st.subheader("Важность признаков")
    #fig, ax = plt.subplots(figsize=(10, 6))
    #xgb.plot_importance(xgboost_model, ax=ax)
    #st.pyplot(fig)

    # Создание DataFrame с результатами
    train_results = pd.DataFrame({
        #'index': X_train.index,
        #'Actual': y_train,
        'Predicted': y_train_pred
    })

    test_results = pd.DataFrame({
        #'index': X_test.index,
        #'Actual': y_test,
        'Predicted': y_test_pred
    })

    return xgboost_model, metrics_df, y_train_pred, y_test_pred

#Support-vector-calc
def svc(X_train, X_test, y_train, y_test, use_cv=False, cv_folds=5):
  svc = SVC()
  svc.fit(X_train, y_train)
  return evaluate_model(svc, X_train, X_test, y_train, y_test, "SVC", use_cv=use_cv, cv_folds=cv_folds)

#knn
def knn_classifier(X_train, X_test, y_train, y_test, neighbors_target = 10, use_cv=False, cv_folds=5):
  knn = KNeighborsClassifier(n_neighbors= neighbors_target)
  knn.fit(X_train, y_train)
  return evaluate_model(knn, X_train, X_test, y_train, y_test, "KNN", use_cv=use_cv, cv_folds=cv_folds)

def perceptron_classifier(X_train, X_test, y_train, y_test,
                        layers_target=2, neurons_target=50,
                        learning_rate_target=0.01, epochs_target=10, use_cv=False, cv_folds=5):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    y_train_onehot = to_categorical(y_train_encoded)
    y_test_onehot = to_categorical(y_test_encoded)

    model = Sequential()
    #model = Classifier()
    model.add(Dense(neurons_target, input_dim=X_train.shape[1], activation='relu'))

    for _ in range(layers_target ):
        model.add(Dense(neurons_target, activation='relu'))

    model.add(Dense(y_train_onehot.shape[1], activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate_target),
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train_onehot,
        epochs=epochs_target,
        batch_size=32,
        validation_data=(X_test, y_test_onehot),
        verbose=0
    )

    # Предсказания
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    # Преобразование предсказаний обратно в классы
    y_train_pred_classes = le.inverse_transform(np.argmax(y_train_pred, axis=1))
    y_test_pred_classes = le.inverse_transform(np.argmax(y_test_pred, axis=1))

    # Расчет метрик
    metrics = {
        'Accuracy': [
            accuracy_score(y_train, y_train_pred_classes),
            accuracy_score(y_test, y_test_pred_classes)
        ],
        'Precision': [
            precision_score(y_train, y_train_pred_classes, average='weighted'),
            precision_score(y_test, y_test_pred_classes, average='weighted')
        ],
        'Recall': [
            recall_score(y_train, y_train_pred_classes, average='weighted'),
            recall_score(y_test, y_test_pred_classes, average='weighted')
        ],
        'F1': [
            f1_score(y_train, y_train_pred_classes, average='weighted'),
            f1_score(y_test, y_test_pred_classes, average='weighted')
        ]
    }

    metrics_df = pd.DataFrame(metrics, index=['Train', 'Test'])

    # Визуализация
    st.subheader("Матрицы ошибок для Perceptron")
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion_matrix(y_train, y_train_pred_classes, 'Train\nPerceptron')
    with col2:
        plot_confusion_matrix(y_test, y_test_pred_classes, 'Test\nPerceptron')

    # График обучения
    st.subheader("График обучения")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.legend()

    st.pyplot(fig)

    # Создание DataFrame с результатами
    train_results = pd.DataFrame({
        #'index': X_train.index,
        #'Actual': y_train,
        'Predicted': y_train_pred_classes
    })

    test_results = pd.DataFrame({
        #'index': X_test.index,
        #'Actual': y_test,
        'Predicted': y_test_pred_classes
    })

    return model, metrics_df, y_train_pred_classes, y_test_pred_classes
# ======== Models row ============
st.subheader("Пояснения к моделям и параметрам")

show_details = st.checkbox("Показать пояснения", value=True)

if show_details:
    with st.expander("📚 Пояснения к моделям и настройкам"):
        st.markdown("""
        ### Общие пояснения к тумблерам:
        * **Создать модель [Название Модели]!**: Этот переключатель запускает обучение и оценку выбранной модели. Если он включен, модель будет создана, обучена на тренировочных данных и протестирована на тестовых данных, а результаты (метрики и матрицы ошибок) будут отображены.

        ### Пояснения к моделям:

        #### **Логистическая регрессия (Logistic Regression)**
        * **Что делает**: Логистическая регрессия — это линейный алгоритм для **классификации**. Несмотря на название "регрессия", она используется для предсказания вероятности принадлежности объекта к одному из классов, а затем на основе этой вероятности присваивается класс.
        * **За что отвечает**: Отвечает за построение линейной границы решения между классами. Хорошо подходит для задач бинарной и многоклассовой классификации, когда классы линейно разделимы или почти разделимы.
        * **Как можно настраивать (параметры по умолчанию в коде)**:
            * `multi_class='multinomial'`: Указывает, что модель будет работать с многоклассовой классификацией (если у вас более двух классов).
            * `max_iter=1000`: Максимальное количество итераций, которое алгоритм будет выполнять для сходимости. Увеличение этого значения может помочь модели сойтись, если данных много или они сложны.

        #### **Дерево решений (Decision Tree)**
        * **Что делает**: Дерево решений — это непараметрический алгоритм обучения с учителем, используемый для **классификации** и регрессии. Он строит модель в виде структуры дерева, где каждый внутренний узел представляет "тест" на атрибуте, каждая ветвь — результат теста, а каждый листовой узел — метку класса.
        * **За что отвечает**: Отвечает за выявление иерархических правил принятия решений на основе признаков данных. Деревья решений интуитивно понятны и хорошо визуализируются.
        * **Как можно настраивать**:
            * **`max_depth` (Максимальная глубина)**: Ограничивает максимальную глубину дерева. Это важный параметр для предотвращения переобучения. Большие значения могут привести к более сложным моделям и переобучению.
            * **`min_samples_split` (Минимальное число образцов для разделения)**: Минимальное количество образцов, необходимое для того, чтобы узел мог быть разделён. Увеличение этого значения делает дерево более "ленивым" и менее склонным к переобучению.

        #### **Случайный лес (Random Forest)**
        * **Что делает**: Случайный лес — это ансамблевый метод обучения, который строит большое количество деревьев решений и объединяет их предсказания для получения более точного и стабильного результата. Он снижает проблему переобучения, характерную для отдельных деревьев.
        * **За что отвечает**: Отвечает за улучшение точности и устойчивости модели за счет использования множества "слабых" учащихся (деревьев), работающих совместно.
        * **Как можно настраивать**:
            * **`n_estimators` (Число деревьев)**: Количество деревьев решений в лесу. Чем больше деревьев, тем более стабильным и точным будет предсказание, но тем дольше будет обучение.
            * **`max_depth` (Максимальная глубина)**: Максимальная глубина каждого дерева в лесу. Аналогично дереву решений, ограничивает сложность отдельных деревьев.
            * **`min_samples_split` (Минимальное число образцов для разделения)**: Минимальное количество образцов, необходимое для разделения узла в каждом дереве.

        #### **XGBoost (Extreme Gradient Boosting)**
        * **Что делает**: XGBoost — это высокопроизводительная и эффективная реализация градиентного бустинга, используемая для **классификации** и регрессии. Она строит деревья решений последовательно, где каждое новое дерево пытается исправить ошибки предыдущих.
        * **За что отвечает**: Отвечает за создание мощных и точных моделей, особенно на табличных данных. Известен своей скоростью и производительностью.
        * **Как можно настраивать**:
            * **`learning_rate` (Темп обучения)**: Шаг, с которым веса новых деревьев добавляются к общей модели. Меньшие значения делают обучение более медленным, но часто приводят к лучшей точности и предотвращают переобучение.
            * **`n_estimators` (Число итераций / деревьев)**: Количество бустинговых раундов или количество деревьев, которые будут построены. Чем больше, тем дольше обучение, но потенциально выше точность.
            * **`max_depth` (Максимальная глубина)**: Максимальная глубина каждого бустингового дерева. Контролирует сложность отдельных деревьев.

        #### **Метод Опорных Векторов (SVC - Support Vector Classifier)**
        * **Что делает**: SVC — это мощный алгоритм для **классификации**, который находит оптимальную гиперплоскость, максимально разделяющую классы в пространстве признаков.
        * **За что отвечает**: Отвечает за нахождение "наилучшей" границы разделения между классами, что делает его устойчивым к шуму и эффективным в высокоразмерных пространствах.
        * **Как можно настраивать (параметры по умолчанию в коде)**:
            * SVC имеет множество параметров (например, `C` для регуляризации, `kernel` для выбора функции ядра, `gamma` для влияния одной тренировочной точки). В данном коде используется реализация по умолчанию, которая подходит для начала. Для более тонкой настройки требуется углубленное понимание этих параметров.

        #### **KNN-классификатор (K-Nearest Neighbors Classifier)**
        * **Что делает**: KNN — это непараметрический алгоритм **классификации**, который присваивает новому объекту класс, основываясь на большинстве голосов его `k` ближайших соседей в тренировочном наборе данных.
        * **За что отвечает**: Отвечает за классификацию на основе схожести с уже известными объектами. Он прост в понимании и реализации, но может быть медленным на больших наборах данных.
        * **Как можно настраивать**:
            * **`n_neighbors` (Количество соседей)**: Определяет количество ближайших соседей, которые будут учитываться при классификации. Нечетное значение часто используется для предотвращения "ничьих" в бинарной классификации. Большие значения могут сглаживать границу решения, меньшие - делать ее более чувствительной к шуму.

        #### **Перцептрон-классификатор (Perceptron Classifier - Нейронная сеть)**
        * **Что делает**: Представляет собой простейшую форму нейронной сети, или многослойный перцептрон (MLP). Используется для **классификации**. Он состоит из входного слоя, одного или нескольких скрытых слоев и выходного слоя, где нейроны в каждом слое связаны с нейронами следующего слоя.
        * **За что отвечает**: Отвечает за изучение сложных нелинейных зависимостей в данных путем прохождения информации через несколько слоев нейронов с применением функций активации.
        * **Как можно настраивать**:
            * **`layers_target` (Количество скрытых слоев)**: Определяет, сколько скрытых слоев будет в нейронной сети. Увеличение количества слоев позволяет модели изучать более сложные абстракции, но также увеличивает риск переобучения и время обучения.
            * **`neurons_target` (Количество нейронов в слое)**: Количество нейронов в каждом скрытом слое. Большее количество нейронов увеличивает "емкость" модели, позволяя ей запоминать больше информации, но также требует больше данных для обучения и может привести к переобучению.
            * **`learning_rate_target` (Темп обучения)**: Определяет размер шага, с которым веса нейронной сети корректируются во время обучения. Малый темп обучения может замедлить сходимость, но помочь найти более оптимальное решение. Большой темп обучения может привести к нестабильности обучения и пропуску оптимального решения.
            * **`epochs_target` (Количество эпох)**: Количество полных проходов по всему тренировочному набору данных. Каждая эпоха позволяет модели уточнить свои веса. Слишком мало эпох может привести к недообучению, слишком много — к переобучению.

        ### Пояснения к тумблерам "Ансамбль моделей":

        * **Использовать кросс-валидацию (use_cv)**: Если включен, модель будет оцениваться с использованием кросс-валидации на тренировочном наборе данных. Это помогает получить более надежную оценку производительности модели, усредняя результаты по нескольким "фолдам" (подмножествам данных).
        * **Количество фолдов (cv_folds)**: Определяет количество частей, на которые будет разбит тренировочный набор данных для кросс-валидации. Например, 5 фолдов означает, что данные будут разбиты на 5 частей, и модель будет обучаться и тестироваться 5 раз, каждый раз используя новую часть в качестве тестовой.
        * **Метод ансамблирования**:
            * **Голосование (Voting)**: Этот метод объединяет предсказания нескольких моделей. Для классификации он выбирает класс, который предсказало большинство моделей (`'hard'` voting). Это простой и эффективный способ улучшить стабильность предсказаний.
            * **Стэкинг (Stacking)**: Более сложный метод ансамблирования. Он обучает "мета-модель" (в данном случае, Логистическую регрессию) на предсказаниях базовых моделей. То есть, базовые модели делают предсказания, а затем эти предсказания используются как входные признаки для обучения финальной модели. Это позволяет мета-модели "учиться" на ошибках и сильных сторонах базовых моделей.
        """)
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
  st.subheader("Logistic Regression")
  run_logreg = st.checkbox("Создать модель Логистической Регрессии!")
  if run_logreg:
    model, metrics_df, y_train_pred, y_test_pred  = logistic_regression(X_train, X_test, y_train, y_test)
    logreg_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())

with col2:
  st.subheader("Decision Tree")
  deps = st.text_input("max_depth", value = 10)
  minsamples = st.text_input("min_samples", value = 10)
  run_tree = st.checkbox("Создать модель дерева!")
  if deps and minsamples and run_tree:
    model, metrics_df, y_train_pred, y_test_pred = tree(X_train, X_test, y_train, y_test, max_depth_target = eval(deps), min_samples_split_target = eval(minsamples))
    tree_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())

with col3:
  st.subheader("Random Forest")
  rf_estimators = st.text_input("RF_estimators", value = 50)
  rf_deps = st.text_input("RF_max_depth", value = 10)
  rf_minsamples = st.text_input("RF_min_samples", value = 10)
  run_rf = st.checkbox("Создать модель Случайного Леса!")
  if rf_estimators and rf_deps and rf_minsamples and run_rf:
    model, metrics_df, y_train_pred, y_test_pred = random_forest(X_train, X_test, y_train, y_test, estimators_target = eval(rf_estimators), max_depth_target = eval(rf_deps), min_samples_split_target = eval(rf_minsamples))
    rf_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())

with col4:
  st.subheader("KNN")
  knn_neighbors = st.text_input("knn_neighbors_target", value = 10)
  run_knn = st.checkbox("Создать модель KNN!")
  if knn_neighbors and run_knn:
    model, metrics_df, y_train_pred, y_test_pred = knn_classifier(X_train, X_test, y_train, y_test, neighbors_target = eval(knn_neighbors))
    knn_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())

with col5:
  st.subheader("XGBoost")
  xgb_learning_rate = st.text_input("xgb_learning_rate", value = 0.01)
  xgb_estimators = st.text_input("XGB_min_samples", value = 50)
  xgb_deps = st.text_input("XGB_max_depth", value = 10)
  run_xgb = st.checkbox("Создать модель XGBoost!")
  if run_xgb and xgb_learning_rate and xgb_estimators and xgb_deps:
    model, metrics_df, y_train_pred, y_test_pred = xgboost(X_train, X_test, y_train, y_test, learning_rate = eval(xgb_learning_rate), n_estimators = eval(xgb_estimators), max_depth = eval(xgb_deps))
    xgb_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())

with col6:
  st.subheader("SVC")
  run_svc = st.checkbox("Создать модель SVC!")
  if run_svc:
    model, metrics_df, y_train_pred, y_test_pred = svc(X_train, X_test, y_train, y_test)
    svc_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())


with col7:
  st.subheader("Perceptron")
  p_layers_target = st.text_input("layers_target", value = 2)
  p_neurons_target = st.text_input("neurons_target", value = 50)
  p_learning_rate_target = st.text_input("learning_rate_target", value = 0.01)
  p_epochs_target = st.text_input("epochs_target", value = 10)
  run_perceptron = st.checkbox("Создать модель Неиросеть!")
  if p_layers_target and p_neurons_target and p_learning_rate_target and p_epochs_target and run_perceptron:
    model, metrics_df, y_train_pred, y_test_pred =  perceptron_classifier(X_train, X_test, y_train, y_test,
                          layers_target = eval(p_layers_target), neurons_target = eval(p_neurons_target), learning_rate_target = eval(p_learning_rate_target),
                          epochs_target = eval(p_epochs_target))
    perceptron_model = model
    st.subheader("Metrics")
    st.dataframe(metrics_df)

    st.subheader("Train predictions")
    st.dataframe(pd.Series(y_train_pred).head())

    st.subheader("Test predictions")
    st.dataframe(pd.Series(y_test_pred).head())

# ======= Доступные функции =======
available_functions = {
    'Логистическая регрессия': logistic_regression,
    'Дерево решений': tree,
    'Рандомный лес': random_forest,
    'XGboost': xgboost,
    'Метод Опорных Векторов': svc,
    'KNN-классификатор': knn_classifier,
    'Перцептрон-классификатор': perceptron_classifier
}

use_cv = st.sidebar.checkbox("Использовать кросс-валидацию", value=False)
cv_folds = st.sidebar.slider("Количество фолдов", 2, 10, 5)

# Ансамблирование

def voting_ensemble(models, X_train, X_test, y_train, y_test):
    """
    Ансамбль методом голосования
    """
    voting = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='hard'
    )
    voting.fit(X_train, y_train)
    return evaluate_model(voting, X_train, X_test, y_train, y_test, "Voting Ensemble")

def stacking_ensemble(models, X_train, X_test, y_train, y_test):
    """
    Ансамбль методом стэкинга
    """
    stacking = StackingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    stacking.fit(X_train, y_train)
    return evaluate_model(stacking, X_train, X_test, y_train, y_test, "Stacking Ensemble")

st.title("Ансамбль моделей")

# Собираем построенные модели
built_models = {}
if 'logreg_model' in globals():
    built_models['Logistic Regression'] = logreg_model
if 'tree_model' in globals():
    built_models['Decision Tree'] = tree_model
if 'rf_model' in globals():
    built_models['Random Forest'] = rf_model
if 'xgb_model' in globals():
    built_models['XGBoost'] = xgb_model
if 'svc_model' in globals():
    built_models['SVC'] = svc_model
if 'knn_model' in globals():
    built_models['KNN'] = knn_model
if 'perceptron_model' in globals():
    built_models['Perceptron'] = perceptron_model

if not built_models:
    st.warning("Постройте хотя бы одну модель перед созданием ансамбля")
    st.stop()

# Выбор моделей для ансамбля
selected_models = st.multiselect(
    "Выберите модели для ансамбля (..но не перцептрон)",
    options=list(built_models.keys()),
    default=list(built_models.keys())
)

if len(selected_models) < 2:
    st.warning("Для ансамбля нужно выбрать хотя бы 2 модели")
    st.stop()

# Выбор метода ансамблирования
ensemble_method = st.selectbox(
    "Метод ансамблирования",
    ["Голосование", "Стэкинг"],
    index=0
)

# Кнопка запуска ансамбля
if st.checkbox("Построить ансамбль"):
    # Собираем выбранные модели
    models_to_ensemble = {name: built_models[name] for name in selected_models}

    with st.spinner("Строим ансамбль..."):
        if ensemble_method == "Голосование":
            model, metrics, train_pred, test_pred = voting_ensemble(
                models_to_ensemble, X_train, X_test, y_train, y_test
            )
        else:
            model, metrics, train_pred, test_pred = stacking_ensemble(
                models_to_ensemble, X_train, X_test, y_train, y_test
            )

        # Сохраняем результаты
        ensemble_model = model
        ensemble_metrics = metrics
        ensemble_train_pred = train_pred
        ensemble_test_pred = test_pred

        # Вывод результатов
        st.success("Ансамбль успешно построен!")

        st.subheader("Метрики ансамбля")
        st.dataframe(metrics.style.format("{:.2%}"))

        st.subheader("Train predictions")
        st.dataframe(pd.DataFrame({
            'Actual': y_train,
            'Predicted': train_pred
        }).head())

        st.subheader("Test predictions")
        st.dataframe(pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_pred
        }).head())
# Реализовать предсказание по построенным моделям с визуализацией и выгрузкой
st.title("Предсказание на новых данных")

new_file = st.file_uploader("Загрузите новые данные (CSV или Excel)", type=["csv", "xlsx"])

if new_file:
    # Определяем тип файла
    if new_file.name.endswith('.csv'):
        new_data = pd.read_csv(new_file, sep=sep_sign, decimal=decimal_sign)
    else:
        new_data = pd.read_excel(new_file)

    st.success(f"Успешно загружено {len(new_data)} записей")
    st.dataframe(new_data.head())

    # 2. Предобработка новых данных
    st.header("Предобработка данных")

    try:
        # Применяем те же преобразования, что и к исходным данным
        processed_data = new_data.copy()

        # Сохраняем обработанные данные
        new_data_processed = handle_missing_values(processed_data, method)

        st.success("Данные успешно предобработаны!")
        st.dataframe(new_data_processed.head())

    except Exception as e:
        st.error(f"Ошибка при предобработке данных: {str(e)}")
        st.stop()

    with st.spinner("Обработка данных..."):
        # Полная обработка данных с разбиением
        X_train_new, X_test_new, y_train_new, y_test_new, train_ids_new, test_ids_new = preprocess_data(
            data=new_data_processed,
            target_col=target_col,
            id_cols=id_cols,
            features=selected_features,
            norm_cols=norm_cols,
            log_cols=log_cols,
            dummy_cols=dummy_cols,
            balance_method=balance_method,
            test_size=0.1,
            random_state=random_state
        )
    new_preprocess_data = np.vstack([X_train_new, X_test_new])

    # 3. Выбор моделей для предсказания
    st.header("3. Выбор моделей для предсказания")

    available_models = built_models

    if not available_models:
        st.warning("Нет доступных моделей. Пожалуйста, постройте хотя бы одну модель.")
        st.stop()

    selected_models = st.multiselect(
        "Выберите модели для предсказания",
        options=list(available_models.keys()),
        default=list(available_models.keys())
    )

    # 4. Выполнение предсказаний
    if st.checkbox("Выполнить предсказания"):
        predictions = pd.DataFrame(index=pd.DataFrame(new_preprocess_data).index)

        for model_name in selected_models:
            model = available_models[model_name]

            try:
                # Особый случай для XGBoost (нужно кодирование меток)
                if model_name == 'XGBoost':
                    if label_encoder not in globals():
                        st.error("Не найден кодировщик меток для XGBoost")
                        continue

                    le = st.session_state.label_encoder
                    pred = model.predict(new_preprocess_data)
                    predictions[model_name] = le.inverse_transform(pred)

                # Особый случай для перцептрона
                elif model_name == 'Perceptron':
                    if perceptron_encoder not in globals():
                        st.error("Не найден кодировщик меток для перцептрона")
                        continue

                    le = perceptron_encoder
                    pred_proba = model.predict(new_preprocess_data)
                    pred = np.argmax(pred_proba, axis=1)
                    predictions[model_name] = le.inverse_transform(pred)

                # Для остальных моделей
                else:
                    predictions[model_name] = model.predict(new_preprocess_data)

            except Exception as e:
                st.error(f"Ошибка при предсказании с помощью {model_name}: {str(e)}")
                continue

        # Добавляем голосование, если выбрано несколько моделей
        if len(selected_models) > 1:
            def majority_vote(row):
                votes = [row[model] for model in selected_models]
                return Counter(votes).most_common(1)[0][0]

            predictions['Majority_Vote'] = predictions.apply(majority_vote, axis=1)

        # Сохраняем предсказания
        predictions = predictions

        # Выводим результаты
        st.header("Результаты предсказаний")
        st.dataframe(predictions)

        # Кнопка для скачивания результатов
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать предсказания как CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )
