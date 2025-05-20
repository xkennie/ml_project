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
from sklearn.naive_bayes import GaussianNB
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
if st.button("Обработать пропущенные значения", type="primary"):
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

st.header("Детальный анализ")

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
st.subheader("Гистограмма")
selected_col = st.selectbox(
        "Выберите колонку для гистограммы",
        options=df.columns,
        index=2
    )
hist_values = np.histogram(df[selected_col], bins=int(round( len(df[selected_col])**0.5 ,0)))
    
    # Преобразуем в DataFrame для st.bar_chart
hist_df = pd.DataFrame({
        'bin_left': hist_values[1][:-1],
        'bin_right': hist_values[1][1:],
        'count': hist_values[0]
    })
hist_df['bin'] = hist_df.apply(lambda x: f"{x['bin_left']:.2f}-{x['bin_right']:.2f}", axis=1)
    
# Отображаем барчарт
st.bar_chart(hist_df.set_index('bin')['count'])
    
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

st.subheader("Подготовка данных для моделирования")
    
    # 1. Выбор таргетной переменной (y)
target_col = st.selectbox(
        "Выберите таргетную переменную (y)",
        options=df.columns,
        index=0,  # Дефолтное значение - первая колонка
        key="target_select"
    )
    
    # 2. Выбор признаков (X) - исключаем таргет из возможных признаков
available_features = [col for col in df.columns if col != target_col]
    
selected_features = st.multiselect(
        "Выберите признаки для модели (X)",
        options=available_features,
        default=available_features,  # По умолчанию выбираем все доступные признаки
        key="features_select"
    )
    
    # Проверка, что выбраны хотя бы один признак
if not selected_features:
  st.error("Пожалуйста, выберите хотя бы один признак для модели")
      
if selected_features:    
    # Формируем X и y
  y = df[target_col]
  X = df[selected_features]
    
    # 3. Разбиение на train/test
  X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Выводим информацию о разбиении
  st.success("Данные успешно подготовлены!")
  st.write(f"Выбран таргет: {target_col}")
  st.write(f"Выбрано признаков: {len(selected_features)}")
  st.write(f"Размер обучающей выборки: {X_train.shape[0]}")
  st.write(f"Размер тестовой выборки: {X_test.shape[0]}")
def preprocess(X_train, X_test, y_train, y_test):
  return X_train, X_test, y_train, y_test
    
#функции с реализациями методов ML
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  
#logreg
def logistic_regression(X_train, X_test, y_train, y_test):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  logreg = LogisticRegression(
    multi_class='multinomial',
    max_iter = 1000
)
  logreg.fit(X_train, y_train)
  y_pred = logreg.predict(X_test)
  predict = pd.Series(y_pred)
  logreg_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  logreg_predicts.columns = ['index', 'Style', 'Logreg_predict']
  return logreg_predicts

#tree
def tree(X_train, X_test, y_train, y_test, max_depth_target = 10, min_samples_split_target = 10):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  tree = DecisionTreeClassifier(
    max_depth = max_depth_target,             # Максимальная глубина
    min_samples_split = min_samples_split_target,     # Минимальное число образцов для разделения
)
  tree.fit(X_train, y_train)
  y_pred = tree.predict(X_test)
  #y_pred_train = tree.predict(X_train)
  predict = pd.Series(y_pred)
  tree_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  #tree_predicts = tree_predicts.replace({0: 'predict'})
  tree_predicts.columns = ['index', 'Style', 'Tree_predict']
  #return tree_train_predicts, tree_predicts
  return tree_predicts

#forest
def random_forest(X_train, X_test, y_train, y_test, estimators_target = 50, max_depth_target = 10, min_samples_split_target = 10):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  random_forest = RandomForestClassifier(
    n_estimators = estimators_target,  # Число деревьев
    max_features='sqrt', 
    max_depth = max_depth_target,
    min_samples_split = min_samples_split_target
)
  random_forest.fit(X_train, y_train)
  y_pred = random_forest.predict(X_test)
  predict = pd.Series(y_pred)
  random_forest_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  random_forest_predicts.columns = ['index', 'Style', 'Random_Forest_predict']
  return random_forest_predicts

#xgboost
def xgboost(X_train, X_test, y_train, y_test, learning_rate_target = 0.01, estimators_target = 50, max_depth_target = 10):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  xgboost = XGBClassifier(
    n_estimators = estimators_target,
    max_depth = max_depth_target,
    learning_rate = learning_rate_target,
    random_state = 42
)
  xgboost.fit(X_train, y_train)
  y_pred = xgboost.predict(X_test)
  y_pred = le.inverse_transform(y_pred)
  predict = pd.Series(y_pred)
  xgboost_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  xgboost_predicts.columns = ['index', 'Style', 'XGBoost_predict']
  return xgboost_predicts

#Support-vector-calc
def svc(X_train, X_test, y_train, y_test):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  svc = SVC()
  svc.fit(X_train, y_train)
  y_pred = svc.predict(X_test)
  predict = pd.Series(y_pred)
  svc_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  svc_predicts.columns = ['index', 'Style', 'SVC_predict']
  return svc_predicts
#knn
def knn_classifier(X_train, X_test, y_train, y_test, neighbors_target = 10):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  knn = KNeighborsClassifier(n_neighbors= neighbors_target)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  predict = pd.Series(y_pred)
  knn_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  knn_predicts.columns = ['index', 'Style', 'KNN_predict']
  return knn_predicts

def perceptron_classifier(X_train, X_test, y_train, y_test,
                          layers_target = 2, neurons_target = 50, learning_rate_target = 0.01,
                          epochs_target = 10):
  X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
  #y_train encode
  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y_train)
  y_train = to_categorical(y_encoded)  
  #y_test encode
  y_encoded = label_encoder.fit_transform(y_test)
  y_test = to_categorical(y_encoded)

  perceptron = Sequential()
  for n in range(layers_target):
    perceptron.add(Dense(neurons_target, input_dim = X_train.shape[1], activation = 'relu'))  # Скрытые слои
  
  perceptron.add(Dense(y_train.shape[1], activation = 'softmax'))  # Выходной слой


  perceptron.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = learning_rate_target), metrics=['accuracy'])

  perceptron.fit(X_train, y_train, epochs= epochs_target, batch_size=10, validation_data=(X_test, y_test))

  predictions = perceptron.predict(X_test)
  predicted_classes = np.argmax(predictions, axis=1)
  predicted_styles = label_encoder.inverse_transform(predicted_classes)
  predict = pd.Series(predicted_styles)

  true_class = np.argmax(y_test, axis = 1)
  true_styles = label_encoder.inverse_transform(true_class)
  y_true = pd.Series(true_styles)

  perceptron_predicts = pd.concat([y_true, predict], axis = 1)
  perceptron_predicts.columns = ['Style', 'Perceptron_predict']
  perceptron_predicts["index"] = [i for i in range(perceptron_predicts.shape[0])]
  perceptron_predicts = perceptron_predicts[["index", "Style", "Perceptron_predict"]]                        
  return perceptron_predicts
# ======== Models row ============
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
  st.subheader("Logistic Regression")
  run_logreg = st.checkbox("Создать модель Логистической Регрессии!")
  if run_logreg:
    #_, result = logistic_regression(X_train, X_test, y_train, y_test)
    result = logistic_regression(X_train, X_test, y_train, y_test)
    acc = (result['Logreg_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)
    
with col2:
  st.subheader("Tree")
  deps = st.text_input("max_depth", value = 10)
  minsamples = st.text_input("min_samples", value = 10)
  run_tree = st.checkbox("Создать модель дерева!")
  if deps and minsamples and run_tree:
    result = tree(X_train, X_test, y_train, y_test, max_depth_target = eval(deps), min_samples_split_target = eval(minsamples))
    acc = (result['Tree_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)

    
with col3:
  st.subheader("Random Forest")
  rf_estimators = st.text_input("RF_estimators", value = 50)
  rf_deps = st.text_input("RF_max_depth", value = 10)
  rf_minsamples = st.text_input("RF_min_samples", value = 10)
  run_rf = st.checkbox("Создать модель Случайного Леса!")
  if rf_estimators and rf_deps and rf_minsamples and run_rf:
    result = random_forest(X_train, X_test, y_train, y_test, estimators_target = eval(rf_estimators), max_depth_target = eval(rf_deps), min_samples_split_target = eval(rf_minsamples))
    acc = (result['Random_Forest_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)

with col4:
  st.subheader("XGBoost")
  xgb_learning_rate = st.text_input("XGB_max_depth", value = 0.01)
  xgb_estimators = st.text_input("XGB_min_samples", value = 50)
  xgb_deps = st.text_input("XGB_max_depth", value = 10)
  run_xgb = st.checkbox("Создать модель XGBoost!")
  if run_xgb and xgb_learning_rate and xgb_estimators and xgb_deps:
    result = xgboost(X_train, X_test, y_train, y_test, learning_rate_target = eval(xgb_learning_rate), estimators_target = eval(xgb_estimators), max_depth_target = eval(xgb_deps))
    acc = (result['XGBoost_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)

with col5:
  st.subheader("SVC")
  run_svc = st.checkbox("Создать модель SVC!")
  if run_svc:
    result = svc(X_train, X_test, y_train, y_test)
    acc = (result['SVC_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)

with col6:
  st.subheader("KNN")
  knn_neighbors = st.text_input("knn_neighbors_target", value = 10)
  run_knn = st.checkbox("Создать модель KNN!")
  if knn_neighbors and run_knn:
    result = knn_classifier(X_train, X_test, y_train, y_test, neighbors_target = eval(knn_neighbors))
    acc = (result['KNN_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)
    
with col7:
  st.subheader("Perceptron")
  p_layers_target = st.text_input("layers_target", value = 2)
  p_neurons_target = st.text_input("neurons_target", value = 50)
  p_learning_rate_target = st.text_input("learning_rate_target", value = 0.01)
  p_epochs_target = st.text_input("epochs_target", value = 10)
  run_perceptron = st.checkbox("Создать модель Неиросеть!")
  if p_layers_target and p_neurons_target and p_learning_rate_target and p_epochs_target and run_perceptron:
    result =  perceptron_classifier(X_train, X_test, y_train, y_test,
                          layers_target = eval(p_layers_target), neurons_target = eval(p_neurons_target), learning_rate_target = eval(p_learning_rate_target),
                          epochs_target = eval(p_epochs_target)) 
    acc = (result['Perceptron_predict'] == result['Style']).mean()
    st.write(f"Accuracy: {acc:.2%}")
    st.write(result)

# ======= Доступные функции =======
available_functions = {
    'logistic_regression': logistic_regression,
    'tree': tree,
    'random_forest': random_forest,
    'xgboost': xgboost,
    'svc': svc,
    'knn_classifier': knn_classifier,
    'perceptron_classifier': perceptron_classifier
}

st.title("ML Ансамбль")

# ======= Выбор моделей =======
selected_function_names = st.multiselect("Выберите модели", list(available_functions.keys()))

# ======= Ввод параметров =======
func_params = {}
for name in selected_function_names:
    func = available_functions[name]
    sig = inspect.signature(func)
    params_to_remove = {'X_train', 'X_test', 'y_train', 'y_test'}
    new_params = {
        name: param 
        for name, param in sig.parameters.items()
        if name not in params_to_remove
    }
    sig = sig.replace(parameters=list(new_params.values()))
    #sig = list(sig)
    #sig = sig.remove(X_train)
    #sig = sig.remove(X_test)
    #sig = sig.remove(y_train)
    #sig = sig.remove(y_test)
    st.subheader(f"Параметры для {name}")
    params = {}
    for param in sig.parameters.values():
        default_val = "" if param.default is param.empty else param.default
        input_val = st.text_input(f"{name} - {param.name}", value=str(default_val))
        params[param.name] = eval(input_val)
    params["X_train"] = X_train 
    params["X_test"] = X_test 
    params["y_train"] = y_train 
    params["y_test"] = y_test 
    func_params[name] = params
    

# ======= Кнопка запуска =======
if st.button("Запустить ансамбль"):
    merged_df = None
    for name in selected_function_names:
        func = available_functions[name]
        params = func_params.get(name, {})
        result = func(**params)
        if merged_df is None:
            merged_df = result
        else:
            merged_df = pd.merge(merged_df, result, on=['index', 'Style'])

    # ======= Голосование =======
    pred_cols = [col for col in merged_df.columns if col.endswith('_predict')]
    def majority_vote(row):
        votes = [row[col] for col in pred_cols]
        return Counter(votes).most_common(1)[0][0]

    merged_df['overall_predict'] = merged_df.apply(majority_vote, axis=1)

    # ======= Accuracy (если есть столбец ground truth) =======
    if 'Style' in merged_df.columns:
        acc = (merged_df['overall_predict'] == merged_df['Style']).mean()
        st.write(f"Accuracy: {acc:.2%}")
    
    st.write("Результат ансамбля:")
    st.dataframe(merged_df[['Style', 'overall_predict']])
