import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import inspect
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# classification models
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Keras/TensorFlow для перцептрона
import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier

# Настройка страницы Streamlit
st.set_page_config(page_title="HSE | ТИИПА", layout="wide")

# Инициализация session_state
def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.backup_df = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.train_ids = None
        st.session_state.test_ids = None
        st.session_state.models = {}
        st.session_state.scaler = None
        st.session_state.onehot_encoder = None
        st.session_state.label_encoders = {}
        st.session_state.target_col = None
        st.session_state.id_cols = []
        st.session_state.selected_features = []
        st.session_state.norm_cols = []
        st.session_state.log_cols = []
        st.session_state.dummy_cols = []
        st.session_state.balance_method = "Нет"
        st.session_state.method = "Заменить на медиану"
        st.session_state.sep_sign = ";"
        st.session_state.decimal_sign = ","
        st.session_state.ensemble_model = None
        st.session_state.preprocess_params = {}

initialize_session_state()

# Заголовок и информация
st.title("© HSE | ТИИПА")
st.markdown("**Fast-touch classification-analytical tool**")
st.markdown("---")

# Блок информации о команде
with st.expander("ℹ О команде разработчиков"):
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

# Функция для загрузки данных
def load_data(uploaded_file):
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            sep_sign = st.selectbox(
                "Выберите разделитель",
                (";", ",", " ", "|"), index=0)
            st.session_state.sep_sign = sep_sign

            decimal_sign = st.selectbox(
                "Выберите отделитель дробной части",
                (".", ","), index=1)
            st.session_state.decimal_sign = decimal_sign

            df = pd.read_csv(uploaded_file, sep=sep_sign, decimal=decimal_sign)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df.copy()
        st.session_state.backup_df = df.copy()
        st.success("Данные успешно загружены!")
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

# Функция для обработки пропущенных значений
def handle_missing_values(df, option):
    df_processed = df.copy()
    
    for col in df_processed.columns:
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
        else:
            if option == "Удалить строки с пропусками":
                df_processed = df_processed.dropna(subset=[col])
            elif option == "Заменить на моду":
                if not df_processed[col].mode().empty:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                else:
                    df_processed[col] = df_processed[col].fillna("Unknown")
            elif option == "Создать новую категорию 'Unknown'":
                df_processed[col] = df_processed[col].fillna("Unknown")
            elif option == "Экстраполяция (последнее значение)":
                df_processed[col] = df_processed[col].fillna(method='ffill')
    
    return df_processed

# Функция для предобработки данных
def preprocess_data(data, target_col, id_cols, features, norm_cols, log_cols, dummy_cols,
                   balance_method, test_size, random_state):
    try:
        # Сохраняем параметры предобработки
        st.session_state.preprocess_params = {
            'target_col': target_col,
            'features': features,
            'norm_cols': norm_cols,
            'log_cols': log_cols,
            'dummy_cols': dummy_cols,
            'balance_method': balance_method
        }
        
        X = data[features].copy()
        y = data[target_col].copy()
        ids = data[id_cols].copy() if id_cols else None

        # Кодируем категориальный таргет
        #if not pd.api.types.is_numeric_dtype(y):
            #le = LabelEncoder()
            #y = le.fit_transform(y)
            #st.session_state.label_encoders['target'] = le

        # Разбиваем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size/100,
            random_state=random_state,
            stratify=y if not pd.api.types.is_numeric_dtype(y) else None
        )

        # Сохраняем ID
        if ids is not None:
            train_ids = ids.loc[X_train.index]
            test_ids = ids.loc[X_test.index]
        else:
            train_ids, test_ids = X_train.index, X_test.index

        # 1. Логарифмирование
        for col in log_cols:
            if col in X_train.columns:
                if (X_train[col] <= 0).any():
                    st.warning(f"Колонка {col} содержит нули/отрицательные значения! Добавлена константа 1 перед логарифмированием.")
                    X_train[col] = np.log1p(X_train[col] + 1)
                    X_test[col] = np.log1p(X_test[col] + 1)
                else:
                    X_train[col] = np.log1p(X_train[col])
                    X_test[col] = np.log1p(X_test[col])

        # 2. One-hot кодирование
        if dummy_cols:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            train_encoded = encoder.fit_transform(X_train[dummy_cols])
            test_encoded = encoder.transform(X_test[dummy_cols])

            encoded_cols = encoder.get_feature_names_out(dummy_cols)
            train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
            test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)

            X_train = X_train.drop(columns=dummy_cols).join(train_encoded_df)
            X_test = X_test.drop(columns=dummy_cols).join(test_encoded_df)
            st.session_state.onehot_encoder = encoder

        # 3. Нормализация
        if norm_cols:
            scaler = StandardScaler()
            X_train[norm_cols] = scaler.fit_transform(X_train[norm_cols])
            X_test[norm_cols] = scaler.transform(X_test[norm_cols])
            st.session_state.scaler = scaler

        # 4. Балансировка классов
        if balance_method != "Нет" and not pd.api.types.is_numeric_dtype(y_train):
            if balance_method == "Random Oversampling":
                sampler = RandomOverSampler(random_state=random_state)
            elif balance_method == "SMOTE":
                sampler = SMOTE(random_state=random_state)
            elif balance_method == "Random Undersampling":
                sampler = RandomUnderSampler(random_state=random_state)
            
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            
            # Визуализация балансировки
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            pd.Series(y).value_counts().plot(kind='bar', ax=ax1, title='До балансировки')
            pd.Series(y_train).value_counts().plot(kind='bar', ax=ax2, title='После балансировки')
            st.pyplot(fig)

        return X_train, X_test, y_train, y_test, train_ids, test_ids
    
    except Exception as e:
        st.error(f"Ошибка при предобработке данных: {str(e)}")
        return None, None, None, None, None, None

# Функция для обработки новых данных
def process_new_data(new_df):
    try:
        params = st.session_state.preprocess_params
        if not params:
            raise ValueError("Сначала нужно подготовить тренировочные данные!")
        
        X_new = new_df[params['features']].copy()
        
        # 1. Логарифмирование
        for col in params['log_cols']:
            if col in X_new.columns:
                X_new[col] = np.log1p(X_new[col] + 1)
        
        # 2. One-hot кодирование
        if 'onehot_encoder' in st.session_state and params['dummy_cols']:
            encoded = st.session_state.onehot_encoder.transform(X_new[params['dummy_cols']])
            encoded_cols = st.session_state.onehot_encoder.get_feature_names_out(params['dummy_cols'])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X_new.index)
            X_new = X_new.drop(params['dummy_cols'], axis=1).join(encoded_df)
        
        # 3. Нормализация
        if 'scaler' in st.session_state and params['norm_cols']:
            X_new[params['norm_cols']] = st.session_state.scaler.transform(X_new[params['norm_cols']])
        
        return X_new
    
    except Exception as e:
        st.error(f"Ошибка при обработке новых данных: {str(e)}")
        return None

# Основной интерфейс
st.title("Данные для анализа")
uploaded_file = st.file_uploader("Выберите файл (CSV или Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.dataframe(df.head())

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
        help="Выберите метод обработки пропущенных значений",
        key="missing_method_select"
    )
    st.session_state.method = method # Сохраняем выбранный метод

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
if st.checkbox("Обработать пропущенные значения", key="process_missing_checkbox"):
    with st.spinner("Обработка данных..."):
        df = handle_missing_values(df, method)
        st.session_state.df = df.copy() # Обновляем df в session_state
        st.success("Обработка завершена!")

        # Вывод результатов
        st.markdown("---")
        st.subheader("Результаты обработки")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Исходное количество строк", len(st.session_state.backup_df))
        with col2:
            st.metric("Количество строк после обработки", len(st.session_state.df))

# Показать обработанный датафрейм
st.dataframe(st.session_state.df)


# Визуальный анализ данных
if st.session_state.df is not None:
    st.title("Визуальный анализ данных")
    st.subheader("Описательная статистика")

    desc = st.session_state.df.describe().T
    desc['missing'] = st.session_state.df.isna().sum()
    desc['missing_percent'] = (desc['missing'] / len(st.session_state.df)).round(2)
    desc['dtype'] = st.session_state.df.dtypes
    desc['unique'] = st.session_state.df.nunique()

    st.dataframe(
        desc.style.format({
            'count': '{:.1f}',
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
    # Выбор типа графика
    chart_type = st.radio(
    "Тип визуализации распределения:",
    ["Гистограмма", "Ящик с усами (Boxplot)", "Плотность распределения"],
    horizontal=True
)

    # Графики распределения
    st.subheader("Распределение признаков")
    col1, col2 = st.columns([3, 1])
    with col1:
        numeric_cols_for_plot = st.session_state.df.select_dtypes(include=['number']).columns
        if len(numeric_cols_for_plot) > 0:
            selected_col = st.selectbox(
                "Выберите переменную для анализа:",
                options=numeric_cols_for_plot,
                index=0
            )
        else:
            st.warning("Нет числовых колонок для построения графиков распределения.")
            selected_col = None
    
    with col2:
        bins = 20 # Значение по умолчанию
        if chart_type == "Гистограмма" and selected_col:
            bins = st.slider(
                "Количество интервалов:",
                min_value=5,
                max_value=50,
                value=int(np.sqrt(len(st.session_state.df)) if len(st.session_state.df) > 0 else 20)
            )
    
    # Отображение выбранного графика
    if selected_col:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 6))
    
        if chart_type == "Гистограмма":
            sns.histplot(st.session_state.df[selected_col], bins=bins, kde=True, ax=ax)
            ax.set_title(f'Распределение {selected_col}')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Частота')
        elif chart_type == "Ящик с усами (Boxplot)":
            sns.boxplot(x=st.session_state.df[selected_col], ax=ax)
            ax.set_title(f'Boxplot для {selected_col}')
        elif chart_type == "Плотность распределения":
            sns.kdeplot(st.session_state.df[selected_col], ax=ax, fill=True)
            ax.set_title(f'Плотность распределения {selected_col}')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Плотность')
    
        st.pyplot(fig, clear_figure=True)
    
    st.markdown("---")
    
    st.subheader("Scatter plot")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_axis = st.selectbox(
            "Ось X",
            options=st.session_state.df.columns,
            index=0
        )
    with col2:
        y_axis = st.selectbox(
            "Ось Y",
            options=st.session_state.df.columns,
            index=1 if len(st.session_state.df.columns) > 1 else 0
        )
    
    with col3:
        color_col = st.selectbox(
            "Цвет",
            options=["None"] + list(st.session_state.df.columns),
            index=0
        )
    with col4:
        size_col = st.selectbox(
            "Размер",
            options=["None"] + list(st.session_state.df.columns),
            index=0
        )
    
    if (color_col == "None") and (size_col == "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis)
    elif (size_col == "None") and (color_col != "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis, color=color_col)
    elif (size_col != "None") and (color_col == "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis, size = size_col)
    elif (size_col != "None") and (color_col != "None"):
        st.scatter_chart(st.session_state.df, x=x_axis, y=y_axis, color = color_col, size = size_col)

# Подготовка данных для моделирования
if st.session_state.df is not None:
    st.title("Подготовка данных для моделирования")
    
    target_col = st.selectbox(
        "Выберите целевую переменную:",
        options=st.session_state.df.columns,
        index=0
    )
    st.session_state.target_col = target_col

    # Анализ по классам
    if st.session_state.target_col:
        st.subheader(f"Анализ по классам ({st.session_state.target_col})")
        class_stats = st.session_state.df.groupby(st.session_state.target_col,as_index=False).mean().merge(pd.DataFrame(st.session_state.df[st.session_state.target_col].value_counts()).reset_index(), how='left')
        st.dataframe(class_stats)
        

    id_cols = st.multiselect(
        "Выберите ID-столбцы (не будут использоваться в модели):",
        options=[col for col in st.session_state.df.columns if col != target_col]
    )
    st.session_state.id_cols = id_cols

    available_features = [col for col in st.session_state.df.columns if col != target_col and col not in id_cols]
    selected_features = st.multiselect(
        "Выберите признаки для модели:",
        options=available_features,
        default=available_features
    )
    st.session_state.selected_features = selected_features

    # Настройки предобработки
    with st.expander("Настройки предобработки"):
        numeric_cols = [col for col in selected_features if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        
        st.subheader("Нормализация")
        norm_cols = st.multiselect(
            "Признаки для нормализации:",
            options=numeric_cols
        )
        st.session_state.norm_cols = norm_cols

        st.subheader("Логарифмирование")
        log_cols = st.multiselect(
            "Признаки для логарифмирования:",
            options=numeric_cols
        )
        st.session_state.log_cols = log_cols

        st.subheader("Кодирование категорий")
        categorical_cols = [col for col in selected_features if not pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        dummy_cols = st.multiselect(
            "Категориальные признаки для one-hot кодирования:",
            options=categorical_cols
        )
        st.session_state.dummy_cols = dummy_cols

        st.subheader("Балансировка классов")
        if not pd.api.types.is_numeric_dtype(st.session_state.df[target_col]):
            balance_method = st.selectbox(
                "Метод балансировки:",
                options=["Нет", "Random Oversampling", "SMOTE", "Random Undersampling"]
            )
            st.session_state.balance_method = balance_method
        else:
            st.warning("Балансировка доступна только для категориальных целевых переменных")

    # Параметры разбиения
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

    if st.button("Подготовить данные"):
        with st.spinner("Обработка данных..."):
            X_train, X_test, y_train, y_test, train_ids, test_ids = preprocess_data(
                st.session_state.df,
                st.session_state.target_col,
                st.session_state.id_cols,
                st.session_state.selected_features,
                st.session_state.norm_cols,
                st.session_state.log_cols,
                st.session_state.dummy_cols,
                st.session_state.balance_method,
                test_size,
                random_state
            )
            
            if X_train is not None:
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.train_ids = train_ids
                st.session_state.test_ids = test_ids
                
                st.success("Данные успешно подготовлены!")
                
                # Показать результаты
                st.subheader("Результаты подготовки")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Обучающая выборка", f"{len(X_train)} строк")
                    st.write("Распределение классов:")
                    st.write(pd.Series(y_train).value_counts())
                    st.dataframe(pd.concat([train_ids, X_train, y_train], axis=1))
                with col2:
                    st.metric("Тестовая выборка", f"{len(X_test)} строк")
                    st.write("Распределение классов:")
                    st.write(pd.Series(y_test).value_counts())
                    st.dataframe(pd.concat([test_ids, X_test, y_test], axis=1))
            else:
                st.error("Ошибка при подготовке данных")
def plot_confusion_matrix(y_true, y_pred, title):
    """Улучшенная визуализация матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title, pad=20)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    st.pyplot(fig, clear_figure=True)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, use_cv=False, cv_folds=5):
    """Улучшенная функция оценки моделей с сохранением в session_state"""
    try:
        # Кросс-валидация
        if use_cv:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            st.write(f"Кросс-валидация (среднее accuracy): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
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
            ]
        }
        
        metrics_df = pd.DataFrame(metrics, index=['Train', 'Test'])
        
        # Визуализация матриц ошибок
        st.subheader(f"Матрицы ошибок для {model_name}")
        col1, col2 = st.columns(2)
        with col1:
            plot_confusion_matrix(y_train, y_train_pred, f'Train: {model_name}')
        with col2:
            plot_confusion_matrix(y_test, y_test_pred, f'Test: {model_name}')
        
        # Сохраняем модель в session_state
        if 'models' not in st.session_state:
            st.session_state.models = {}
        st.session_state.models[model_name] = model
        
        return model, metrics_df, y_train_pred, y_test_pred
    
    except Exception as e:
        st.error(f"Ошибка при оценке модели {model_name}: {str(e)}")
        return None, None, None, None

# Модель логистической регрессии
def logistic_regression(X_train, X_test, y_train, y_test, use_cv=False, cv_folds=5):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial'
    )
    return evaluate_model(model, X_train, X_test, y_train, y_test, 
                         "Logistic Regression", use_cv, cv_folds)

# Модель дерева решений
def decision_tree(X_train, X_test, y_train, y_test, max_depth=10, min_samples_split=10, use_cv=False, cv_folds=5):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    return evaluate_model(model, X_train, X_test, y_train, y_test,
                         "Decision Tree", use_cv, cv_folds)

# Модель случайного леса
def random_forest(X_train, X_test, y_train, y_test, n_estimators=50, max_depth=10, 
                 min_samples_split=10, use_cv=False, cv_folds=5):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    return evaluate_model(model, X_train, X_test, y_train, y_test,
                         "Random Forest", use_cv, cv_folds)

# Модель XGBoost
def xgboost_model(X_train, X_test, y_train, y_test, learning_rate=0.01, n_estimators=50, 
                 max_depth=5, use_cv=False, cv_folds=5):
    # Кодируем метки для XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model, metrics_df, _, _ = evaluate_model(model, X_train, X_test, 
                                           y_train_encoded, y_test_encoded,
                                           "XGBoost", use_cv, cv_folds)
    
    # Сохраняем кодировщик меток
    st.session_state.label_encoder_xgboost = le
    
    return model, metrics_df, None, None

# Модель KNN
def knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5, use_cv=False, cv_folds=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return evaluate_model(model, X_train, X_test, y_train, y_test,
                         "KNN", use_cv, cv_folds)

# Модель SVM
def svm_classifier(X_train, X_test, y_train, y_test, use_cv=False, cv_folds=5):
    model = SVC(random_state=42, probability=True)
    return evaluate_model(model, X_train, X_test, y_train, y_test,
                         "SVM", use_cv, cv_folds)

# Модель перцептрона
def perceptron_classifier(X_train, X_test, y_train, y_test, layers=1, neurons=64, 
                         learning_rate=0.01, epochs=10, use_cv=False, cv_folds=5):
    # Кодируем метки для перцептрона
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)
    
    # Преобразуем метки в one-hot encoding
    y_train_onehot = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Создаем модель
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Обучение модели
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )
    
    # Предсказания
    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Обратное преобразование меток
    y_train_pred = le.inverse_transform(y_train_pred)
    y_test_pred = le.inverse_transform(y_test_pred)
    
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
        ]
    }
    
    metrics_df = pd.DataFrame(metrics, index=['Train', 'Test'])
    
    # Визуализация
    st.subheader(f"Матрицы ошибок для Perceptron")
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion_matrix(y_train, y_train_pred, 'Train: Perceptron')
    with col2:
        plot_confusion_matrix(y_test, y_test_pred, 'Test: Perceptron')
    
    # График обучения
    st.subheader("График обучения")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    ax2.legend()
    
    st.pyplot(fig, clear_figure=True)
    
    # Сохраняем модель и кодировщик
    st.session_state.perceptron_model = model
    st.session_state.label_encoder_perceptron = le
    
    return model, metrics_df, y_train_pred, y_test_pred

# Функции ансамблирования
def create_voting_ensemble(models, X_train, X_test, y_train, y_test):
    """Создает ансамбль методом голосования"""
    voting = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='hard'
    )
    return evaluate_model(voting, X_train, X_test, y_train, y_test, "Voting Ensemble")

def create_stacking_ensemble(models, X_train, X_test, y_train, y_test):
    """Создает ансамбль методом стэкинга"""
    stacking = StackingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    return evaluate_model(stacking, X_train, X_test, y_train, y_test, "Stacking Ensemble")

# Интерфейс для ансамблирования
def ensemble_interface(X_train, X_test, y_train, y_test):
    st.title("Ансамбли моделей")
    
    if 'models' not in st.session_state or not st.session_state.models:
        st.warning("Сначала постройте хотя бы одну модель")
        return
    
    available_models = st.session_state.models
    selected_models = st.multiselect(
        "Выберите модели для ансамбля",
        options=list(available_models.keys()),
        default=list(available_models.keys())
    )
    
    if len(selected_models) < 2:
        st.warning("Для ансамбля нужно выбрать хотя бы 2 модели")
        return
    
    ensemble_type = st.selectbox(
        "Тип ансамбля",
        ["Голосование", "Стэкинг"]
    )
    
    if st.button("Построить ансамбль"):
        models_to_ensemble = {name: available_models[name] for name in selected_models}
        
        with st.spinner("Строим ансамбль..."):
            if ensemble_type == "Голосование":
                model, metrics, _, _ = create_voting_ensemble(
                    models_to_ensemble, X_train, X_test, y_train, y_test
                )
            else:
                model, metrics, _, _ = create_stacking_ensemble(
                    models_to_ensemble, X_train, X_test, y_train, y_test
                )
            
            st.session_state.ensemble_model = model
            st.success("Ансамбль успешно построен!")
            
            st.subheader("Метрики ансамбля")
            st.dataframe(metrics)

# Интерфейс для предсказаний на новых данных
def prediction_interface():
    st.title("Предсказание на новых данных")
    
    uploaded_file = st.file_uploader("Загрузите новые данные (CSV или Excel)", 
                                   type=["csv", "xlsx"])
    
    if not uploaded_file:
        return
    
    # Загрузка данных
    try:
        if uploaded_file.name.endswith('.csv'):
            new_data = pd.read_csv(uploaded_file, sep=st.session_state.sep_sign, 
                                 decimal=st.session_state.decimal_sign)
        else:
            new_data = pd.read_excel(uploaded_file)
        
        st.success(f"Успешно загружено {len(new_data)} записей")
        st.dataframe(new_data.head())
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return
    
    # Предобработка новых данных
    try:
        X_new = process_new_data(new_data)
        if X_new is None:
            st.error("Не удалось обработать новые данные")
            return
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {str(e)}")
        return
    
    # Выбор моделей для предсказания
    available_models = {}
    if 'models' in st.session_state:
        available_models.update(st.session_state.models)
    if 'ensemble_model' in st.session_state:
        available_models['Ensemble'] = st.session_state.ensemble_model
    
    if not available_models:
        st.warning("Нет доступных моделей для предсказания")
        return
    
    selected_models = st.multiselect(
        "Выберите модели для предсказания",
        options=list(available_models.keys()),
        default=list(available_models.keys())
    )
    if st.button("Выполнить предсказания"):
        predictions = pd.DataFrame()
        
        for model_name in selected_models:
            model = available_models[model_name]
            
            try:
                # Особые случаи для моделей, требующих кодирования
                if model_name == 'XGBoost' and 'label_encoder_xgboost' in st.session_state:
                    le = st.session_state.label_encoder_xgboost
                    pred = le.inverse_transform(model.predict(X_new))
                elif model_name == 'Perceptron' and 'label_encoder_perceptron' in st.session_state:
                    le = st.session_state.label_encoder_perceptron
                    pred_proba = model.predict(X_new)
                    pred = le.inverse_transform(np.argmax(pred_proba, axis=1))
                else:
                    pred = model.predict(X_new)
                
                predictions[model_name] = pred
            except Exception as e:
                st.error(f"Ошибка при предсказании с помощью {model_name}: {str(e)}")
                continue
        
        # Голосование, если выбрано несколько моделей
        if len(selected_models) > 1:
            predictions['Majority_Vote'] = predictions.mode(axis=1)[0]
        
        st.subheader("Результаты предсказаний")
        st.dataframe(predictions)
        
        # Кнопка скачивания
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Скачать предсказания",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

# Основной интерфейс моделей
def models_interface(X_train, X_test, y_train, y_test):
    st.title("Модели машинного обучения")
    
    use_cv = st.sidebar.checkbox("Использовать кросс-валидацию", value=False)
    cv_folds = st.sidebar.slider("Количество фолдов", 2, 10, 5) if use_cv else 5
    
    # Выбор моделей через табы
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Logistic Regression", "Decision Tree", "Random Forest", 
        "XGBoost", "KNN", "SVM", "Perceptron"
    ])
    
    with tab1:
        st.subheader("Логистическая регрессия")
        if st.button("Обучить Logistic Regression"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = logistic_regression(
                    X_train, X_test, y_train, y_test, use_cv, cv_folds
                )
                st.dataframe(metrics)
    
    with tab2:
        st.subheader("Дерево решений")
        max_depth = st.number_input("max_depth", 1, 50, 10)
        min_samples = st.number_input("min_samples_split", 2, 20, 10)
        if st.button("Обучить Decision Tree"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = decision_tree(
                    X_train, X_test, y_train, y_test, 
                    max_depth, min_samples, use_cv, cv_folds
                )
                st.dataframe(metrics)
    
    with tab3:
        st.subheader("Случайный лес")
        n_estimators = st.number_input("n_estimators", 10, 500, 50)
        max_depth = st.number_input("max_depth", 1, 50, 10)
        min_samples = st.number_input("min_samples_split", 2, 20, 10)
        if st.button("Обучить Random Forest"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = random_forest(
                    X_train, X_test, y_train, y_test,
                    n_estimators, max_depth, min_samples, use_cv, cv_folds
                )
                st.dataframe(metrics)
    
    with tab4:
        st.subheader("XGBoost")
        learning_rate = st.number_input("learning_rate", 0.001, 1.0, 0.01)
        n_estimators = st.number_input("n_estimators", 10, 500, 50)
        max_depth = st.number_input("max_depth", 1, 20, 5)
        if st.button("Обучить XGBoost"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = xgboost_model(
                    X_train, X_test, y_train, y_test,
                    learning_rate, n_estimators, max_depth, use_cv, cv_folds
                )
                st.dataframe(metrics)
    
    with tab5:
        st.subheader("KNN")
        n_neighbors = st.number_input("n_neighbors", 1, 50, 5)
        if st.button("Обучить KNN"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = knn_classifier(
                    X_train, X_test, y_train, y_test,
                    n_neighbors, use_cv, cv_folds
                )
                st.dataframe(metrics)
    
    with tab6:
        st.subheader("Метод опорных векторов")
        if st.button("Обучить SVM"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = svm_classifier(
                    X_train, X_test, y_train, y_test,
                    use_cv, cv_folds
                )
                st.dataframe(metrics)
    
    with tab7:
        st.subheader("Перцептрон")
        layers = st.number_input("Количество слоев", 1, 5, 2)
        neurons = st.number_input("Нейронов в слое", 16, 256, 64)
        learning_rate = st.number_input("learning_rate", 0.0001, 1.0, 0.01)
        epochs = st.number_input("Количество эпох", 1, 100, 10)
        if st.button("Обучить Perceptron"):
            with st.spinner("Обучение модели..."):
                model, metrics, _, _ = perceptron_classifier(
                    X_train, X_test, y_train, y_test,
                    layers, neurons, learning_rate, epochs, use_cv, cv_folds
                )
                st.dataframe(metrics)
# ======== Models row ============
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


