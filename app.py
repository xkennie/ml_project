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
from ydata_profiling import ProfileReport
import inspect
from collections import Counter
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
st.title("Проект по анализу данных. No-code для создания ансамбля")
st.write("Команда проекта: некий Вадим, некий Константин, некий Андрей, некая Таня")

st.title("Загрузи файл с пивой сюда, друг")
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  d = ProfileReport(data)
  st.write("Твой EDA, брат")
  print(d)

#чтение данных
def read_data():
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
  for column in ['Size(L)',
 'OG',
 'FG',
 'ABV',
 'IBU',
 'Color',
 'BoilSize',
 'BoilGravity',
  'Efficiency',
               'MashThickness']:
  #print(column)
    df[column] = df[column].str.replace(",", ".")
    df[column] = df[column].astype(float)
    df[column] = df[column].fillna(0)
  return df
#обработка данных
def preprocess(df):
  df = df.dropna()
#df = df[:5000]
  top_50 = df.groupby("Style", as_index = False).agg({'StyleID': 'count'}).sort_values(by = 'StyleID', ascending = False).head(50)["Style"].tolist()
  df = df[df["Style"].isin(top_50)]
  X = df.drop(columns=['Style', 'StyleID'])
  y = df['Style']

#кодирование целевой переменной
#label_encoder = LabelEncoder()
#y_encoded = label_encoder.fit_transform(y)
#y_categorical = to_categorical(y_encoded)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
  return X_train, X_test, y_train, y_test

#функции с реализациями методов ML
df = read_data()
#logreg
def logistic_regression():
  X_train, X_test, y_train, y_test = preprocess(df)
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
def tree(max_depth_target, min_samples_split_target):
  X_train, X_test, y_train, y_test = preprocess(df)
  tree = DecisionTreeClassifier(
    max_depth = max_depth_target,             # Максимальная глубина
    min_samples_split = min_samples_split_target,     # Минимальное число образцов для разделения
)
  tree.fit(X_train, y_train)
  y_pred = tree.predict(X_test)
  predict = pd.Series(y_pred)
  tree_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  #tree_predicts = tree_predicts.replace({0: 'predict'})
  tree_predicts.columns = ['index', 'Style', 'Tree_predict']
  return tree_predicts

#forest
def random_forest(estimators_target):
  X_train, X_test, y_train, y_test = preprocess(df)
  random_forest = RandomForestClassifier(
    n_estimators = estimators_target,  # Число деревьев
    max_features='sqrt',  
)
  random_forest.fit(X_train, y_train)
  y_pred = random_forest.predict(X_test)
  predict = pd.Series(y_pred)
  random_forest_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  random_forest_predicts.columns = ['index', 'Style', 'Random_Forest_predict']
  return random_forest_predicts

#xgboost
def xgboost(learning_rate_target, estimators_target, max_depth_target):
  X_train, X_test, y_train, y_test = preprocess(df)
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
def svc():
  X_train, X_test, y_train, y_test = preprocess(df)
  svc = SVC()
  svc.fit(X_train, y_train)
  y_pred = svc.predict(X_test)
  predict = pd.Series(y_pred)
  svc_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  svc_predicts.columns = ['index', 'Style', 'SVC_predict']
  return svc_predicts
#knn
def knn_classifier(neighbors_target):
  X_train, X_test, y_train, y_test = preprocess(df)
  knn = KNeighborsClassifier(n_neighbors= neighbors_target)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  predict = pd.Series(y_pred)
  knn_predicts = pd.concat([y_test.reset_index(), predict], axis = 1)
  knn_predicts.columns = ['index', 'Style', 'KNN_predict']
  return knn_predicts

def perceptron_classifier(layers_target, neurons_target, learning_rate_target = 0.01,
                          epochs_target = 10, activation_function_target = 'relu'):
  X_train, X_test, y_train, y_test = preprocess(df)
  #y_train encode
  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y_train)
  y_train = to_categorical(y_encoded)  
  #y_test encode
  y_encoded = label_encoder.fit_transform(y_test)
  y_test = to_categorical(y_encoded)

  perceptron = Sequential()
  for n in range(layers_target):
    perceptron.add(Dense(neurons_target, input_dim = X_train.shape[1], activation = activation_function_target))  # Скрытые слои
  
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
  perceptron_predicts
  perceptron_predicts.columns = ['Style', 'Perceptron_predict']
  return perceptron_predicts

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
    st.subheader(f"Параметры для {name}")
    params = {}
    for param in sig.parameters.values():
        default_val = "" if param.default is param.empty else param.default
        input_val = st.text_input(f"{name} - {param.name}", value=str(default_val))
        params[param.name] = eval(input_val)
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
