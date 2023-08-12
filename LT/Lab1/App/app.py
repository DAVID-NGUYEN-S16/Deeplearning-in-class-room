import streamlit as st
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import pandas as pd

# Load mô hình đã lưuD:\Deeplearning\LT\Lab1\App\weight_best
with open('weight_best/best_model_logistic_regression.pkl', 'rb') as file:
    loaded_model = joblib.load(file)
# "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
# Tiêu đề ứng dụng
st.title('Dự Đoán Kết Quả')

# Thêm mô tả và hướng dẫn
st.write('Nhập dữ liệu của bạn vào <>')

# Thêm thanh trượt cho các biến đầu vào
feature1 = st.slider('Sepal length', min_value=0, max_value=10, value=5)
feature2 = st.slider('Sepal width', min_value=0, max_value=10, value=5)
feature3 = st.slider('Petal length', min_value=0, max_value=10, value=5)
feature4 = st.slider('Petal width', min_value=0, max_value=10, value=5)
# Thêm các thanh trượt cho các biến khác (nếu cần)

# Gộp các biến đầu vào để dự đoán
input_data = np.array([[feature1, feature2, feature3, feature4]])  # Chú ý việc sử dụng mảng lồng nhau
# input_data = pd.
# Dự đoán kết quả và hiển thị
cag = [
    'Iris-setosa' ,
    'Iris-versicolor',
    'Iris-virginica'
]
# Thêm tiện ích để tải tệp tin
uploaded_file = st.file_uploader('Tải lên tệp tin dữ liệu Lưu ý dữ liệu chỉ có 4 cột  là sepal_length, sepal_width,	petal_length, petal_width(.csv)', type=['csv'])
# Dữ liệu
data = {'sepal_length': [5.1, 4.9, 4.7, 4.6],
        'sepal_width': [3.5, 3.0, 3.2, 3.1],
        'petal_length': [1.4, 1.4, 1.3, 1.5],
        'petal_width': [0.2, 0.2, 0.2, 0.2]}

# Tạo DataFrame từ dữ liệu
data = pd.DataFrame(data)
st.write(f'Ví dụ:')
st.write(data)

if uploaded_file is not None and st.button('Dự đoán file'):
    data = pd.read_csv(uploaded_file)
    # data = data[['sepal_length',	'sepal_width',	'petal_length'	,'petal_width']]
    predictions = loaded_model.predict(data)
    st.write('Kết quả dự đoán:')
    data['Species predict'] = predictions
    data['Species predict'] = data['Species predict'].apply(lambda x: cag[x])
    # result = pd.concat([data, df2], axis=1)

    st.write(data)
if st.button('Dự đoán'):
    prediction = loaded_model.predict(input_data)
    st.write('Kết quả dự đoán:')
    # print(prediction)
    st.write(cag[prediction[0]])