import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


super_unsuper = ""
super_regre_class = ""
ML_option = ""



########################################
# Machine Learning Algorithms
st.sidebar.title("Machine Learning Models")
super_unsuper = st.sidebar.radio("Choose a model", ("Supervised Learning", "Unsupervised Learning"))
if super_unsuper == "Supervised Learning":
    super_regre_class = st.sidebar.radio("Choose a type", ("Regression", "Classification"))
    if super_regre_class == "Regression":
        ML_option = st.sidebar.radio("", ("Linear Regression", "KNN Regression", "Random Forest", "Naive Bayes", "Support Vector Regression"))
    else:
        pass
else:
    pass






st.title("Machine Learning Analysis")

def TakeFile():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data

def TakeFeatures(dataframe, colunas):
    df_X = dataframe[[colunas[0]]]
    for i in range(1,len(colunas)):
        df_X = df_X.join(dataframe[[colunas[i]]])
    return df_X

def TakeTarget(dataframe, colunas):
    df_X = dataframe[[colunas[0]]]
    for i in range(1,len(colunas)):
        df_X = df_X.join(dataframe[[colunas[i]]])
    return df_X

try:
    data_csv = TakeFile()
    st.write(data_csv.head())
except:
    st.write("Adicione um dataset.")

X_features = st.text_input("Features: ").split()
y_target = st.text_input("Target: ").split()
try:
    data_feature = TakeFeatures(data_csv, X_features)
    data_target = TakeTarget(data_csv, y_target)
    st.write("Features", data_feature.head(), "Target",data_target.head())
except:
    st.write("Não foi possível carregar feature e target.")

lr_ts = st.number_input("Test size: ")
lr_rs = st.number_input("Random state: ")
try:
    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=lr_ts, random_state=int(lr_rs))
except:
    st.write("Não possivel separar os dados")

########################################
# LINEAR REGRESSION
if ML_option == "Linear Regression":
    try:
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        pred = linReg.predict(X_test)
        st.write("R2 Score: ", r2_score(y_test, pred))
        st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred))
        st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, pred))
        st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred)))
    except:
        st.write("Preencha todos os parâmetros")

    try:
        coeff_df = pd.DataFrame(linReg.coef_,index=["Coefficient"] ,columns=[data_feature.columns])
        st.write(coeff_df)
    except:
        pass

    try:
        plt.scatter(y_test,pred)
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()
    except:
        pass
    try:
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-pred),bins=int(ibins))
        plt.xlabel("Target")
        st.pyplot()
    except:
        pass