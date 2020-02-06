import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


### MENU VARIABLES ###
super_unsuper = ""
super_regre_class = ""
ML_option = ""



########################################
# Machine Learning Algorithms Menu
st.sidebar.title("Machine Learning Models")
super_unsuper = st.sidebar.radio("Choose a model", ("Supervised Learning", "Unsupervised Learning"))
if super_unsuper == "Supervised Learning":
    super_regre_class = st.sidebar.radio("Choose a type", ("Regression", "Classification"))
    if super_regre_class == "Regression":
        ML_option = st.sidebar.radio("", ("Linear Regression", "KNN Regression", "Decision Tree Regression", "Random Forest", "Naive Bayes", "Support Vector Regression"))
    else:
        ML_option = st.sidebar.radio("", ("Logistic Regression", "KNN Classifier", "Decision Tree Classifier", "Random Forest", "Linear Discriminant Analysis", "Naive Bayes","Support Vector Classifier"))
        pass
else:
    pass
########################################



########################################
# Title
st.title("Machine Learning Analysis")

# Function to get csv file
def GetFile():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data

# Function to get features
def GetFeatures(dataframe, colunas):
    df_X = dataframe[[colunas[0]]]
    for i in range(1,len(colunas)):
        df_X = df_X.join(dataframe[[colunas[i]]])
    return df_X

# Function to get target
def GetTarget(dataframe, colunas):
    df_X = dataframe[[colunas[0]]]
    for i in range(1,len(colunas)):
        df_X = df_X.join(dataframe[[colunas[i]]])
    return df_X

# Show CSV head
try:
    data_csv = GetFile()
    st.write(data_csv.head())
except:
    st.write("Adicione um dataset.")


try:
    X_features = st.multiselect("Features: ", data_csv.columns)
    y_target = st.multiselect("Target: ", data_csv.columns)
except:
    pass

try:
    data_feature = GetFeatures(data_csv, X_features)
    data_target = GetTarget(data_csv, y_target)
    st.write("Features", data_feature.head(), "Target",data_target.head())
except:
    st.write("Não foi possível carregar feature e target.")

# Train Test Split
lr_ts = st.number_input("Test size: ")
lr_rs = st.number_input("Random state: ", min_value=1, step=1)
try:
    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=lr_ts, random_state=int(lr_rs))
except:
    st.write("Não possivel separar os dados")
########################################



########################################
# LINEAR REGRESSION
if ML_option == "Linear Regression":
    # Fit the model and predict X_test. Show some analysis.
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

    # Show features coefficient
    try:
        coeff_df = pd.DataFrame(linReg.coef_,index=["Coefficient"] ,columns=[data_feature.columns])
        st.write(coeff_df)
    except:
        pass
    
    # Scatter Plot
    try:
        plt.scatter(y_test,pred)
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()
    except:
        pass

    # Distribuition Plot
    try:
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-pred),bins=int(ibins))
        plt.xlabel("Target")
        st.pyplot()
    except:
        pass


# KNN REGRESSION
if ML_option == "KNN Regression":
    # Fit the model and predict X_test. Show some analysis.
    try:
        Neigh = st.number_input("Number of neighbors: ", min_value=1, step=1)
        KNNReg = KNeighborsRegressor(n_neighbors=Neigh)
        KNNReg.fit(X_train, y_train)
        pred = KNNReg.predict(X_test)
        st.write("R2 Score: ", r2_score(y_test, pred))
        st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred))
        st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, pred))
        st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred)))
    except:
        st.write("Preencha todos os parâmetros")

    
    # Scatter Plot
    try:
        plt.scatter(y_test,pred)
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()
    except:
        pass

    # Distribuition Plot
    try:
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-pred),bins=int(ibins))
        plt.xlabel("Target")
        st.pyplot()
    except:
        pass


# KNN REGRESSION
if ML_option == "Random Forest":
    # Fit the model and predict X_test. Show some analysis.
    try:
        Neigh = st.number_input("Number of neighbors: ", min_value=1, step=1)
        KNNReg = KNeighborsRegressor(n_neighbors=Neigh)
        KNNReg.fit(X_train, y_train)
        pred = KNNReg.predict(X_test)
        st.write("R2 Score: ", r2_score(y_test, pred))
        st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred))
        st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, pred))
        st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred)))
    except:
        st.write("Preencha todos os parâmetros")

    
    # Scatter Plot
    try:
        plt.scatter(y_test,pred)
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()
    except:
        pass

    # Distribuition Plot
    try:
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-pred),bins=int(ibins))
        plt.xlabel("Target")
        st.pyplot()
    except:
        pass

























########################################
# CLASSIFICATION

if ML_option == "Logistic Regression":
    # Fit the model and predict X_test. Show some analysis.

    try:
        logReg = LogisticRegression()
        logReg.fit(X_train, y_train)
        pred = logReg.predict(X_test)
        st.write('R2 Score: ', r2_score(y_test, pred))
        st.write('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred))
        st.write('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, pred))
        st.write('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred)))
        st.write('Accuracy of Logistic Regression on training set: ', logReg.score(X_train, y_train))
        st.write('Accuracy of Logistic Regression  on test set: ', logReg.score(X_test, y_test))
        st.write('Confusion Matrix:', confusion_matrix(y_test,pred))
        st.write("\n")
        st.subheader("Classificarion Report")
        st.text(classification_report(y_test,pred))
    except:
        st.write("Preencha todos os parâmetros")