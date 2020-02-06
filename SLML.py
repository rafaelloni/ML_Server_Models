import streamlit as st 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


super_unsuper = ""
super_regre_class = ""
ML_option = ""



########################################
# Machine Learning Algorithms
st.sidebar.title("Machine Learning")
super_unsuper = st.sidebar.radio("Choose a model", ("Supervised Learning", "Unsupervised Learning"))
if super_unsuper == "Supervised Learning":
    super_regre_class = st.sidebar.radio("Choose a type", ("Regression", "Classification"))
    if super_regre_class == "Regression":
        ML_option = st.sidebar.radio("", ("Linear Regression", "KNN Regression", "Random Forest", "Naive Bayes", "Support Vector Regression"))
    else:
        pass
else:
    pass






st.title("Put your title here.")

def TakeFile():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data

data_csv = TakeFile()
if st.checkbox("Show all data"):
    st.write(data_csv)






#### VARIABLES
# X = None
# y = None
# X_train = None
# X_test = None
# y_train = None
# y_test = None
# test_size = 0.3
# random_state = 101

########################################
# LINEAR REGRESSION
if ML_option == "Linear Regression":
    X_features = st.text_input("Features: ")
    y_target = st.text_input("Target: ")
    lr_ts = st.number_input("Test size: ")
    lr_rs = st.number_input("Random state: ")
    try:
        X = data_csv[["{}".format(X_features)]]
        y = data_csv[["{}".format(y_target)]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=lr_ts, random_state=int(lr_rs))
    except:
        pass
    try:
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        pred = linReg.predict(X_test)
        st.write(r2_score(y_test, pred))
    except:
        st.write("Preencha todos os parâmetros")
