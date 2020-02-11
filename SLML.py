import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn import metrics

from sklearn.pipeline import Pipeline
import scikitplot.plotters as skplt 
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall
from scikitplot.metrics import plot_calibration_curve

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt
import seaborn as sns



#######################
st.sidebar.title(">>> Litbrench")
st.sidebar.markdown("**A workbench for machine learning created by Streamlit**")

#######################




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
        ML_option = st.sidebar.radio("Choose a regressor", ("Linear Regression", "KNN Regression", "Decision Tree Regressor","Random Forest Regressor", "Bayesian Ridge Regression", "Support Vector Regression", "Pipeline Regression"))
    else:
        ML_option = st.sidebar.radio("Choose a classifier", ("Logistic Regression", "KNN Classifier", "Decision Tree Classifier", "Random Forest Classifier", "Linear Discriminant Analysis", "Naive Bayes","Support Vector Classifier","Pipeline"))
else:
    st.sidebar.title("Under construction - coming soon.")
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

# Function to label encoder
def GetEncoder(dataframe, colunas):
    le = LabelEncoder()
    for col in colunas:
        dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe

# Show CSV head
try:
    data_csv = GetFile()
    st.write(data_csv.head())
except:
    st.write("You must upload a dataset.")


# Label encoder
try:
    if st.checkbox("Label encoder"):
        columns_encoder = st.multiselect("Choose columns to encoder: ", data_csv.columns)
        data_csv = GetEncoder(data_csv, columns_encoder)        
except:
    pass


# Choose features and target
st.subheader("Features and Target")
try:
    X_features = st.multiselect("Features: ", data_csv.columns)
    y_target = st.multiselect("Target: ", data_csv.columns)
except:
    pass


try:
    # Get features
    data_feature = GetFeatures(data_csv, X_features)
    # Features normalization
    if st.checkbox("Normalize features"):
        scaler = StandardScaler()
        scaler.fit(data_feature)
        df_scaled = scaler.transform(data_feature)
        data_feature = pd.DataFrame(df_scaled, columns=data_feature.columns)
    # Get target
    data_target = GetTarget(data_csv, y_target)
    st.write("Features", data_feature.head(), "Target",data_target.head())
except:
    st.write("Feature and Target cannot be loaded.")


# Train Test Split
lr_ts = st.number_input("Test size: ")
lr_rs = st.number_input("Random state: ", min_value=1, step=1)
try:
    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_target, test_size=lr_ts, random_state=int(lr_rs))
except:
    st.write("Data cannot be splitted.")
########################################



########################################
# LINEAR REGRESSION
########################################
if ML_option == "Linear Regression":
    # Fit the model and predict X_test. Show some analysis.
    try:
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        pred = linReg.predict(X_test)
        st.write("R2 Score: ", round(r2_score(y_test, pred),4))
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        
        # Show features coefficient
        coeff_df = pd.DataFrame(linReg.coef_,index=["Coefficient"] ,columns=[data_feature.columns])
        st.write(coeff_df)

        # Learning curve
        skplt.plot_learning_curve(linReg, X_train, y_train)
        st.pyplot()
    
        # Scatter Plot
        plt.scatter(y_test,pred)
        plt.title("Scatter Plot")
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()
    
        # Distribuition Plot
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-pred),bins=int(ibins))
        plt.title("Distribution Plot")
        plt.xlabel("Target")
        st.pyplot()
    except:
        st.write("Fill all parameters.")


########################################
# KNN REGRESSION
########################################
if ML_option == "KNN Regression":
    # Fit the model and predict X_test. Show some analysis.
    try:
        st.subheader("KNN Parameters")
        Neigh = st.number_input("Number of neighbors: ", min_value=1, step=1)
        KNNReg = KNeighborsRegressor(n_neighbors=Neigh)
        KNNReg.fit(X_train, y_train)
        pred = KNNReg.predict(X_test)
        st.write("R2 Score: ", round(r2_score(y_test, pred),4))
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
    
        # Learning curve
        skplt.plot_learning_curve(KNNReg, X_train, y_train)
        st.pyplot()
    
        # Scatter Plot
        plt.scatter(y_test,pred)
        plt.title("Scatter Plot")
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()

        # Distribuition Plot
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-pred),bins=int(ibins))
        plt.title("Distribution Plot")
        plt.xlabel("Target")
        st.pyplot()
    except:
        st.write("Fill all parameters.")


########################################
# DECISION TREE REGRESSOR
########################################
if ML_option == "Decision Tree Regressor":
    # Fit the model and predict X_test. Show some analysis.
    try:
        DTreeReg = DecisionTreeRegressor()
        DTreeReg.fit(X_train, y_train)
        pred = DTreeReg.predict(X_test)
        st.write("R2 Score: ", round(r2_score(y_test, pred),4))
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))

        # Learning curve
        skplt.plot_learning_curve(DTreeReg, X_train, y_train)
        st.pyplot()

        # Scatter Plot
        plt.scatter(y_test,pred)
        plt.title("Scatter Plot")
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()

        # Distribuition Plot
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-[pred]),bins=int(ibins))
        plt.title("Distribution Plot")
        plt.xlabel("Target")
        st.pyplot()
    except:
        st.write("Fill all parameters.")


########################################
# RANDOM FOREST REGRESSION
########################################
if ML_option == "Random Forest Regressor":
    # Fit the model and predict X_test. Show some analysis.
    try:
        st.subheader("Random Forest Parameters")
        Nestim = st.number_input("Number of estimators: ", min_value=1, step=1)
        RanStaFor = st.number_input("Random state for random forest model: ", min_value=1, step=1)
        RForest = RandomForestRegressor(n_estimators=Nestim, random_state=RanStaFor)
        RForest.fit(X_train, y_train)
        pred = RForest.predict(X_test)
        st.write("R2 Score: ", round(r2_score(y_test, pred),4))
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))

        # Learning curve
        skplt.plot_learning_curve(RForest, X_train, y_train)
        st.pyplot()
    
        # Scatter Plot
        plt.scatter(y_test,pred)
        plt.title("Scatter Plot")
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()

        # Distribuition Plot
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-[pred]),bins=int(ibins))
        plt.title("Distribution Plot")
        plt.xlabel("Target")
        st.pyplot()
    except:
        st.write("Fill all parameters.")


########################################
# BAYESIAN RIDGE REGRESSION
########################################
if ML_option == "Bayesian Ridge Regression":
    # Fit the model and predict X_test. Show some analysis.
    try:
        st.subheader("Bayesian Ridge Parameters")
        Niter = st.number_input("Number of iterations: ", min_value=1, step=1)
        BayRReg = MultiOutputRegressor(BayesianRidge(n_iter=Niter))
        BayRReg.fit(X_train, y_train)
        pred = BayRReg.predict(X_test)
        st.write("R2 Score: ", round(r2_score(y_test, pred),4))
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))

        # Learning curve
        skplt.plot_learning_curve(BayRReg, X_train, y_train)
        st.pyplot()
    
        # Scatter Plot
        plt.scatter(y_test,pred)
        plt.title("Scatter Plot")
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()

        # Distribuition Plot
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-[pred]),bins=int(ibins))
        plt.title("Distribution Plot")
        plt.xlabel("Target")
        st.pyplot()
    except:
        st.write("Fill all parameters.")


########################################
# SUPPORT VECTOR REGRESSION
########################################
if ML_option == "Support Vector Regression":
    # Fit the model and predict X_test. Show some analysis.
    try:
        st.subheader("Support Vector Parameters")
        Ngamma = st.number_input("gamma: ", min_value=0.01, step=0.01)
        Cvalue = st.number_input("C: ", min_value=0.1, step=0.1)
        Evalue = st.number_input("epsilon: ", min_value=0.1, step=0.1)
        SVReg = MultiOutputRegressor(SVR(gamma=Ngamma, C=Cvalue, epsilon=Evalue))
        SVReg.fit(X_train, y_train)
        pred = SVReg.predict(X_test)
        st.write("R2 Score: ", round(r2_score(y_test, pred),4))
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))

        # Learning curve
        skplt.plot_learning_curve(SVReg, X_train, y_train)
        st.pyplot()

        # Scatter Plot
        plt.scatter(y_test,pred)
        plt.title("Scatter Plot")
        plt.xlabel("Real")
        plt.ylabel("Predictions")
        st.pyplot()

        # Distribuition Plot
        ibins = st.number_input("bins: ",min_value=1,step=1)
        sns.distplot((y_test-[pred]),bins=int(ibins))
        plt.title("Distribution Plot")
        plt.xlabel("Target")
        st.pyplot()
    except:
        st.write("Fill all parameters.")
    

########################################
# PIPELINE REGRESSION
########################################
if ML_option == "Pipeline Regression":
    try:
        pipe_lr = Pipeline( [ ('scl', StandardScaler()), ('clf', LinearRegression()) ] )
        pipe_knn = Pipeline([('scl', StandardScaler()), ('clf', KNeighborsRegressor())])
        pipe_dt = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeRegressor())])
        pipe_rf = Pipeline([('scl', StandardScaler()),('clf', RandomForestRegressor())])
        pipe_br = Pipeline([('scl', StandardScaler()),('clf', MultiOutputRegressor(BayesianRidge()))])
        pipe_sv = Pipeline([("scl", StandardScaler()),('clf', MultiOutputRegressor(SVR()))])
        
        pipelines = [pipe_lr, pipe_knn, pipe_dt, pipe_rf, pipe_br, pipe_sv]

        pipe_dict = {0: "Linear Regression", 1: 'KNN Regression', 2: 'Decision Tree Regressor', 3: 'Random Forest Regressor', 4: 'Bayesian Ridge Regression',
                5: "Support Vector Regression"}

        for pipe in pipelines:
            pipe.fit(X_train, y_train)

        for idx, val in enumerate(pipelines):
            st.write('%s pipeline test accuracy: ' % (pipe_dict[idx]), round(val.score(X_test, y_test),4))
        # para cada modelo treinado obtem val score
        best_acc = 0.0
        best_clf = 0
        best_pipe = ''

        for idx, val in enumerate(pipelines):
        # Descobre o melhor val.score e armazen em best_clf
            if val.score(X_test, y_test) > best_acc:
                best_acc = val.score(X_test, y_test)
                best_pipe = val
                best_clf = idx
        st.write('\n')        
        st.subheader('Regressor with best accuracy: %s' % pipe_dict[best_clf])    
    except:
        pass











##########################################################
##########################################################
# CLASSIFICATION

########################################
# LOGISTIC REGRESSION
########################################
if ML_option == "Logistic Regression":
    # Fit the model and predict X_test. Show some analysis.

    try:
        logReg = MultiOutputClassifier(LogisticRegression())
        logReg.fit(X_train, y_train)
        pred = logReg.predict(X_test)
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of Logistic Regression on training set: ', round(logReg.score(X_train, y_train),4))
        st.write('Accuracy of Logistic Regression  on test set: ', round(logReg.score(X_test, y_test),4))

        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

        try:
            # Confusion matrix
            plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
            bottom,top = plt.ylim()
            plt.ylim(bottom+0.5,top-0.5)
            st.pyplot()
        except:
            st.write("Confusion matrix do not support multioutput.")
        

    except:
        st.write("Fill all parameters.")

    # plot_calibration_curve(y_test, [pred])
    # st.pyplot()

########################################
# KNN CLASSIFIER
########################################        
if ML_option == "KNN Classifier":
    # Fit the model and predict X_test. Show some analysis.
    try:
        st.subheader("KNN Parameters")
        Neigh = st.number_input("Number of neighbors: ", min_value=1, step=1)
        KNNCla = KNeighborsClassifier(n_neighbors=Neigh)
        KNNCla.fit(X_train, y_train)
        pred = KNNCla.predict(X_test)
        st.write("teste")
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of KNN Classifier on training set: ', round(KNNCla.score(X_train, y_train),4))
        st.write('Accuracy of KNN Classifier on test set: ', round(KNNCla.score(X_test, y_test),4))
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

        #Confusion matrix
        plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
        bottom,top = plt.ylim()
        plt.ylim(bottom+0.5,top-0.5)
        st.pyplot()
    except:
        st.write("Fill all parameters.")

    # plot_calibration_curve(y_test, [pred])
    # st.pyplot()        

########################################
# DECISION TREE CLASSIFIER
########################################  
if ML_option == "Decision Tree Classifier":
    try:
        DTreeCla = DecisionTreeRegressor()
        DTreeCla.fit(X_train, y_train)
        pred = DTreeCla.predict(X_test)
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of Decision Tree Classifier on training set: ', round(DTreeCla.score(X_train, y_train),4))
        st.write('Accuracy of Decision Tree Classifier on test set: ', round(DTreeCla.score(X_test, y_test),4))
            
        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

            #Confusion matrix
        plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
        bottom,top = plt.ylim()
        plt.ylim(bottom+0.5,top-0.5)
        st.pyplot()
    except:
        st.write("Fill all parameters.")

########################################
# RANDOM FOREST CLASSIFIER
######################################## 
if ML_option == "Random Forest Classifier":
    # Fit the model and predict X_test. Show some analysis.
    try:
        st.subheader("Random Forest Parameters")
        Nestim = st.number_input("Number of estimators: ", min_value=1, step=1)
        RanStaFor = st.number_input("Random state for random forest model: ", min_value=1, step=1)
        RForest = RandomForestClassifier(n_estimators=Nestim, random_state=RanStaFor)
        RForest.fit(X_train, y_train)
        pred = RForest.predict(X_test)
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of Decision Tree Classifier on training set: ', round(RForest.score(X_train, y_train),4))
        st.write('Accuracy of Decision Tree Classifier on test set: ', round(RForest.score(X_test, y_test),4))
            
        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

            #Confusion matrix
        plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
        bottom,top = plt.ylim()
        plt.ylim(bottom+0.5,top-0.5)
        st.pyplot()

    except:
        st.write("Fill all parameters.")

########################################
# LINEAR DISCRIMINANT CLASSIFIER
######################################## 
if ML_option == "Linear Discriminant Analysis":
    # Fit the model and predict X_test. Show some analysis.
    try:
        lda = MultiOutputClassifier(LinearDiscriminantAnalysis())
        lda.fit(X_train, y_train)
        pred = lda.predict(X_test)
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of Decision Tree Classifier on training set: ', round(lda.score(X_train, y_train),4))
        st.write('Accuracy of Decision Tree Classifier on test set: ', round(lda.score(X_test, y_test),4))
            
        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

            #Confusion matrix
        plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
        bottom,top = plt.ylim()
        plt.ylim(bottom+0.5,top-0.5)
        st.pyplot()

    except:
        st.write("Fill all parameters.")

########################################
# NAIVE BAYES CLASSIFIER
######################################## 
if ML_option == "Naive Bayes":
    # Fit the model and predict X_test. Show some analysis.
    try:
        gnb = MultiOutputClassifier(GaussianNB())
        gnb.fit(X_train, y_train)
        pred = gnb.predict(X_test)
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of Decision Tree Classifier on training set: ', round(gnb.score(X_train, y_train),4))
        st.write('Accuracy of Decision Tree Classifier on test set: ', round(gnb.score(X_test, y_test),4))
            
        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

            #Confusion matrix
        plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
        bottom,top = plt.ylim()
        plt.ylim(bottom+0.5,top-0.5)
        st.pyplot()

    except:
        st.write("Fill all parameters.")

########################################
# SUPPORT VECTOR CLASSIFIER
######################################## 
if ML_option == "Support Vector Classifier":
    # Fit the model and predict X_test. Show some analysis.
    try:
        svm = MultiOutputClassifier(SVC())
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        st.write('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, pred),4))
        st.write('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, pred),4))
        st.write('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, pred)),4))
        st.write('Accuracy of Decision Tree Classifier on training set: ', round(svm.score(X_train, y_train),4))
        st.write('Accuracy of Decision Tree Classifier on test set: ', round(svm.score(X_test, y_test),4))
            
        st.subheader("Classification Report")
        st.text(classification_report(y_test,pred))

            #Confusion matrix
        plot_confusion_matrix(y_test,pred, figsize=(7,5), cmap="PuBuGn")
        bottom,top = plt.ylim()
        plt.ylim(bottom+0.5,top-0.5)
        st.pyplot()

    except:
        st.write("Fill all parameters.")

if ML_option == "Pipeline":
    try:
        pipe_lr = Pipeline( [ ('scl', StandardScaler()), ('clf', MultiOutputClassifier(LogisticRegression()))])
        pipe_knn = Pipeline([('scl', StandardScaler()), ('clf', KNeighborsClassifier())])
        pipe_dt = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier())])
        pipe_rand = Pipeline([('scl', StandardScaler()),('clf', RandomForestClassifier())])
        pipe_lda = Pipeline([('scl', StandardScaler()),('clf', MultiOutputClassifier(LinearDiscriminantAnalysis()))])
        pipe_gnb = Pipeline([("scl", StandardScaler()),('clf', MultiOutputClassifier(GaussianNB()))])
        pipe_SVM = Pipeline([("scl", StandardScaler()), ('clf', MultiOutputClassifier(SVC()))])

        pipelines = [pipe_lr, pipe_knn, pipe_dt, pipe_rand, pipe_lda, pipe_gnb, pipe_SVM]

        pipe_dict = {0: "Logistic Regression", 1: 'KNN Classifier', 2: 'Decision Tree Classifier', 3: 'Random Forest Classifier', 4: 'Linear Discriminant Analysis',
                5: "Naive Bayes", 6: "Support Vector Classifier"}

        for pipe in pipelines:
            pipe.fit(X_train, y_train)

        for idx, val in enumerate(pipelines):
            st.write('%s pipeline test accuracy: ' % (pipe_dict[idx]), round(val.score(X_test, y_test),4))
        # para cada modelo treinado obtem val score
        best_acc = 0.0
        best_clf = 0
        best_pipe = ''

        for idx, val in enumerate(pipelines):
        # Descobre o melhor val.score e armazen em best_clf
            if val.score(X_test, y_test) > best_acc:
                best_acc = val.score(X_test, y_test)
                best_pipe = val
                best_clf = idx
        st.write('\n')        
        st.subheader('Classifier with best accuracy: %s' % pipe_dict[best_clf])    
    except:
        pass


















###############################################

st.sidebar.markdown("**Help us to improve this application. See the source code below. Follow us!**")
st.sidebar.markdown("[Source code](https://github.com/rafaelloni/ML_Server_Models)")
st.sidebar.markdown("**Contributors:**")
st.sidebar.markdown("[Rafael Loni](https://github.com/rafaelloni)")
st.sidebar.markdown("[Gabriel Filetti](https://github.com/GabrielFiletti)")