import pandas as pd
import numpy as np
import ipaddress
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.utils import shuffle

def print_best_valued_feature(df):

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X = df.iloc[:,0:29]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(20,'Score'))  #print 10 best features
    
    #===========================================================================
    X = df.iloc[:,0:29]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.show()

def importdata(): 
    print("LDAP")
    LDAP = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\LDAP.csv")
    print("MSSQL")
    MSSQL = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\MSSQL.csv")
    print("NetBIOS")
    NetBIOS = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\NetBIOS.csv")
    print("Syn")
    Syn = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\Syn.csv")
    print("UDP")
    UDP = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\UDP.csv")
    print("UDPLag")
    UDPLag = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\UDPLag.csv")
    print("Portmap")
    Portmap = data_preprocessing("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\Portmap.csv")
    df = LDAP.append([MSSQL, NetBIOS, Syn, UDP, UDPLag, Portmap])
    #df.to_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\final_dataset_small_mix.csv", index = False)
    return df

def data_preprocessing(filepath):
    df = pd.read_csv(filepath)
    
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.sample(n = 30000) 
    
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    df.Label.loc[df.Label == "BENIGN"] = 0
    df.Label.loc[df.Label != 0] = 1
    
    df['Source_IP'] = df['Source_IP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
    df['Destination_IP'] = df['Destination_IP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
    
    #df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))   
    #df['Timestamp'] = pd.to_datetime(df['Timestamp'] )
    names = ['Destination_IP',  
             'Flow_Duration',
             'Source_IP',
             'Total_Length_of_Bwd_Packets',
             'Bwd_IAT_Mean',
             'Fwd_IAT_Mean',
             'Flow_IAT_Mean',
             'Destination_Port',
             'Bwd_Packet_Length_Mean',
             'Source_Port',
             'Average_Packet_Size',
             'Total_Backward_Packets',
             'Subflow_Bwd_Packets',
             'Fwd_Packet_Length_Mean',
             'Packet_Length_Mean',
             'Total_Fwd_Packets',
             'Subflow_Fwd_Packets',
             'Total_Length_of_Fwd_Packets',
             'Down_Up_Ratio',
             'Protocol',
             'Label']
    df = df[names]
    df = shuffle(df)
    
    return df

def splitdataset(balance_data): 
    # Separating the target variable 
    X = balance_data.iloc[:,0:20]  #independent columns
    Y = balance_data.iloc[:,-1] 
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)  
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))  
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
    #print("Report : ", classification_report(y_test, y_pred)) 

# Driver code 
def main():     
    # Building Phase 
    data = importdata() 
    
    print("<<< --- Class Label COUNTS --- >>>")
    print(data['Label'].value_counts())
    
    #print_best_valued_feature(data)  #to get top 2O features
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    
    #Decision Tree
    print("-----------Decision Tree----------")
    from sklearn.tree import DecisionTreeClassifier 
    #clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    #clf_gini.fit(X_train, y_train) 
    #y_pred_gini = clf_gini.predict(X_test)
    #cal_accuracy(y_test, y_pred_gini) 
      
    clf_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    clf_entropy.fit(X_train, y_train) 
    y_pred_entropy = clf_entropy.predict(X_test)
    cal_accuracy(y_test, y_pred_entropy) 
    
    #Naive Bayes
    print("-----------Naive Bayes----------")
    from sklearn.naive_bayes import GaussianNB 
    gnb = GaussianNB() 
    gnb.fit(X_train, y_train) 
    y_pred_nb = gnb.predict(X_test)
    cal_accuracy(y_test, y_pred_nb) 

    #Logistic Regression
    print("-----------Logistic Regression----------")
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(solver='lbfgs', random_state = 0)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    cal_accuracy(y_test, y_pred_lr) 
    
    #Support Vector Machine
    print("-----------Support Vector Machine----------")
    from sklearn.svm import SVC
    clf = SVC(kernel = 'poly', random_state = 0)
    clf.fit(X_train, y_train)
    y_pred_svc = clf.predict(X_test)
    cal_accuracy(y_test, y_pred_svc) 
    
    #K Nearest Neighbor
    print("-----------K Nearest Neighbor----------")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    cal_accuracy(y_test, y_pred_knn)
    
    #Random Forest
    print("-----------Random Forest----------")
    from sklearn.ensemble import RandomForestClassifier
    rndForest =RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    rndForest.fit(X_train,y_train)
    y_pred_rf = rndForest.predict(X_test)
    cal_accuracy(y_test, y_pred_rf)
   
    
# Calling main function 
if __name__=="__main__": 
    main() 
    








    





