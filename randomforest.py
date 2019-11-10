#Libraries

import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from IPython.display import Image

#Importing the data and Encoding the categorical variables
def importdata():
    attrition = pd.read_csv("D:\Sem 2\A-CRM\Attrition.csv", sep = ",", 
                            header = 0)
    
    attrition_original = attrition
    attrition_original.shape
    
    #Drop columns that are closely correlated
    attrition = attrition.drop(["JobLevel", "MonthlyIncome", 
                                "TotalWorkingYears", "YearsAtCompany",
                                "YearsInCurrentRole", "YearsWithCurrManager",
                                "DailyRate", "HourlyRate"], 
                                axis = 1)
    
    #check for missing or null value
    print(attrition.isnull().sum())
    
    #Label encoding
    le = preprocessing.LabelEncoder()    
    attrition['BusinessTravel_code'] = le.fit_transform(
                                       attrition['BusinessTravel'])
    attrition['Department_code'] = le.fit_transform(attrition['Department'])
    attrition['EducationField_code'] = le.fit_transform(
                                       attrition['EducationField'])
    attrition['Gender_code'] = le.fit_transform(attrition['Gender'])
    attrition['JobRole_code'] = le.fit_transform(attrition['JobRole'])
    attrition['MaritalStatus_code'] = le.fit_transform(
                                      attrition['MaritalStatus'])
    attrition['OverTime_code'] = le.fit_transform(attrition['OverTime'])
    attrition['Over18_code'] = le.fit_transform(attrition['Over18'])    
    attrition['Attrition_code'] = le.fit_transform(attrition["Attrition"])
    
    print("Encoding Done")
    attrition.head()
    
    #drop the non-encoded columns
    attrition_new = attrition.drop(["Attrition", "BusinessTravel",
                                    "Department", "EducationField", "Gender", 
                                    "JobRole", "MaritalStatus", "OverTime", 
                                    "Over18", "Education", "EmployeeCount", 
                                    "EmployeeNumber", "JobRole_code",
                                    "Over18_code", "Over18"], axis = 1)
    
    print(attrition_new.head())
    return attrition_new

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'
                .format(accuracy, misclass))
    plt.show()


#RandomForestClassifier Train and Test
def randomclf(X_train, Y_train, X_test, Y_test):
    
    #develop the RF model
    clf = RandomForestClassifier(n_estimators = 1000, 
                                 random_state = 12, n_jobs = -1)   
       
    #train the model
    clf.fit(X_train, Y_train)
    
    #test the model
    y_pred = clf.predict(X_test)
    
    for feature in (X_train.columns.values, clf.feature_importances_):
        print(feature)
    
    
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    print("Accuracy of Random Forest = ", accuracy)
    
    all_accuracies = cross_val_score(estimator = clf, X = X_train, y = Y_train,
                                     cv=10)
    
    print("10-fold CrossValidation(CV) scores : ", all_accuracies)
    print("Accuracy of the Model (Mean of 10-f CV) : ", all_accuracies.mean())
    
    #Confusion matrix
    from sklearn.metrics import confusion_matrix
 
    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix \n", cm)
    
    plot_confusion_matrix(cm, 
                      normalize    = False,
                      target_names = ['Yes', 'No'],
                      title        = "Confusion Matrix")
    
    #Random foster tree in an image saved to PDF
    estimator = clf.estimators_[1]
    
    #Get column names
    cols = X_train.columns.values
    
    from sklearn.tree import export_graphviz
    # Export as dot file
    tree = export_graphviz(estimator, 
                    feature_names = cols,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
    graph = pydotplus.graph_from_dot_data(tree) 
    graph.write_pdf("attrition_random.pdf")
    Image(graph.create_png())
    
    #Extract feature importance of variables
    feat_importances = pd.Series(clf.feature_importances_, index=cols)
    feat_importances.plot(kind='barh')          

#Split the Data into Training set and Testing set
def splitdata(dataset):
    
    #get column names
    cols = dataset.columns.values
    
    #get number of columns
    print("no of cols = ", len(dataset.columns))
    
    #independent and dependent variables
    X = pd.DataFrame(dataset.values[:, 0:21], columns = (cols[0:21] ))
    Y = dataset.values[:, 21]

    
    print("Old values of Attrition 1-Yes 0-No :", Counter(Y))    
    
    #over-sampling because of imbalanced data
    #train the model and apply it to the variables
    ada = SMOTE(sampling_strategy = "minority",
                random_state = 12)    
    X_new, Y_new = ada.fit_sample(X,Y)    
    X_new = pd.DataFrame(X_new,columns = (cols[0:21]) )
    
    
    print("No of rows after balancing = ", X_new.shape)
    print("New Dataset \n", X_new)
    print("New values of Attrition 1-Yes 0-No :", Counter(Y_new))
    
    #split the data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, 
                                                        train_size = 0.3,
                                            random_state = 12)
    return X_train, X_test, Y_train, Y_test

#Main Function
def main():
    data = importdata()
    
    X_train, X_test, Y_train, Y_test = splitdata(data)
    
    randomclf(X_train, Y_train, X_test, Y_test)

    
#Calling main Function
main()