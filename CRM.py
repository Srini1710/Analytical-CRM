#Libraries
import pandas as pd 
import pydotplus
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import preprocessing
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.tree import _tree

def importdata():
    attrition = pd.read_csv("D:\Sem 2\A-CRM\Attrition.csv", sep = ",", header = 0)
    le = preprocessing.LabelEncoder()
    attrition['Attrition_code'] = le.fit_transform(attrition["Attrition"])
    attrition['BusinessTravel_code'] = le.fit_transform(attrition['BusinessTravel'])
    attrition['Department_code'] = le.fit_transform(attrition['Department'])
    attrition['EducationField_code'] = le.fit_transform(attrition['EducationField'])
    attrition['Gender_code'] = le.fit_transform(attrition['Gender'])
    attrition['JobRole_code'] = le.fit_transform(attrition['JobRole'])
    attrition['MaritalStatus_code'] = le.fit_transform(attrition['MaritalStatus'])
    attrition['OverTime_code'] = le.fit_transform(attrition['OverTime'])
                                                    
    print(attrition.head())
    return attrition

def split_data(dataset):
    dat = dataset.values[:,1:12]
    print("dat=")
    print(dat)
    X = pd.DataFrame(data = dat, columns = ["Age","EnvironmentSatisfaction","JobLevel","JobInvolvement",
                    "MonthlyIncome", "OverTime","StockOptionLevel","TotalWorkingYears",
                    "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager"], index = None)
    print(X)
    Y = dataset.values[:,0]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, 
                                                        random_state = 0)
    print("X_train=")
    print(X_train)
    
    return X, Y, X_train, X_test, Y_train, Y_test

def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 0,max_depth=4, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini

def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 0, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy

def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred

def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))
    
def tree_to_code(tree, feature_names):

	'''
	Outputs a decision tree model as a Python function
	
	Parameters:
	-----------
	tree: decision tree model
		The decision tree to represent as a function
	feature_names: list
		The feature names of the dataset used for building the decision tree
	'''

	tree_ = tree.tree_
	feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]
	print ("def tree({}):".format(", ".join(feature_names)))

	def recurse(node, depth):
		indent = "  " * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			print ("{}if {} <= {}:".format(indent, name, threshold))
			recurse(tree_.children_left[node], depth + 1)
			print ("{}else:  # if {} > {}".format(indent, name, threshold))
			recurse(tree_.children_right[node], depth + 1)
		else:
			print ("{}return {}".format(indent, tree_.value[node]))

	recurse(0, 1)
    
def main(): 
      
    # Building Phase 
    data = importdata() 
    data = data[["Attrition_code","Age","EnvironmentSatisfaction","JobLevel","JobInvolvement",
                    "MonthlyIncome", "OverTime_code","StockOptionLevel","TotalWorkingYears",
                    "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager"]]
    X, Y, X_train, X_test, y_train, y_test = split_data(data) 
    #print(X)
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    cols = ["Age","EnvironmentSatisfaction","JobLevel","JobInvolvement",
                    "MonthlyIncome", "OverTime","StockOptionLevel","TotalWorkingYears",
                    "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager"];
    dot_data = export_graphviz(clf_entropy, feature_names = cols, filled=True, rounded = True)
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("attrition.pdf")
    Image(graph.create_png())
    
    tree_to_code(clf_gini, list(cols))

if __name__=="__main__": 
    main() 
    
    