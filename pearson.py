import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#%matplotlib inline

attrition = pd.read_csv("D:\Sem 2\A-CRM\Attrition.csv", sep = ",", header = 0)
attrition.head()

print(attrition.columns)

le = preprocessing.LabelEncoder()
attrition = attrition.apply(le.fit_transform)
#attrition_noattr = attrition.drop("Attrition",axis=1)

#attrition_less_corr = attrition.drop(["Department", "JobLevel", "JobRole",
#                                      "MaritalStatus", "MonthlyIncome", 
#                                      "NumCompaniesWorked", "YearsInCurrentRole",
#                                      "TotalWorkingYears", "YearsAtCompany", 
#                                      "YearsSinceLastPromotion", 
#                                      "YearsWithCurrManager"], axis=1)


# Let's see the correlation matrix 
plt.figure(figsize = (20,20))     # Size of the figure
plot = sns.heatmap(attrition.corr(method="spearman"),annot = True) 
fig = plot.get_figure()
fig.savefig("spearman.png", dpi=400)

#plt.figure(figsize = (25,25))     # Size of the figure
#plot = sns.heatmap(attrition_less_corr.corr(method="spearman"),annot = True) 
#fig = plot.get_figure()
#fig.savefig("spearman_less.png", dpi=400)