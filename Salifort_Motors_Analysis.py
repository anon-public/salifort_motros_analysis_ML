
# %% Cell 2
#Import packages
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,f1_score,accuracy_score,precision_score,recall_score, \
classification_report

import matplotlib.pyplot as plt
import seaborn as sns
# %% Cell 3
df = pd.read_csv("HR_capstone_dataset.csv")
# %% Cell 4
df.head(10)
# %% Cell 5
df.info()
# %% Cell 6
df.shape
# %% Cell 7
df.describe()
# %% Cell 8
df = df.rename(columns={"time_spend_company":"tenure","last_evaluation":"last_performance",
                        "average_montly_hours":"average_monthly_hours",
                        "Work_accident":"work_accident"
                        ,"Department":"department"
                       })
df
# %% Cell 9
df.isna().sum()
# %% Cell 10
df.duplicated().sum()

# %% Cell 11
df = df.drop_duplicates(keep="first")
df
# %% Cell 12
sns.boxplot(df["tenure"])
plt.title("Boxplot of tenure")
# %% Cell 13
p25 = df["tenure"].quantile(0.25)
p75 = df["tenure"].quantile(0.75)

iqr = p75-p25
ut = p75 + 1.5*iqr
lt = p25 - 1.5*iqr

print("Lower limit:",lt)
print("Upper limit:",ut)

ot = df[(df["tenure"] < lt) | (df["tenure"] > ut)]
print("Outliers in tenure:",len(ot))
# %% Cell 14
print(df["left"].value_counts())
print(df["left"].value_counts(normalize=True))
# %% Cell 15
fig, ax = plt.subplots(1 ,2 ,figsize = (22,8))

sns.boxplot(data=df,x = df["average_monthly_hours"],y = df["number_project"],orient="h",hue="left",ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title("Avg Monthly hours by no. of projects")

tenure_stay = df[df["left"]==0]["number_project"]
tenure_left = df[df["left"]==1]["number_project"]
sns.histplot(data=df,x ="number_project",hue = "left",multiple="dodge",ax=ax[1])
ax[1].set_title("No. of projects histogram")
plt.show()
# %% Cell 16
df[df["number_project"]==7]["left"].value_counts()
# %% Cell 17
sns.scatterplot(data=df,x ="average_monthly_hours",y="satisfaction_level",hue="left",alpha=0.4,palette={1:"orange",0:"steelblue"})
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', linestyle='--')
plt.legend(["166.67 hrs./mo.","stayed","left",], loc='upper right')
plt.title("Monthly hours by last performance score")



# %% Cell 18
fig, ax = plt.subplots(1 ,2 ,figsize = (22,8))

sns.boxplot(data=df,x = df["satisfaction_level"],y = df["tenure"],orient="h",hue="left",ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title("Satisfaction level by tenure")

tenure_stay = df[df["left"]==0]["tenure"]
tenure_left = df[df["left"]==1]["tenure"]
sns.histplot(data=df,x ="tenure",hue = "left",multiple="dodge",ax=ax[1])
ax[1].set_title("Tenure histogram")
plt.show()
# %% Cell 19
df.groupby("left")["satisfaction_level"].agg(["mean","median"])
# %% Cell 20
fig, ax = plt.subplots(1 ,2 ,figsize = (22,8))

tenure_long = df[df["tenure"]>6]
tenure_short = df[df["tenure"]<7]

sns.histplot(data=tenure_long,x = "tenure",hue = "salary",multiple="dodge",hue_order=["low","medium","high"],ax=ax[0])
ax[0].set_title("Tenure long by salary histogram")

sns.histplot(data=tenure_short,x = "tenure",hue = "salary",multiple="dodge",hue_order=["low","medium","high"],ax=ax[1])
ax[1].set_title("Tenure short by salary histogram")
# %% Cell 21
sns.scatterplot(data=df,x ="average_monthly_hours",y="last_performance",hue="left",alpha=0.4,palette={1:"orange",0:"steelblue"})
plt.axvline(x =166.67,color="red",ls="--",label="166.67 hrs./mo.")
plt.legend(["166.67 hrs./mo.","stayed","left"], loc='upper right')
plt.title("Monthly hours by last performance score")



# %% Cell 22
sns.scatterplot(data=df,x ="average_monthly_hours",y="promotion_last_5years",hue="left",alpha=0.4,palette={1:"orange",0:"steelblue"})
plt.axvline(x =166.67,color="red",ls="--",label="166.67 hrs./mo.")
plt.legend(["166.67 hrs./mo.","stayed","left"], loc='upper right')
plt.title("Monthly hours by promotion last 5 yrs score")



# %% Cell 23
df["department"].value_counts()
# %% Cell 24
sns.histplot(data=df,x="department",hue="left",alpha=0.4,hue_order=[0,1],shrink=.5,multiple="dodge")
plt.xticks(rotation=45)
plt.title("Counts of left/stayed by department")
# %% Cell 25
df0 =df.drop(columns=["department","salary"])
sns.heatmap(df0.corr(),vmin=-1,vmax=1,annot=True,cmap =sns.color_palette("vlag",as_cmap=True))
plt.title("Correlation Heatmap")
# %% Cell 26
df_ml = df.copy()

df_ml["salary"] = (df_ml["salary"].astype("category").cat.set_categories(["low","medium","high"]).cat.codes)

df_ml = pd.get_dummies(df_ml,drop_first=False)
df_ml
# %% Cell 27
sns.heatmap(df_ml[["satisfaction_level","last_performance","number_project","average_monthly_hours","tenure"]].corr(),annot=True,cmap="crest")
plt.title("Correlation Heatmap")
plt.show()
# %% Cell 28
#_____________________Building logestic Regression___________________ 
df_lr = df_ml[(df_ml["tenure"] >= lt) & (df_ml["tenure"] <= ut)]
df_lr
# %% Cell 29
y = df_lr["left"]
X = df_lr.drop("left",axis =1)

# %% Cell 30
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0,test_size=0.25,stratify=y)
# %% Cell 31
lr = LogisticRegression(random_state=0,max_iter=500).fit(X_train,y_train)
# %% Cell 32
y_pred = lr.predict(X_test)
# %% Cell 33
cm = confusion_matrix(y_test,y_pred,labels=lr.classes_)
lr_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr.classes_)
lr_disp.plot(values_format='')
plt.show()
# %% Cell 34
target_name= ["Predicted would not leave ","Predicted would leave"]
print(classification_report(y_test,y_pred,target_names=target_name))     
# %% Cell 35
# _________________Building Decision Tree______________________
dt = DecisionTreeClassifier(random_state=0)

dt_params = {"max_depth":[2,4,6,None],
             "min_samples_leaf":[2,5,1],
             "min_samples_split":[2,4,6]}
scoring = ["accuracy","precision","recall","f1","roc_auc"]
t1 = GridSearchCV(dt,dt_params,scoring=scoring,cv=4,refit="roc_auc")
# %% Cell 36
t1.fit(X_train,y_train)
# %% Cell 37
t1.best_score_
# %% Cell 38
def result_table(model_name:str,model_object,metric:str):
    '''
    Arguments:
    model_name (string): what you want the model to be called in the output table
    model_object: a fit GridSearchCV object
    metric (string): precision, recall, f1, accuracy, or auc
    
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''
    metric_dict = {"roc_auc":"mean_test_roc_auc",
    "precision" : "mean_test_precision",
    "recall" : "mean_test_recall",
    "accuracy" : "mean_test_accuracy",
    "f1" : "mean_test_f1"
    }
    
    cv_result = pd.DataFrame(model_object.cv_results_)
    be_result = cv_result.iloc[cv_result[metric_dict[metric]].idxmax(), :]
    
    auc = be_result.mean_test_roc_auc
    precision= be_result.mean_test_precision
    recall = be_result.mean_test_recall
    accuracy = be_result.mean_test_accuracy
    f1 = be_result.mean_test_f1
    t =pd.DataFrame({"model_name":[model_name],
    "precision":[precision],
    "accuracy":[accuracy],
    "recall":[recall],
    "f1 score":[f1],
    "auc":[auc]})
    return t
    
    

# %% Cell 39
dt_pred = t1.predict(X_test)
# %% Cell 40
table = result_table("Decision Tree",t1,"f1")
table
# %% Cell 41
cm = confusion_matrix(y_test,dt_pred,labels=t1.classes_)
dt_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=t1.classes_)
dt_disp.plot(values_format='')
plt.show()
# %% Cell 42
#___________________Building Random Forest____________________
rf = RandomForestClassifier(random_state=0)
rf_params = {"max_depth":[2,4,6,None],
             "min_samples_leaf":[2,5,1],
             "min_samples_split":[2,4,6]
}
scoring = ["accuracy","precision","recall","f1","roc_auc"]
rf1 = GridSearchCV(rf,rf_params,scoring=scoring,cv=4,refit="roc_auc")
    
# %% Cell 43
rf1.fit(X_train,y_train)
# %% Cell 44
rf1.best_score_
# %% Cell 45
table = result_table("Random Forest",rf1,"f1")
table
# %% Cell 46
rf_pred = rf1.predict(X_test)
# %% Cell 47
cm = confusion_matrix(y_test,rf_pred,labels=rf1.classes_)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf1.classes_)
rf_disp.plot(values_format='')
plt.show()
# %% Cell 48
#______________Building XGBoost model_______________
xgb = XGBClassifier(random_state=0)
xgb_params = {
                "max_depth":          [2,4,6,None],
                "n_estimators":      [100, 200, 300],
                "learning_rate":     [0.01, 0.1, 0.2],
                "subsample":         [0.6, 0.8, 1.0],
}
scoring = ["accuracy","precision","recall","f1","roc_auc"]
xgb1 = GridSearchCV(rf,rf_params,scoring=scoring,cv=4,refit="roc_auc")
# %% Cell 49
xgb1.fit(X_train,y_train)
# %% Cell 50
xgb1.best_score_
# %% Cell 51
xgb_pred = xgb1.predict(X_test)
# %% Cell 52
table = result_table("XGBoost",xgb1,"f1")
table
# %% Cell 53
cm = confusion_matrix(y_test,xgb_pred,labels=xgb1.classes_)
xgb_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=xgb1.classes_)
xgb_disp.plot(values_format='')
plt.show()
# %% Cell 54
import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame with the provided metrics
data = {
    "model_name": ["XGBoost", "Decision Tree", "Random Forest"],
    "precision": [0.989146, 0.975966, 0.989146],
    "accuracy": [0.982806, 0.982807, 0.982806],
    "recall": [0.907880, 0.920634, 0.907880],
    "f1_score": [0.946732, 0.947485, 0.946732],
    "auc": [0.979546, 0.967296, 0.979546]
}

dataframe = pd.DataFrame(data).set_index("model_name")



# Plotting the metrics for comparison
plt.figure()
dataframe.plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# %% Cell 55
#___________Feature Importance__________________
feature_names = X_train.columns
importances   = xgb1.best_estimator_.feature_importances_
feat_imp = pd.Series(importances, index=feature_names)
top_n    = feat_imp.nlargest(10) 

plt.figure(figsize=(8, 6))
top_n.sort_values().plot(kind='barh')
plt.xlabel("Importance")
plt.title("Top 10 XGBoost Feature Importances")
plt.tight_layout()
plt.show()
# %% Cell 56

