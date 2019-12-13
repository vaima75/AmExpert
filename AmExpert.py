# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:58:56 2018

@author: 17020343032
"""

%reset -f
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### Memory Computation
def DataFrame_mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def dtype_mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        for dtype in ['float','int','object','category','datetime']:
            selected_dtype = pandas_obj.select_dtypes(include=[dtype])
            mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
            mean_usage_mb = mean_usage_b / 1024 ** 2
            print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))


#==============================================================================
# def convert_int_to_obj(pandas_obj):
#     if isinstance(pandas_obj, pd.DataFrame):
#         empty_data = pd.DataFrame()
#         train_obj = pandas_obj.copy()
#         if ['int64','float64','object'] in train_obj.dtypes[0]:
#             for col in train_obj.columns:
#                 num_unique_values = len(train_obj[col].unique())
#                 num_total_values = len(train_obj[col])
#                 if num_unique_values / num_total_values < 0.5:
#                     empty_data.loc[:,col] = train_obj[col].astype('object')
#                 else:
#                     empty_data.loc[:,col] = train_obj[col]
#         return empty_data
#==============================================================================

############################## Train Data #################################
amex_train = pd.read_csv('train.csv')#, nrows = 200000)
amex_train.shape
amex_train.columns
amex_train.count()
amex_train.dtypes
amex_train.get_dtype_counts()
amex_train['DateTime'] = pd.to_datetime(amex_train['DateTime'])

# Memory check
amex_train.info()
DataFrame_mem_usage(amex_train)
dtype_mem_usage(amex_train)

##### Descriptive Stats
#asdf = amex_train.describe(include = ['number'])
#fdsa = amex_train.describe(include = ['object','category'])
# Catagorical or object
col_desc = list(amex_train.describe(include = ['object','category']).columns)
for col in col_desc:
    print(amex_train[col].value_counts(), end = '\n\n')
# Float or Int
num_desc = list(amex_train.describe(include = ['number']).columns)
for col in num_desc:
    print(amex_train[col].value_counts(), end = '\n\n')


# Modifing column values
rep_web = {13787:1, 60305:2, 28529:3, 
           6970:4, 45962:5, 53587:6,
           1734:7, 11085:8, 51181:9}

rep_cam = {359520:1, 405490:2, 360936:3,
           118601:4, 98970:5, 414149:6,
           404347:7, 82320:8, 105960:9, 396664:10}

amex_train = amex_train.replace({'webpage_id' : rep_web})
amex_train = amex_train.replace({'campaign_id' : rep_cam})
amex_train.head(4)

# =============================================================================
# # Data type Conversions
# amex_train['webpage_id'] = amex_train['webpage_id'].astype('category')
# amex_train['campaign_id'] = amex_train['campaign_id'].astype('category')
# amex_train['user_group_id'] = amex_train['user_group_id'].astype('category')
# amex_train['product_category_1'] = amex_train['product_category_1'].astype('category')
# amex_train['var_1'] = amex_train['var_1'].astype('category')
# amex_train['is_click'] = amex_train['is_click'].astype('category')
# amex_train['age_level'] = amex_train['age_level'].astype('category')
# amex_train['city_development_index'] = amex_train['city_development_index'].astype('category')
# amex_train['user_depth'] = amex_train['user_depth'].astype('category')
# #amex_train['product'] = amex_train['product'].astype('category')
# #amex_train['gender'] = amex_train['gender'].astype('category')
# 
# =============================================================================

##### Missing Values Treatment (in %)
# Removing high null columns
amex_train.isnull().mean() * 100 #OR amex_train.isnull().sum()/len(amex_train) * 100
amex_train = amex_train.loc[:,(amex_train.isnull().mean()*100 < 20.00)]
impute_col_list = list(amex_train.loc[:,(amex_train.isnull().mean()*100 != 0.00)].columns)

# Imputation (Categorical)
amex_train[impute_col_list].describe()
amex_train[impute_col_list].mode()
amex_train = amex_train.fillna({"user_group_id": 3.0,
                                "gender": "Male",
                                "age_level": 3.0,
                                "user_depth": 3.0})

##### Removing unnessary columns
rem_train_cols = ['DateTime', 'user_id', 'session_id']
amex_train = amex_train.drop(rem_train_cols, axis = 1)

# =============================================================================
########## Load Historical DataSet
# amex_hist = pd.read_csv('historical_user_logs.csv')#, nrows = 100000)
# amex_hist.shape
# amex_hist.columns
# amex_hist.count()
# amex_hist.dtypes
# amex_hist.get_dtype_counts()
# amex_hist = amex_hist.rename(columns = {'DateTime' : 'Enqury_DateTime'})
# amex_hist['Enqury_DateTime'] = pd.to_datetime(amex_hist['Enqury_DateTime'])
# amex_hist['product'] = amex_hist['product'].astype('category')
# amex_hist['action'] = amex_hist['action'].astype('category')
# 
# # Memory check
# amex_hist.info()
# DataFrame_mem_usage(amex_hist)
# dtype_mem_usage(amex_hist)
# 
# # Descriptive Stats
# amex_hist.describe(include = ['number'])
# amex_hist.describe(include = ['object','category'])
# amex_hist['product'].value_counts()
# amex_hist['action'].value_counts()
# 
# # merge Datasets
# result = pd.merge(amex_train, amex_hist, how = 'left', on = ['user_id','product'])
# result = result.drop_duplicates(keep = False)
# result.shape
# result.columns
# result.count()
# result.dtypes
# result.get_dtype_counts()
# result.isnull().mean() * 100 #OR amex_train.isnull().sum()/len(amex_train) * 100
# 
# =============================================================================



####### Model Development ####
# Data Preprocessing
cols = ['product', 'gender']
xxx = pd.get_dummies(amex_train[cols])
result = pd.concat([amex_train, xxx], axis=1)


Xnames = ['campaign_id', 'webpage_id',
          'product_category_1', 'user_group_id',
          'age_level', 'user_depth',
          'var_1']
Xnames = Xnames +  list(xxx.columns)
Ynames = ['is_click']

X = result[Xnames]
Y = result[Ynames]

# Train - Test Split
from sklearn.model_selection import train_test_split
test_size = 0.3
seed = 100
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=test_size,
                                                    random_state = seed)

### Logistic Regression - Accuracy, ROC_AUC, Confusion Matrix, Classification Report
from sklearn.linear_model import LogisticRegression
glm = LogisticRegression()
glm_model = glm.fit(X_train, np.ravel(Y_train, order = 'C'))
# Accuracy
acc_glm = glm_model.score(X_test, Y_test)
round(acc_glm*100.0, 5) # Accuracy (93.14113)
# ROC_AUC
# Method 1 (Cross Validation and K-Fold)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits = 10, random_state = seed)
scoring = 'roc_auc'
glm_roc_auc = cross_val_score(glm_model, X_train, Y_train, cv=kfold, scoring = scoring)
round(glm_roc_auc.mean(),5)    # is the AUC (0.55733)

# Method 2 (Perform AUC on Test Set - with Probabilities)
fpr_glm, tpr_glm, _ = roc_curve(Y_test, glm_model.predict_proba(X_test)[:,1])
roc_auc_glm = auc(fpr_glm, tpr_glm)
#roc_auc_score(Y_test, glm_model.predict(X_test))
# Roc Curve
plt.figure()
plt.plot(fpr_glm, tpr_glm, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_glm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Method 3 (Perform AUC on Test Set - without Probabilities)
#fpr_glm_2, tpr_glm_2, _ = roc_curve(Y_test, glm_model.predict(X_test))
#auc(fpr_glm, tpr_glm)
#roc_auc_score(Y_test, glm_model.predict(X_test))

# =============================================================================
# # Confusion Matrix
# from sklearn.metrics import confusion_matrix
# pred_glm = glm_model.predict(X_test)
# matrix_glm = confusion_matrix(Y_test, pred_glm)
# print(matrix_glm)
# # Classification report
# from sklearn.metrics import classification_report
# report_glm = classification_report(Y_test, pred_glm)
# print(report_glm)
# # Class preportion (Class Balance Ratio)
# Y_train['is_click'].value_counts()/Y_train.shape[0]
# Y_test['is_click'].value_counts()/Y_test.shape[0]
# 
# =============================================================================

### KNN Classification
from sklearn.neighbors import KNeighborsClassifier
KNneigh = KNeighborsClassifier(n_neighbors = 5)
KNneigh_model = KNneigh.fit(X_train, np.ravel(Y_train, order = 'C'))
# AUC
fpr_knn, tpr_knn, _ = roc_curve(Y_test, KNneigh_model.predict_proba(X_test)[:,1])
roc_auc_knn = auc(fpr_knn, tpr_knn)
# ROC Curve
plt.figure()
plt.plot(fpr_knn, tpr_knn, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve KNN Classification')
plt.legend(loc="lower right")
plt.show()


### Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB_model = NB.fit(X_train, np.ravel(Y_train, order = 'C'))
# AUC
fpr_nb, tpr_nb, _ = roc_curve(Y_test, NB_model.predict_proba(X_test)[:,1])
roc_auc_nb = auc(fpr_nb, tpr_nb)
# ROC Curve
plt.figure()
plt.plot(fpr_nb, tpr_nb, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Naive Bayes')
plt.legend(loc="lower right")
plt.show()


### CART Classification
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC_model = DTC.fit(X_train, np.ravel(Y_train, order = 'C'))
# AUC
fpr_dtc, tpr_dtc, _ = roc_curve(Y_test, DTC_model.predict_proba(X_test)[:,1])
roc_auc_dtc = auc(fpr_dtc, tpr_dtc)
# ROC Curve
plt.figure()
plt.plot(fpr_dtc, tpr_dtc, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_dtc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve CART (Decision Trees)')
plt.legend(loc="lower right")
plt.show()


### Spport Vector Classifier
from sklearn.svm import SVC
svm_svc = SVC(probability = True)
svm_svc_model = svm_svc.fit(X_train, np.ravel(Y_train, order = 'C'))
# AUC
fpr_svc, tpr_svc, _ = roc_curve(Y_test, svm_svc_model.predict_proba(X_test)[:,1])
roc_auc_svc = auc(fpr_svc, fpr_svc)
# ROC Curve
plt.figure()
plt.plot(fpr_svc, tpr_svc, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Spport Vector Classifier')
plt.legend(loc="lower right")
plt.show()


### Random Forest Classifier
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = clf_rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_model = rf_random.fit(X_train, np.ravel(Y_train, order = 'C'))
# AUC
fpr_rf, tpr_rf, _ = roc_curve(Y_test, rf_model.predict_proba(X_test)[:,1])
roc_auc_rf = auc(fpr_rf, fpr_rf)
# ROC Curve
plt.figure()
plt.plot(fpr_rf, fpr_rf, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


############################## Test Data ################################
amex_test = pd.read_csv('test.csv')#, nrows = 200000)
amex_test.shape
amex_test.columns
amex_test.count()
amex_test.dtypes
amex_test.get_dtype_counts()
amex_test['DateTime'] = pd.to_datetime(amex_test['DateTime'])

# Memory check
amex_test.info()
DataFrame_mem_usage(amex_test)
dtype_mem_usage(amex_test)

##### Descriptive Stats
#asdf = amex_train.describe(include = ['number'])
#fdsa = amex_train.describe(include = ['object','category'])
# Catagorical or object
col_desc = list(amex_test.describe(include = ['object','category']).columns)
for col in col_desc:
    print(amex_test[col].value_counts(), end = '\n\n')
# Float or Int
num_desc = list(amex_test.describe(include = ['number']).columns)
for col in num_desc:
    print(amex_test[col].value_counts(), end = '\n\n')


# Modifing column values
amex_test = amex_test.replace({'webpage_id' : rep_web})
amex_test = amex_test.replace({'campaign_id' : rep_cam})
amex_test.head(4)

# =============================================================================
# # Data type Conversions
# amex_test['webpage_id'] = amex_test['webpage_id'].astype('category')
# amex_test['campaign_id'] = amex_test['campaign_id'].astype('category')
# amex_test['user_group_id'] = amex_test['user_group_id'].astype('category')
# amex_test['product_category_1'] = amex_test['product_category_1'].astype('category')
# amex_test['var_1'] = amex_test['var_1'].astype('category')
# #amex_test['is_click'] = amex_test['is_click'].astype('category')
# amex_test['age_level'] = amex_test['age_level'].astype('category')
# amex_test['city_development_index'] = amex_test['city_development_index'].astype('category')
# amex_test['user_depth'] = amex_test['user_depth'].astype('category')
# #amex_test['product'] = amex_test['product'].astype('category')
# #amex_test['gender'] = amex_test['gender'].astype('category')
# 
# =============================================================================

##### Missing Values Treatment (in %)
# Removing high null columns
amex_test.isnull().mean() * 100 #OR amex_train.isnull().sum()/len(amex_train) * 100
amex_test = amex_test.loc[:,(amex_test.isnull().mean()*100 < 20.00)]
impute_col_list = list(amex_test.loc[:,(amex_test.isnull().mean()*100 != 0.00)].columns)

# Imputation (Categorical)
amex_test[impute_col_list].describe()
amex_test[impute_col_list].mode()
amex_test = amex_test.fillna({"user_group_id": 3.0,
                                "gender": "Male",
                                "age_level": 3.0,
                                "user_depth": 3.0})

##### Remove unnecessary columns
rem_test_cols = ['DateTime', 'user_id']
amex_test = amex_test.drop(rem_test_cols, axis = 1)


# Data Preprocessing
xxx_test = pd.get_dummies(amex_test[cols])
result_test = pd.concat([amex_test, xxx_test], axis=1)

Xnames_test = ['campaign_id', 'webpage_id',
          'product_category_1', 'user_group_id',
          'age_level', 'user_depth',
          'var_1']

Xnames_test = Xnames_test +  list(xxx_test.columns)

X_amex_test = result_test[Xnames_test]


# =============================================================================
# ### Submission File (DTC)
# Y_amex_test = DTC_model.predict(X_amex_test)
# df_DTC = pd.DataFrame(Y_amex_test)
# df_DTC.columns=['is_click']
# 
# df_DTC = pd.concat([amex_test['session_id'], df_DTC], axis=1)
# 
# df_DTC.to_csv("DTC_file_path.csv")
# 
# =============================================================================
### Submission File (Random Forest Classifier)
Y_amex_test = rf_model.predict(X_amex_test)
df_DTC = pd.DataFrame(Y_amex_test)
df_DTC.columns=['is_click']
df_DTC = pd.concat([amex_test['session_id'], df_DTC], axis=1)
df_DTC.to_csv("rf_file_path.csv")
