import pandas as pd
from pandas import Series
import pandas.core.algorithms as algos
import numpy as np
import scipy.stats.stats as stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import re # for regular expression
import traceback # for trracing an error back to its source
import string

df_loan = pd.read_csv("lending_club_loan_data.csv")

# analyse the target variables - loan_status
df_loan["loan_status"].value_counts()

loan_filter = df_loan["loan_status"].isin([
    "Fully Paid",
    "Charged Off",
    "Default"
])
df_loan = df_loan[loan_filter]
df_loan["loan_status"].value_counts()

def CreateTarget(status):
    if status == "Fully Paid":
        return 0
    else:
        return 1

df_loan["Late_Loan"] = df_loan["loan_status"].map(CreateTarget)
df_loan["Late_Loan"].value_counts()
df_loan["Late_Loan"].mean()

# drop features with more than 10% missing values
features_missing_series = df_loan.isnull().sum() > len(df_loan)/10
features_missing_series = features_missing_series[features_missing_series == True]
features_missing_list = features_missing_series.index.tolist()
df_loan = df_loan.drop(features_missing_list, axis=1)

df_loan_1 = df_loan.drop(["id", "member_id", "loan_status", "url", "zip_code", "policy_code", "application_type", "last_pymnt_d", "last_credit_pull_d", "verification_status", "pymt_plan", "funded_amnt", "funded_amnt_inv", "sub_grade", "out_prncp_inv", "total_pymt_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "initial_list_status", "earliest_cr_line"], axis=1)
df_loan_1["emp_length"].fillna("Unknown", inplace=True)
df_loan_1["emp_title"].fillna("Unknown", inplace=True)
df_loan_1["title"].fillna("Unknown", inplace=True)
df_loan_1["revol_util"].fillna(df_loan_1["revol_util"].mean(), inplace=True)
df_loan_1["collections_12_mths_ex_med"].fillna(df_loan_1["collections_12_mths_ex_med"].mean(), inplace=True)
df_loan_1.isnull().sum()

# Binning
max_bin = 20
force_bin = 3

## binning function
def mono_bin(Y, X, n=max_bin):
    df1 = pd.DataFrame({
        "X": X,
        "Y": Y
    })
    justmiss = df1[["X", "Y"]][df1.X.isnull()]
    notmiss = df1[["X", "Y"]][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({
                "X": notmiss.X,
                "Y": notmiss.Y,
                # quantile cut: each bin will be roughly the same
                "Bucket": pd.qcut(notmiss.X, n)
            })
            d2 = d1.groupby("Bucket", as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n-1
        except Exception as e:
            n = n-1

    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({
            "X": notmiss.X,
            "Y": notmiss.Y,
            # cut based on the predefined range
            "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)
        })
        d2 = d1.groupby("Bucket", as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({
            "MIN_VALUE": np.nan
        }, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[["VAR_NAME","MIN_VALUE", "MAX_VALUE", "COUNT", "EVENT", "EVENT_RATE", "NONEVENT", "NON_EVENT_RATE", "DIST_EVENT", "DIST_NON_EVENT", "WOE", "IV"]]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return(d3)

def char_bin(Y, X):

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[["X", "Y"]][df1.X.isnull()]
    notmiss = df1[["X", "Y"]][df1.X.notnull()]    
    df2 = notmiss.groupby("X",as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({
            "MIN_VALUE":np.nan
            },index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[["VAR_NAME","MIN_VALUE", "MAX_VALUE", "COUNT", "EVENT", "EVENT_RATE", "NONEVENT", "NON_EVENT_RATE", "DIST_EVENT", "DIST_NON_EVENT", "WOE", "IV"]]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r"\((.*?)\).*$").search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count += 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count += 1
            
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)
    
    iv = pd.DataFrame({
        "IV": iv_df.groupby("VAR_NAME").IV.max()
    })
    iv = iv.reset_index()

    return(iv_df, iv)

final_iv, IV = data_vars(df_loan_1, df_loan_1.Late_Loan)
final_iv

IV.sort_values("IV")

transform_vars_list = df_loan_1.columns.difference(["Late_Loan"])
transform_prefix = "new_" # leave this value blank if you need replace the original column values

transform_vars_list
df = df_loan_1
df.head()

for var in transform_vars_list:
    small_df = final_iv[final_iv["VAR_NAME"] == var]
    transform_dict = dict(zip(small_df.MAX_VALUE, small_df.WOE))
    replace_cmd = ""
    replace_cmd1 = ""
    for i in sorted(transform_dict.items()):
        replace_cmd = replace_cmd + str(i[i]) + str(" if x <= ") + str(i[0]) + " else "
        replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
    replace_cmd = replace_cmd + "0"
    replace_cmd1 = replace_cmd1 + "0"
    if replace_cmd != "0":
        try:
            df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd))
        except:
            df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd1))

df["new_grade"].value_counts()
df["grade"].value_counts()

df_loan.groupby("grade").mean()

y = df["Late_Loan"]
y.mean()
features = [
    "new_addr_state",
    "new_annual_inc",
    "new_delinq_2yrs",
    "new_dti",
    "new_emp_length",
    "new_grade",
    "new_home_ownership",
    "new_inq_last_6mths",
    "new_int_rate",
    "new_issue_d",
    "new_loan_amnt",
    "new_open_acc",
    "new_purpose",
    "new_revol_util",
    "new_term"
]
X = df[features]

import statsmodels.api as sm
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())


from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

dtree = DecisionTreeClassifier(criterion="gini",
                               random_state=0,
                               max_depth=5,
                               min_samples_leaf=5)
dtree.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_pred_tree = dtree.predict(X_test)
print("Accuracy of Logistic Regression Classifier on Test Set: {:.2f}", format(logreg.score(X_test, y_test)))
        
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
confusion_matrix = confusion_matrix(y, y_pred)
pd.crosstab(y, y_pred)
print(classification_report(y, y_pred))
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
tree_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict(X_test)[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, dtree.predict(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label="Logistic Regression (area=%0.2f)"%logit_roc_auc)
plt.plot(fpr, tpr, label="Logistic Regression (area=%0.2f)"%tree_roc_auc)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")
plt.legend(loc="lower right")
plt.savefig("Log_ROC")
plt.show()
