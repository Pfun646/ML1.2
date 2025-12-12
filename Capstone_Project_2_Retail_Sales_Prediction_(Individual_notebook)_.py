#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-12T16:32:58.819Z
"""

# <a href="https://colab.research.google.com/github/Gaurav-Yaduvanshi/Capstone-Project-2-Retail-Sales-Prediction/blob/main/Capstone_Project_2_Retail_Sales_Prediction_(Individual_notebook)_.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# # Project Title : Retail Sales Prediction : Predicting sales of a major store chain Rossmann
# # Problem Description
# ### Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.
# ### You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.
# # Data Description
# ## Rossmann Stores Data.csv - historical data including Sales
# ## store.csv - supplemental information about the stores
# # Data fields
# ### Id - an Id that represents a (Store, Date) duple within the test set
# ### Store - a unique Id for each store
# ### Sales - the turnover for any given day (this is what you are predicting)
# ### Customers - the number of customers on a given day
# ### Open - an indicator for whether the store was open: 0 = closed, 1 = open
# ### StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a=public holiday, b=Easter holiday, c=Christmas, 0=None
# ### SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
# ### StoreType - differentiates between 4 different store models: a, b, c, d
# ### Assortment - describes an assortment level: a = basic, b = extra, c = extended
# ### CompetitionDistance - distance in meters to the nearest competitor store
# ### CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
# ### Promo - indicates whether a store is running a promo on that day
# ### Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
# ### Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
# ### PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#importing the data, and dispaling first 5 line of data
RS_df = pd.read_csv(r"C:\Users\HP\Desktop\BDS 4.1\PROJECT\Sales Prediction\Working station\Rossmann Stores Data.csv")
RS_df.head()

#file2 = '/Users/HP/Desktop/BDS 4.1/PROJECT/Sales Prediction/Working station/store.csv/'
#S_df = pd.read_csv(file2 + 'store.csv')


S_df = pd.read_csv(r"C:\Users\HP\Desktop\BDS 4.1\PROJECT\Sales Prediction\Working station\store.csv")
#"C:\Users\HP\Desktop\BDS 4.1\PROJECT\Sales Prediction\Working station\store.csv"
#"C:\Users\HP\Desktop\BDS 4.1\PROJECT\Sales Prediction\Working station\store.csv"
#Users/HP/Desktop/BDS 4.1/PROJECT/Sales Prediction/Working station/store.csv/

# # DATA PREPROCESSING


RS_df.shape,S_df.shape

RS_df.info()


RS_df.head()

RS_df.tail()

S_df.info()

S_df.head()

S_df.describe()

# ## DATA CLEANING


# Checking Null values
S_df.isna().sum()


# There are many Nan values in columns -'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear, Promointerval', 'Promo2sinceWeek' and 'Promo2sinceYear'. Also CompetitionDistance has only 3 null values.


# duplicates in Store.csv
len(S_df[S_df.duplicated()])

# duplicates in Rossmann Stores Data.csv
len(RS_df[RS_df.duplicated()])

# Null values of CompetitionDistance 
S_df[pd.isna(S_df.CompetitionDistance)]

# Replacing Null values in CompetitionDistance with median.
S_df['CompetitionDistance'].fillna(S_df['CompetitionDistance'].median(), inplace = True)

# Replacing Null values with 0 in CompetitionOpenSinceMonth
S_df['CompetitionOpenSinceMonth'] = S_df['CompetitionOpenSinceMonth'].fillna(0)

# Replacing Nan values with 0 in CompetitionOpenSinceYear
S_df['CompetitionOpenSinceYear'] = S_df['CompetitionOpenSinceYear'].fillna(0)

# Replacing Nan values with 0 in Promo2SinceWeek
S_df['Promo2SinceWeek'] = S_df['Promo2SinceWeek'].fillna(0)

## Replacing Nan values with 0 in Promo2SinceYear
S_df['Promo2SinceYear'] = S_df['Promo2SinceYear'].fillna(0)

## Replacing Nan values with 0 in PromoInterval
S_df['PromoInterval'] = S_df['PromoInterval'].fillna(0)

## Now checking Nan values
S_df.isna().sum()


# Changing StateHoliday dtype from object to int.
RS_df.loc[RS_df['StateHoliday'] == '0', 'StateHoliday'] = 0
RS_df.loc[RS_df['StateHoliday'] == 'a', 'StateHoliday'] = 1
RS_df.loc[RS_df['StateHoliday'] == 'b', 'StateHoliday'] = 2
RS_df.loc[RS_df['StateHoliday'] == 'c', 'StateHoliday'] = 3
RS_df['StateHoliday'] = RS_df['StateHoliday'].astype(int, copy=False)

RS_df['StateHoliday'].dtype

# Changing Assortment dtype to int.
S_df.loc[S_df['PromoInterval'] == 'Jan,Apr,Jul,Oct', 'PromoInterval'] = 1
S_df.loc[S_df['PromoInterval'] == 'Feb,May,Aug,Nov', 'PromoInterval'] = 2
S_df.loc[S_df['PromoInterval'] == 'Mar,Jun,Sept,Dec', 'PromoInterval'] = 3
S_df['PromoInterval'] = S_df['PromoInterval'].astype(int, copy=False)

S_df['PromoInterval'].dtype

# Store column is common in both the datasets
# Merging both datasets
df = pd.merge(RS_df,S_df , on = 'Store', how ='left')

df.head()

df.shape

# Changing StoreType and Assortment
df['StoreType'] = df['StoreType'].map({'a':1,'b':2,'c':3,'d':4})
df['Assortment'] = df['Assortment'].map({'a':0, 'c':1,'b':2})


df.info()

# Extract year, month, day and from "Date"
df['Date']=pd.to_datetime(df['Date'])
df['Year'] = df['Date'].apply(lambda x: x.year)
df['Month'] = df['Date'].apply(lambda x: x.month)
df['Day'] = df['Date'].apply(lambda x: x.day)
df.drop('Date',axis=1,inplace=True)

df.head()

df.describe().apply(lambda s: s.apply('{0:.2f}'.format))

df.columns

# # EDA - Exploratory Data Analysis


sns.set(rc = {'figure.figsize':(15,8)})
categorical_variables = ['DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday','StoreType','Assortment',
                         'CompetitionOpenSinceMonth','Promo2','Promo2SinceYear','PromoInterval']
for value in categorical_variables:
  ax = sns.barplot(x=df[value], y=df['Sales'])
  plt.show()   

df['Customers'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(data=df,x='Day',y='Sales',hue='Promo')

sns.lmplot(
    data=df, x='Month', y="Sales",
    hue="Promo"
)

plt.figure(figsize=(12,6))
sns.lineplot(data=df,x='Customers',y='Sales',hue='DayOfWeek')

plt.figure(figsize=(12,6))
sns.barplot(data=df,x='Month',y='Sales')

plt.figure(figsize=(12,6))
sns.barplot(data=df,x=df['Year'].value_counts(),y='Sales')

plt.figure(figsize=(15,8))
sns.barplot(data=df,x='Day',y='Sales')

plt.figure(figsize=(15,8))
sns.lineplot(data = df,x ='Month', y='Sales', hue='Year',palette=['r', 'g', 'b'])
plt.show()

plt.figure(figsize=(15,10))
sns.distplot(df['Sales'],color="y")

dfcount = (df['Sales'] < 20000).value_counts()
dfcount

logsales = np.log10(df['Sales'])

sns.displot(logsales,color="y")

plt.figure(figsize=(15,10))
sns.boxplot(df['Sales'],color="y")

sns.factorplot(data = df, x = "Month", y = "Sales", col = "Year", hue = "StoreType")

g = sns.PairGrid(df[['Sales','Customers','CompetitionDistance', 'CompetitionOpenSinceMonth', 'Promo2SinceWeek']])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

plt.figure(figsize=(15,8))
sns.scatterplot(data=df,x='Customers',y='Sales',hue='StoreType')
plt.show()

percent = df["StoreType"].value_counts()
percent.plot.pie(fontsize= 18,autopct="%.1f%%",figsize=(10,8),textprops={'fontsize': 20})
plt.title('Store Type')
plt.show()

percent = df["Assortment"].value_counts()
percent.plot.pie(fontsize= 18,autopct="%.1f%%",figsize=(10,8),textprops={'fontsize': 20})
plt.show()

plt.figure(figsize=(15,8))
sns.scatterplot(data=df,x='Customers',y='Sales',hue='Assortment')
plt.show()

df.columns

plt.figure(figsize=(15,8))
sns.scatterplot(data=df,x='Customers',y='Sales',hue='Store')
plt.show()

plt.figure(figsize=(15,8))
sns.scatterplot(data=df,x='Customers',y='Sales',hue='CompetitionOpenSinceMonth')
plt.show()

plt.figure(figsize=(15,8))
sns.scatterplot(data=df,x='Customers',y='Sales',hue='Month')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="Assortment", y="Sales", data=df)
plt.title('Boxplot For Sales Values on the basis of Assortment Level')

plt.figure(figsize=(10, 6))
sns.boxplot(x="StoreType", y="Sales", data=df)
plt.title('Boxplot For Sales Values')



corr_df = df[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']].corr()
plt.figure(figsize=(18,9))
sns.heatmap(corr_df,cmap='coolwarm',annot=True)
plt.show()

# # Multicollinearity


#Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(df[[i for i in df.describe().columns if i not in ['Sales','Open']]])

# # Feature Selection 
# 
# 


import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import log_loss

# Create the data of independent variables
dependent_variable = 'Sales'
# Create a list of independent variables
independent_variables = list(set(df.columns.tolist()) - {dependent_variable})
X = df[independent_variables].values

# Create the dependent variable data
y = df[dependent_variable].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Transforming data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ## LINEAR REGRESSION


# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Checking the score on train set.
regressor.score(X_train, y_train)

#checking the coefficeint
regressor.coef_

#check the intercept
regressor.intercept_

# Predicting the Test set results
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

from numpy import math
math.sqrt(mean_squared_error(y_test, y_pred_test))

# Checking the Performance on train set
r2_train= r2_score(y_train, y_pred_train)
print(r2_train)

# Checking the Performance on test set
r2_test= r2_score(y_test, y_pred_test)
print(r2_test)

zipped = dict(zip(y_pred_test,np.array(y_test)))
predict = pd.DataFrame(zipped,index=[0])

plt.figure(figsize=(30,8))
plt.plot(y_pred_test)
plt.plot(np.array(y_test))
plt.legend(["Predicted","Actual"])
plt.show()

### Heteroscadacity
plt.scatter((y_pred_test),(y_test)-(y_pred_test))
plt.show()



# ## LASSO REGRESSION


# Createing an instance of Lasso Regression implementation
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.0001, max_iter=3000)
# Fit the Lasso model
lasso.fit(X_train, y_train)
# Create the model score
print(lasso.score(X_test, y_test), lasso.score(X_train, y_train))

# Predicting the X_train and X-test value
y_pred_train_lasso=lasso.predict(X_train)
y_pred_test_lasso=lasso.predict(X_test)

# Checking the Performance on train set
r2_train_lasso= r2_score(y_train, y_pred_train_lasso)
print(r2_train_lasso)

# Checking the Performance on test set
r2_test_lasso= r2_score(y_test, y_pred_test_lasso)
print(r2_test_lasso)

# Plot 
plt.figure(figsize=(12,8))
plt.plot(np.array(y_pred_test_lasso))
plt.plot(np.array(y_test))
plt.legend(["Predicted","Actual"])
plt.show()



# ## RIDGE REGRESSION


#import the packages
from sklearn.linear_model import Ridge
ridge= Ridge(alpha=0.01)
# Fitting the model
ridge.fit(X_train,y_train)
# check the score
ridge.score(X_train, y_train)

#Predict the X_train and X-test values
y_pred_train_ridge=ridge.predict(X_train)
y_pred_test_ridge=ridge.predict(X_test)

# Checking the Performance on train set
r2_train_ridge = r2_score(y_train, y_pred_train_ridge)
print(r2_train_ridge)

# Checking the Performance on test set
r2_test_ridge = r2_score(y_test, y_pred_test_ridge)
print(r2_test_ridge)

# ## Decision Tree


from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# defining dependent variable
dep_var = 'Sales'

# defining independent variable
indep_var = df.columns.drop(['Year','Month','Day','Open','Sales'])

# Create the data of independent variables
X = df[indep_var]

# Create the dependent variable data
y = df[dep_var]

# the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


from sklearn.preprocessing import StandardScaler
# Transforming data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# DecisionTreeRefressor
Decision_tree = DecisionTreeRegressor()
Decision_tree_reg = Decision_tree.fit(X_train, y_train)

#predict the X_train and X-test values
y_pred_train_d = Decision_tree.predict(X_train)
y_pred_test_d = Decision_tree.predict(X_test)

# Checking the Performance on train set
r2_train_decision = r2_score(y_train, y_pred_train_d)
print(r2_train_decision)

# Checking the Performance on test set
r2_test_decision= r2_score(y_test, y_pred_test_d)
print(r2_test_decision)

#Plot
plt.figure(figsize=(12,6))
plt.plot(np.array(y_pred_test_d))
plt.plot(np.array(y_test))
plt.legend(["Predicted","Actual"])
plt.show()

# ## Decision Tree With Hyper Parameter Tuning


Decision_tree1= DecisionTreeRegressor(min_samples_leaf=8,min_samples_split=5)
Decision_tree_reg1 = Decision_tree1.fit(X_train, y_train)

train_score = Decision_tree_reg1.score(X_train, y_train)
Test_Score = Decision_tree_reg1.score(X_test, y_test)
print(train_score)
print(Test_Score)

# ## Random Forest


# 


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix

#I trained Model with hyper parameters..to not run everytime i record the result
# Here are our best parameters for Random Forest
# n_estimators=80,min_samples_split=2,min_samples_leaf=1,max_depth=None 

#Random Forest with Parameters
rdf = RandomForestRegressor(n_estimators=80,min_samples_split=2, min_samples_leaf=1,max_depth=None,n_jobs=-1)
rdfreg = rdf.fit(X_train, y_train)

# Making predictions on train and test data
y_pred_train_r = rdf.predict(X_train)
y_pred_test_r = rdf.predict(X_test)

# Checking the Performance on train set
r2_train_random = r2_score(y_train, y_pred_train_r)
print(r2_train_random)

# Checking the Performance on test set
r2_test_random= r2_score(y_test, y_pred_test_r)
print(r2_test_random)

# Heteroscadacity
plt.scatter((y_pred_test_r),(y_test)-(y_pred_test_r))
plt.show()

# # Feature Importance


df.columns

X_train = pd.DataFrame(X_train, columns = X.columns)

X_train.columns

importances = rdf.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),
                   'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)

importance_df['Feature Importance'] = round(importance_df['Feature Importance'],2)

importance_df.sort_values(by=['Feature Importance'],ascending=False)

# # Conclusion


Score_df = pd.DataFrame({'Regressions':['LinearRegression','LassoRegression','RidgeRegresion', 'DecisionTreeRegressor', 'RandomForestRegressor'],'Train_Score':[r2_train, r2_train_lasso, r2_train_ridge,r2_train_decision, r2_train_random],'Test_Score':[r2_test,r2_test_lasso,r2_test_ridge, r2_test_decision, r2_test_random]})

# Comparsion of regression models
Score_df

# # CONCLUSION
# In our analysis, we initially did EDA on all the features of our datset. We first analysed our dependent variable, 'Sales' and also transformed it. Next we analysed categorical variable and replaced null values, we also analysed numerical variable, found out the correlation, distribution and their relationship with the dependent variable using corr() Function. We also removed some numerical features who had mostly 0 values and hot encoded the categorical variables.
# 
# Next we implemented six machine learning algorithms Linear Regression, lasso,ridge, decission tree, Random Forest. We did hyperparameter tuning into improve our model performance.
# 
# 1. The sales in the month of December is the highest sales among others.
# 2. The Sales is highest on Monday and start declining from Tuesday to Saturday and on Sunday Sales almost near to Zero.
# 3. Those Stores who takes participate in Promotion got their Sales increased.
# 4. Type of Store plays an important role in opening pattern of stores. All Type ‘b’ stores never closed except for refurbishment or other reason.
# 5. We can observe that most of the stores remain closed during State holidays. But it is interesting to note that the number of stores opened during School Holidays were more than that were opened during State Holidays.
# 6. The R Squared score of all Liner Regression Algorithm with or without Regularization are quit good which is 0.86.
# 7. the R Squared score of the Decision Tree Regressor model we got 0.97 on test set which is also good.
# 8. The Random Forest regressor model performed 0.98 which is very well amoung the others.
# 10. We can say that random forest regressor model is our optimal model and can be deploy.