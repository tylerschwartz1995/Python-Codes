#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../Pierian_Data_Logo.PNG"></a>
# <strong><center>Copyright by Pierian Data Inc.</center></strong> 
# <strong><center>Created by Jose Marcial Portilla.</center></strong>

# # Keras API Project Exercise
# 
# ## The Data
# 
# We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# ## NOTE: Do not download the full zip from the link! We provide a special version of this file that has some extra feature engineering for you to do. You won't be able to follow along with the original file!
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ### Our Goal
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model thatcan predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. Keep in mind classification metrics when evaluating the performance of your model!
# 
# The "loan_status" column contains our label.
# 
# ### Data Overview

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----

# ## Starter Code
# 
# #### Note: We also provide feature information on the data as a .csv file for easy lookup throughout the notebook:

# In[203]:


import pandas as pd


# In[204]:


data_info = pd.read_csv('DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[205]:


print(data_info.loc['revol_util']['Description'])


# In[206]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[207]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[208]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[209]:


df = pd.read_csv('DATA/lending_club_loan_two.csv')


# In[210]:


df.info()


# # Project Tasks
# 
# **Complete the tasks below! Keep in mind is usually more than one way to complete the task! Enjoy**
# 
# -----
# ------
# 
# # Section 1: Exploratory Data Analysis
# 
# **OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data**
# 
# 
# ----

# **TASK: Since we will be attempting to predict loan_status, create a countplot as shown below.**

# In[132]:


sns.countplot("loan_status", data = df)


# **TASK: Create a histogram of the loan_amnt column.**

# In[133]:


sns.histplot(df["loan_amnt"], bins = 20)


# **TASK: Let's explore correlation between the continuous feature variables. Calculate the correlation between all continuous numeric variables using .corr() method.**

# In[211]:


df_corr = df.corr()
df_corr


# **TASK: Visualize this using a heatmap. Depending on your version of matplotlib, you may need to manually adjust the heatmap.**
# 
# * [Heatmap info](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)
# * [Help with resizing](https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot)

# In[135]:


plt.figure(figsize=(10,6))
sns.heatmap(df_corr, cmap = 'coolwarm', annot=True)


# **TASK: You should have noticed almost perfect correlation with the "installment" feature. Explore this feature further. Print out their descriptions and perform a scatterplot between them. Does this relationship make sense to you? Do you think there is duplicate information here?**

# In[136]:


print(data_info.loc['installment']['Description'])


# In[137]:


print(data_info.loc['loan_amnt']['Description'])


# In[138]:


sns.scatterplot(x = 'installment', y= 'loan_amnt', data = df, hue = "loan_status")
plt.legend(loc='lower right')


# **TASK: Create a boxplot showing the relationship between the loan_status and the Loan Amount.**

# In[139]:


sns.boxplot(x = 'loan_status', y = 'loan_amnt', data = df)


# **TASK: Calculate the summary statistics for the loan amount, grouped by the loan_status.**

# In[140]:


df.groupby('loan_status').describe()['loan_amnt']


# **TASK: Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?**

# In[141]:


df['grade'].value_counts()


# In[142]:


df['sub_grade'].value_counts()


# **TASK: Create a countplot per grade. Set the hue to the loan_status label.**

# In[143]:


sns.countplot(x='grade', data = df, hue = 'loan_status')


# **TASK: Display a count plot per subgrade. You may need to resize for this plot and [reorder](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) the x axis. Feel free to edit the color palette. Explore both all loans made per subgrade as well being separated based on the loan_status. After creating this plot, go ahead and create a similar plot, but set hue="loan_status"**

# In[144]:


plt.figure(figsize=(12,4))
sns.countplot(x=df['sub_grade'].sort_values(ascending = True), palette='coolwarm')


# In[145]:


plt.figure(figsize=(12,4))
sns.countplot(x=df['sub_grade'].sort_values(ascending = True), palette='coolwarm', hue = df['loan_status'])


# **TASK: It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.**

# In[146]:


df_fg = df[df['grade'].isin(['F','G'])]
plt.figure(figsize=(12,4))
sns.countplot(x=df_fg['sub_grade'].sort_values(ascending = True), hue = df_fg['loan_status'])


# **TASK: Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[212]:


loan_repaid = pd.get_dummies(df['loan_status'],drop_first=True)
df['loan_repaid'] = loan_repaid


# In[213]:


df.filter(items=['loan_status', 'loan_repaid'])


# **CHALLENGE TASK: (Note this is hard, but can be done in one line!) Create a bar plot showing the correlation of the numeric features to the new loan_repaid column. [Helpful Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html)**

# In[149]:


df.corr()['loan_repaid'][:-1].sort_values().plot(kind='bar')


# ---
# ---
# # Section 2: Data PreProcessing
# 
# **Section Goals: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.**
# 
# 

# In[214]:


#Making whole dataset in lower case for text columns
df = df.applymap(lambda s: s.lower() if type(s) == str else s)


# In[215]:


#Imputing unknown employee title to 'unk' and making dummy
df['emp_title'].fillna(value = 'Unknown', inplace = True)

#Dropping colummn
df.drop('emp_title', inplace = True, axis = 1)


# In[216]:


#Imputing unknown employee length to 'unknown' and making column dummy variables
df['emp_length'].fillna(value = 'Unknown', inplace = True)

#Converting categorical variables to dummy variables
emp_length = pd.get_dummies(df['emp_length'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,emp_length], axis=1)


# In[217]:


#Seeing title variable value counts
df['title'].value_counts().head(50)

#Imputing unknown to na's
df['title'].fillna(value = 'Unknown', inplace = True)

#Replacing counts under 500 to other
df_map = df['title'].map(df['title'].value_counts()) < 500
df['title'] =  df['title'].mask(df_map, 'other')

#Converting categorical variables to dummy variables
title = pd.get_dummies(df['title'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,title], axis=1)


# In[155]:


#revol_util
df['revol_util'].head(10)

#Make dummy for if na value
for i in range(0, len(df)):
    if pd.isnull(df['revol_util'][i]) == True:
        df["revol_util_na"] = 1
    else:
        df["revol_util_na"] = 0


# In[218]:


#Imputing value into na's for column
df['revol_util'].fillna(value = df['revol_util'].mean(), inplace = True)


# In[219]:


#Seeing mort_acc variable value counts
df['mort_acc'].value_counts().head(50)

#Imputing unknown to na's
df['mort_acc'].fillna(value = 'Unknown_Mort', inplace = True)

#Converting categorical variables to dummy variables
mort = pd.get_dummies(df['mort_acc'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,mort], axis=1)


# In[220]:


#pub_rec_bankruptcies

#Imputing unknown to na's
df['pub_rec_bankruptcies'].fillna(value = 'Unknown_bankrupt', inplace = True)

#Converting categorical variables to dummy variables
bankrupt = pd.get_dummies(df['pub_rec_bankruptcies'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,bankrupt], axis=1)


# In[221]:


#Dropping all columns moved to dummy
df.drop(["emp_length", "title", "mort_acc", "pub_rec_bankruptcies"], inplace = True, axis = 1)


# In[222]:


df.isnull().values.any() #No more na values present


# In[223]:


df.head()


# ## Categorical Variables and Dummy Variables
# 
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**
# 
# **TASK: List all the columns that are currently non-numeric. [Helpful Link](https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type)**
# 
# [Another very useful method call](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)

# In[224]:


df.select_dtypes(['object']).columns


# ---
# **Let's now go through all the string features to see what we should do with them.**
# 
# ---
# 
# 
# ### term feature
# 
# **TASK: Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().**

# In[225]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[226]:


#Extracting zip code from address


# In[227]:


df['address'].head()


# In[228]:


def zip_code(text):
    text = list(text.split())
    code = text[-1]
    city = list(code)
    city = code[0:2]
    return city

df['zipcode_1']= df['address'].apply(zip_code)


# In[229]:


#Dropping address
df.drop('address', inplace = True, axis =1)


# In[230]:


#Converting categorical variables to dummy variables
zip_code = pd.get_dummies(df['zipcode_1'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,zip_code], axis=1)


# In[231]:


df.drop('zipcode_1', inplace = True, axis =1)


# ### grade feature
# 
# **TASK: We already know grade is part of sub_grade, so just drop the grade feature.**

# In[232]:


df.drop('grade', inplace = True, axis =1)


# **TASK: Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.**

# In[233]:


#Converting categorical variables to dummy variables
sub_grade = pd.get_dummies(df['sub_grade'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,sub_grade], axis=1)


# In[234]:


df.select_dtypes(['object']).columns


# ### verification_status, application_type,initial_list_status,purpose 
# **TASK: Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[235]:


#Converting categorical variables to dummy variables
verification_status = pd.get_dummies(df['verification_status'],drop_first=True)
application_type = pd.get_dummies(df['application_type'],drop_first=True)
initial_list_status = pd.get_dummies(df['initial_list_status'],drop_first=True)
purpose = pd.get_dummies(df['purpose'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,verification_status,application_type,initial_list_status,purpose], axis=1)

#Dropping columns
df.drop(['sub_grade','verification_status','application_type', 'initial_list_status', 'purpose'], inplace = True, axis =1)


# In[236]:


df.select_dtypes(['object']).columns


# ### home_ownership
# **TASK:Review the value_counts for the home_ownership column.**

# In[237]:


df['home_ownership'].value_counts()


# **TASK: Convert these to dummy variables, but [replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[238]:


df['home_ownership'] = df['home_ownership'].str.replace('any', 'other')
df['home_ownership'] = df['home_ownership'].str.replace('none', 'other')


# In[239]:


#Converting categorical variables to dummy variables
home_ownership = pd.get_dummies(df['home_ownership'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,home_ownership], axis=1)


# In[240]:


#Dropping column
df.drop('home_ownership', inplace = True, axis =1)


# ### issue_d 
# 
# **TASK: This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**

# In[241]:


#Dropping column
df.drop('issue_d', inplace = True, axis =1)


# In[242]:


df.select_dtypes(['object']).columns


# ### earliest_cr_line
# **TASK: This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.**

# In[243]:


df['earliest_cr_line'].head(10)


# In[244]:


df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
df['year'] = df['earliest_cr_line'].apply(lambda date:date.year)

#Converting categorical variables to dummy variables
year = pd.get_dummies(df['year'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,year], axis=1)

df.drop(['year','earliest_cr_line'], inplace = True, axis =1)


# In[245]:


df.select_dtypes(['object']).columns


# ## Train Test Split

# **TASK: Import train_test_split from sklearn.**

# In[246]:


from sklearn.model_selection import train_test_split


# **TASK: drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.**

# In[247]:


df.drop('loan_status', inplace = True, axis = 1)


# **TASK: Set X and y variables to the .values of the features and label.**

# In[249]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# ----
# ----
# 
# # OPTIONAL
# 
# ## Grabbing a Sample for Training Time
# 
# ### OPTIONAL: Use .sample() to grab a sample of the 490k+ entries to save time on training. Highly recommended for lower RAM computers or if you are not using GPU.
# 
# ----
# ----

# In[251]:


df = df.sample(frac=0.2,random_state=101)
print(len(df))


# **TASK: Perform a train/test split with test_size=0.2 and a random_state of 101.**

# In[260]:


#Test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)

#Validation
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=141)


# ## Normalizing the Data
# 
# **TASK: Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data.**

# In[261]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# # Creating the Model
# 
# **TASK: Run the cell below to import the necessary Keras functions.**

# In[262]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping


# **TASK: Build a sequential model to will be trained on the data. You have unlimited options here, but here is what the solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron. OPTIONAL: Explore adding [Dropout layers](https://keras.io/layers/core/) [1](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) [2](https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab)**

# In[263]:


# CODE HERE
model = Sequential()

#Hidden layers
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, activation='relu'))

#Output layer
model.add(Dense(units=1, activation='sigmoid'))

#Compile
model.compile(loss = 'binary_crossentropy', optimizer='adam')


# In[264]:


#Early stopping
early_stop = EarlyStopping(monitor='val_loss',
                           mode = 'min',
                           verbose = 1, patience=25)


# **TASK: Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256.**

# In[265]:


model.fit(x=X_train, y=y_train,
          epochs = 600,
          validation_data = (X_val, y_val),
          callbacks = [early_stop],
          verbose = 1)


# **TASK: OPTIONAL: Save your model.**

# In[134]:


#Saving and loading model
from tensorflow.keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


# # Section 3: Evaluating Model Performance.
# 
# **TASK: Plot out the validation loss versus the training loss.**

# In[266]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot() 


# **TASK: Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# In[267]:


predictions = model.predict_classes(X_test)


# In[268]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# **TASK: Given the customer below, would you offer this person a loan?**

# In[269]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[271]:


new_customer = scaler.transform(new_customer.values.reshape(1,227))


# In[272]:


predict_new = model.predict_classes(new_customer)


# In[273]:


print(predict_new)


# **TASK: Now check, did this person actually end up paying back their loan?**

# In[274]:


df.iloc[random_ind]['loan_repaid']


# # GREAT JOB!
