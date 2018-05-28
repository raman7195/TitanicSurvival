
# coding: utf-8

# ### IMPORTING important MODULES

# In[ ]:

# data analysis and wrangling
import numpy as np
import pandas as pd
import random as rnd

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC , LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = [train,test]


# In[ ]:

train.head()


# In[ ]:

test.head()


# In[ ]:

train.describe()


# In[ ]:

test.describe()


# In[ ]:

test.info()


# In[ ]:

train.info()


# In[ ]:

train.describe(include=['O'])


# ## Finding correlation in Pclass and Survived

# In[ ]:

train[['Pclass','Survived']].groupby(['Pclass'],
                                           as_index=False).mean().sort_values(
    by = 'Survived', ascending = True)


# ## correlation in Sex and Survived

# In[ ]:

train[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# ## Correlation in SibSp and Survived

# In[ ]:

train[['SibSp','Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived',ascending = False)


# In[ ]:

## Correlation in Parch and Survived


# In[ ]:

train[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# # Analysis through Visualization

# In[ ]:

g = sns.FacetGrid(train, col = 'Survived')
g.map(plt.hist,'Age',bins = 20)


# In[ ]:

grid1 = sns.FacetGrid(train, col = 'Pclass',hue = 'Survived')
grid = sns.FacetGrid(train,col = 'Survived', row = 'Pclass', size =2.2, aspect = 1.6) 
grid.map(plt.hist,'Age',alpha=1,bins=20)
grid1.map(plt.hist,'Age',alpha =1,bins=20)
grid1.add_legend();


# In[ ]:

grid = sns.FacetGrid(
    train_df , row = 'Embarked', size = 3.2, aspect = 1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette = 'deep')
grid.add_legend()




# ## Dropping unnecessary features

# In[ ]:

print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)


test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]

"After", train.shape, test.shape, combine[0].shape, combine[1].shape


# ## Creating new features extracting from existing

# #### Looking at the correlation between the Title and the survival,extracting the Titles here below.

# In[ ]:

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False) #expand = False reurns a dataset
pd.crosstab(train['Title'],train['Sex'])    
    


# ### Replace many titles with the common once or classify them as Rare

# In[ ]:

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('MS','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train[['Title','Survived']].groupby(['Title'], as_index=False).mean()    
    


# ####  We can convert the categorical titles to ordinal

# In[ ]:

title_mapping = {"Mr" :1, "Miss":2, "Mrs":3,"Master":4,"Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train.head()    


# ####  We can now safely drop the Name feature from training and testing dataset, we also donot require the pasenger ID anymore

# In[ ]:

train = train.drop(['Name','PassengerId'], axis = 1)
test = test.drop(['Name'],axis=1)
combine = [train,test]
train.shape,test.shape


# ### Completing a categorical feature
# ##### Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.

# In[ ]:

#converting sex feature to new feature called gender where m=0,f=1
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)
train.head()    


# ### Completing a numerical continous feature

# In[ ]:

#grid1 = sns.FacetGrid(train, col = 'Pclass',hue = 'Gender')
grid = sns.FacetGrid(train,row='Pclass', col = 'Sex',size=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=0.5,bins=20)
#grid1.map(plt.hist,'Age',alpha=.5,bins=20)
#grid1.add_legend()


# #### Let us start by preparing an empty array to contain guessed Age values based on Pclass X Gender combinations

# In[ ]:

guess_ages = np.zeros((2,3))
guess_ages


# #### Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[ ]:

for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) &
                               (dataset['Pclass']== j+1)]['Age'].dropna()
            #age_mean=guess_df.mean()
            #age_std=guess_df.std()
            #age_guess = rnd.uniform(age_mean-age_std,age_mean+age_std)
            age_guess=guess_df.median()
            
            #Convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5
        
        for i in range(0,2):
            for j in range(0,3):
                dataset.loc[(dataset.Age.isnull()) & 
                            (dataset.Sex == i) & 
                            (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
            
            dataset['Age'] = dataset['Age'].astype(int)
            
train.head()            
            


# ### Let us create the Age bands and determine the correlation with the survived 
# 

# In[ ]:

train['AgeBand'] = pd.cut(train['Age'],5)
train[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by='AgeBand', ascending = True)


# #### Let us replace Age with ordinals based on these bands

# In[ ]:

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16,'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=32),'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <=48),'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <=64),'Age'] = 3
    dataset.loc[dataset['Age']>64,'Age'] = 4
train.head()    
    


# In[ ]:

# Removing the Ageband feature as we donot need it.
train = train.drop(['AgeBand'],axis = 1)
combine = [train,test]
train.head()


# ### Create new feature combining existing features
# ##### new feature for FamilySize which combines Parch and SibSp

# In[166]:

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] +dataset['Parch'] + 1
train[['FamilySize','Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived',ascending = False)


# In[169]:

# Creating another feature called IsAlone
for dataset in combine:
    dataset['IsAlone']=0
    dataset.loc[dataset['FamilySize'] == 1,'IsAlone']=1
train[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean()    


# In[170]:

#Dropping the features Parch,SibSp and Family size in favor of IsAlone
train = train.drop(['Parch','SibSp','FamilySize'], axis = 1)
combine = [train,test]
train.head()


# In[173]:

#Completing the categorical feature

freq_port = train.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)    


# In[174]:

# converting categorical feature to numeric

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
train.head()    


# In[175]:

#Completing and converting a numeric feature

test['Fare'].fillna(test['Fare'].dropna().median(), inplace = True)
test.head()


# In[176]:

# Creating a fare band 
train['FareBand'] = pd.qcut(train['Fare'],4)
train[['FareBand','Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived',ascending=True)


# In[181]:

# Convert the Fare feature to ordinal values based on the FareBand

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train= train.drop(['FareBand'], axis=1)
combine = [train, test]
    
train.head(10)


# In[187]:


test.head(10)


# In[189]:

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[190]:

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[192]:

submission = pd.DataFrame({
                           "PassengerID":test["PassengerId"],
                            "Survived": Y_pred
                        })

submission.head()


# In[195]:

# creating a submission file
submission.to_csv('submission.csv', index=False)

