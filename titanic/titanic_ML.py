"""
TITANIC: MACHINE LEARNING FROM DISASTER

A python script that performs data munging
and regression/classification to predict
the liklihood of survival for passengers
onboard the S.S. Titanic. 

This enables us to define key survival
indicators such as gender, age and fare.
The original dataset contains missing
values and so a clean up operation is the
first step before the machine learning
predictive models are applied.

Ben Stanley
Dec' 2015
    
"""
import pandas as pd
import numpy as np
import csv as csv
import RegTools as lrt
from sklearn import svm
from sklearn import linear_model
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

######################################################################################
# Load Training and Test Data #
######################################################################################

titanic_train = pd.read_csv('train.csv', header=0)
titanic_test = pd.read_csv('test.csv', header=0)

# Save indexes for output file
ids = titanic_test['PassengerId'].values

# Merge data to perfrom munging on entire train and test set
all_data = pd.concat([titanic_train, titanic_test], ignore_index=True)

######################################################################################
# Clean 'Fare' Data #
######################################################################################
"""
Find the mean value of the fare paid for each class
and replace zero values
"""

# Convert zero values into 'NaN'
all_data.Fare = all_data.Fare.map(lambda x: np.nan if x==0 else x)
# Create a pivot table that aggregates the mean of the fare for each class group
class_mean =  all_data.pivot_table('Fare', index='Pclass', aggfunc='mean')
# Replace all null values with the mean of specific classes
all_data.Fare = all_data[['Fare', 'Pclass']].apply(lambda x: class_mean[x['Pclass']]
                                    if pd.isnull(x['Fare']) else x['Fare'], axis=1 )

######################################################################################
# Fit Linear Model For Age #
######################################################################################

"""    
Locate and separate data rows
with NaN ages values
"""
# Find index values with a 'null' value for age
no_age_idx = all_data.loc[all_data.Age.isnull(), 'Age'].index.tolist()

# Collect the data with missing age values
no_age = all_data.loc[no_age_idx]

# Drop unused data from the original dataset
age_exists = all_data.drop(all_data.index[no_age_idx])
age_exists = age_exists.drop(['PassengerId','Survived', 'Name', 'Sex',
                              'Ticket', 'Cabin', 'Embarked'], axis=1)

no_age = no_age.drop(['PassengerId', 'Survived', 'Name', 'Sex',
                      'Ticket', 'Cabin', 'Embarked'], axis=1)

"""
Age Training Data
"""
y_train_age = np.asarray(age_exists['Age'])
X_train_age = np.asarray(age_exists.drop(['Age'], axis=1))

"""
Age Test Data
"""
X_test_age = np.asarray(no_age)
X_test_age = np.delete(X_test_age, 0, axis=1)

"""
Fit linear model to age data & fill in missing values
"""
ageModel = linear_model.LinearRegression(fit_intercept=True, normalize=True)
ageModel.fit(X_train_age, y_train_age)

pred_ages = ageModel.predict(X_test_age)

all_data.loc[no_age_idx, 'Age'] = pred_ages

######################################################################################
# Convert Useful String Data to Ints #
######################################################################################
"""
Convert the 'Sex' column of strings into binary integer
values 'Gender' female = 0, Male = 1
"""
all_data['Gender'] = all_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

"""
Complete missing values in the embarked embarked column,
fill in with the most common embarked port (modal average)
and convert to integer.
"""
# Fill in missing values
all_data.Embarked[ all_data.Embarked.isnull() ] = all_data.Embarked.dropna().mode().values

# Determine each unique port
Ports = list(enumerate(np.unique(all_data['Embarked'])))

# set up a dictionary in the form  Ports : index
Ports_dict = { name : i for i, name in Ports }

# Convert all Embark strings to int
all_data.Embarked = all_data.Embarked.map( lambda x: Ports_dict[x]).astype(int)

######################################################################################
# Engineer New Features #
######################################################################################
"""
Create 'Family Size' feature, which is Siblings + Spouse + Parent
"""
all_data['FamilySize'] = all_data.Parch + all_data.SibSp

"""
Create 'AgeClass' feature, which is Age * Class
"""
all_data['AgeClass'] = all_data.Age * all_data.Pclass

######################################################################################
# Define  Datasets #
######################################################################################

# Drop unused features
all_data = all_data.drop(['PassengerId', 'Name', 'Sex',
                          'Ticket', 'Cabin', ], axis=1)

# Slice data back into train and test sets
train_df = all_data.loc[0:890]
test_df = all_data.loc[891::]

# Split training data into input and target
y = np.asarray(train_df['Survived'])
X = np.asarray(train_df.drop(['Survived'],axis=1))

# Normalise and add a column of ones for the intercept term
X_norm = lrt.normalise_features(X)

# Declare the training set
X_train = X_norm[0:-200, :]
y_train = y[0:-200]

# Declare the cross validation set
X_valid = X_norm[-200::, :]
y_valid = y[-200::]

# Declare the test set
X_test = np.asarray(test_df.drop(['Survived'], axis=1))

# Normalise and add a column of ones for the intercept term
X_test_norm = lrt.normalise_features(X_test)

######################################################################################
# Logistic Regression #
######################################################################################
"""
Use cross-validation set to find
optimum parameter for 'C'
"""

lamda = np.arange(0.1, 10, 0.3)
lr_costs = []

for i in range(0, len(lamda)):

    logReg = linear_model.LogisticRegression(penalty='l2', C=lamda[i])
    logReg.fit(X_train, y_train)
    lr_costs.append(logReg.score(X_valid, y_valid))


max_valid_idx = lr_costs.index(max(lr_costs))
best_lamda = lamda[max_valid_idx]

print "Logistic Regression: optimum 'lamda' value = ", best_lamda

"""
Train based on the optimum parameter 'C'
"""
logReg = linear_model.LogisticRegression(penalty='l2', C=best_lamda)
logReg.fit(X_norm, y)
logReg_predictions = logReg.predict(X_test_norm)
lr_score = logReg.score(X_norm, y)

print "Logistic Regression Score = ", lr_score
print "--------------------------------------"

######################################################################################
# Support Vector Machine #
######################################################################################
"""
Use cross-validation set to find
optimum parameter for 'C'
"""

c = np.arange(0.1, 10, 0.3)
svm_costs = []

for i in range(0, len(c)):
    
    svm_test = svm.SVC(C=c[i])
    svm_test.fit(X_train, y_train)
    svm_costs.append(svm_test.score(X_valid, y_valid))


svm_max_idx = svm_costs.index(max(svm_costs))
best_c = c[svm_max_idx]

print "Support Vector Machine: optimum 'C' value = ", best_c

"""
Train based on the optimum parameter 'C'
"""
clf = svm.SVC(C=100)
clf.fit(X_norm, y)
svm_predictions = clf.predict(X_test_norm)
svm_score = clf.score(X_norm, y)

print "SVM Accuracy on Training Set = ", svm_score
print "--------------------------------------"

######################################################################################
# Save the results to a .csv file #
######################################################################################
"""
Choose the model that shows the best score
"""

if lr_score > svm_score:
    print "Saving results from logistic regression"
    predictions = logReg_predictions
else:
    print "Saving results from support vector machine"
    predictions = svm_predictions

"""
Save results to file
"""
submission = list(zip(ids, predictions.astype(int)))
df = pd.DataFrame(data = submission, columns=['PassengerId', 'Survived'])
df.to_csv('titanic_results.csv', index=False, header=True)

print "COMPLETE!"
print "Please check working directory for results file"
print "--------------------------------------"