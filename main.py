import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Importing the data set
df = pd.read_csv('dataset.csv')
print(df.head())


'''Data processing and null value imputation'''

# Label encoding
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Dependents'].replace('3+', 3, inplace=True)
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Semiurban': 1, 'Urban': 2, 'Rural': 3})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Null value imputation
rev_null = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'LoanAmount', 'Loan_Amount_Term']
df[rev_null] = df[rev_null].replace({np.nan: df['Gender'].mode(),
                                     np.nan: df['Married'].mode(),
                                     np.nan: df['Dependents'].mode(),
                                     np.nan: df['Self_Employed'].mode(),
                                     np.nan: df['Credit_History'].mode(),
                                     np.nan: df['LoanAmount'].mean(),
                                     np.nan: df['Loan_Amount_Term'].mean()})

# Creating the train test split
X = df.drop(columns=['Loan_ID', 'Loan_Status']).values
y = df['Loan_Status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Printing the shapes
print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)
print("Shape of y_train", y_train.shape)
print("Shape of y_test", y_test.shape)


'''Building and evaluating the model'''

# Building the decision tree model
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

# Evaluating the model on training set by the f1_score
dt_pred_train = dt.predict(X_train)
print("Decision Tree Classifier's Training Set Evaluation f1_score: ", f1_score(y_train, dt_pred_train))

# Evaluating on the Test set
dt_pred_test = dt.predict(X_test)
print("Decision Tree Classifier's Testing Set Evaluation f1_score: ", f1_score(y_test, dt_pred_test))


# Building the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion='entropy', random_state=42)
rfc.fit(X_train, y_train)

# Evaluating on the Train set
rfc_pred_trian = rfc.predict(X_train)
print("Random Forest Classifier's Training Set Evaluation f1_score: ", f1_score(y_train, rfc_pred_trian))

# Evaluating on the Test set
rfc_pred_test = rfc.predict(X_test)
print("Random Forest Classifier's Testing Set Evaluation f1_score: ", f1_score(y_test, rfc_pred_test))


# Plotting the models
feature_importance = pd.DataFrame({
    'rfc': rfc.feature_importances_,
    'dt': dt.feature_importances_
}, index=df.drop(columns=['Loan_ID', 'Loan_Status']).columns)
feature_importance.sort_values(by='rfc', ascending=True, inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18, 8))
rfc_feature = ax.barh(index, feature_importance['rfc'], 0.4, color='purple', label='Random Forest')
dt_feature = ax.barh(index + 0.4, feature_importance['dt'], 0.4, color='lightgreen', label='Decision Tree')
ax.set(yticks=index + 0.4, yticklabels=feature_importance.index)

ax.legend()
plt.show()
