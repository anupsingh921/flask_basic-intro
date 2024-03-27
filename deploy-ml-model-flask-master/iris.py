import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Read the CSV file with the correct file path
df = pd.read_csv('C:/Users/anupk/Downloads/deploy-ml-model-flask-master/deploy-ml-model-flask-master/iris.data')

# Separate features (X) and target variable (y)
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4])

# Apply label encoding to the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Support Vector Classifier (SVC)
sv = SVC(kernel='linear').fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('iri.pkl', 'wb') as file:
    pickle.dump(sv, file)
