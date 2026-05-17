import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

iris_data = pd.read_excel('iris_dataset.xlsx')

print("========= DATASET LOADED SUCCESSFULLY =========")

print("========= FIRST 5 ROWS =========")
print(iris_data.head())

print("========= DATASET INFORMATION =========")
print(iris_data.info())

print("========= DATASET SHAPE =========")
print(iris_data.shape)

print("========= UNIQUE CLASSES =========")
print(iris_data['species'].unique())

X = iris_data[[
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]]

y = iris_data['species']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


predictions = model.predict(X_test)

print("========= PREDICTIONS =========")
print(predictions)


accuracy = accuracy_score(y_test, predictions)

print("========= MODEL ACCURACY =========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("========= CLASSIFICATION REPORT =========")
print(classification_report(y_test, predictions))


cm = confusion_matrix(y_test, predictions)

print("========= CONFUSION MATRIX =========")
print(cm)

plt.figure(figsize=(8, 5))

species_codes = pd.factorize(y)[0]

plt.scatter(
    iris_data['sepal_length'],
    iris_data['petal_length'],
     c=species_codes
)

plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Iris Dataset Visualization')

plt.show()


sample_data = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample_data)

print("========= CUSTOM PREDICTION =========")
print("Predicted Flower:", prediction[0])
