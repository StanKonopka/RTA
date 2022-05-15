
import pickle

from flask import Flask, request
import pandas as pd
import numpy as np

app = Flask(__name__)

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    
iris_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
y = pd.read_csv(iris_path, names=col_names)
x = y.iloc[0:100, [0, 2]].values
y = y.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

Classifier = Perceptron()
Classifier.fit(x,y)

@app.route('/')
def iris_prediction():
    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    prediction = Classifier.predict(np.array([sepal_length, sepal_width]))
    if prediction == 1:
        species = 'versicolor'
    elif prediction == -1:
        species = 'setosa'
        
    return f'Sepal length is {sepal_length} cm and sepal width is {sepal_width} cm. Iris prediction is {prediction} - {species}.'

with open('model.pkl','rb') as picklefile:
    model = pickle.load(picklefile)
   
if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")

#Przyklady do sprawdzenia
#http://127.0.0.1:5000/?sepal_length=4.1&sepal_width=2.0
#http://127.0.0.1:5000/?sepal_length=6.5&sepal_width=7.3


# In[ ]:




