from sklearn import datasets
from sklearn.svm import SVC
import joblib

# Exemple simple
X, y = datasets.load_iris(return_X_y=True)
model = SVC()
model.fit(X, y)

joblib.dump(model, "app/model.joblib")
