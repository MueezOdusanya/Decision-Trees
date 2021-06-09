import sklearn
import sklearn.datasets
import sklearn.metrics
from sklearn import tree
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.datasets import fetch_openml
def main():
    d = fetch_openml(data_id=1504)

    mytree = tree.DecisionTreeClassifier(criterion="entropy")
    mytree = mytree.fit(d.data, d.target)
    predicted = mytree.predict(d.data)
    print(sklearn.metrics.accuracy_score(d.target, predicted))
    mytree = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1)
    mytree = mytree.fit(d.data, d.target)

    predicted = mytree.predict(d.data)

    scores = model_selection.cross_val_score(mytree, d.data, d.target, cv=10)
    print(sklearn.metrics.accuracy_score(d.target,predicted),scores.mean())
    
main()
