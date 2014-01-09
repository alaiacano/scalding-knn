import numpy as np
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target
idx = np.arange(len(y))

Xtrain = X[idx % 3 != 0]
Xtest = X[idx % 3 == 0]

ytrain = y[idx % 3 != 0]
ytest = y[idx % 3 == 0]


h = .02  # step size in the mesh


clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', algorithm='brute')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
print sum(ytest == ypred)
print sum(ytest != ypred)
print zip(ytest[ypred!=ytest], ypred[ypred!=ytest])
