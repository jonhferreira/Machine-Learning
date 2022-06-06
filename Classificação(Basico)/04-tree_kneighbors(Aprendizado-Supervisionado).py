from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
flor = load_iris()

features = flor.data
labels = flor.target

feat_train, feat_test, label_train, label_test = train_test_split(features, labels, test_size =.5)

# classificador com arvore de decisao
clastree = DecisionTreeClassifier()
clastree.fit(feat_train, label_train)

pre_t = clastree.predict(feat_test)

print("tree")
print("%.2f"%accuracy_score(pre_t,label_test))
print()


# classificador de vizinho mais proximo
clasneigh = KNeighborsClassifier()
clasneigh.fit(feat_train, label_train)

pre_n = clasneigh.predict(feat_test)

print("neighbors")
print("%.2f"%accuracy_score(pre_n,label_test))

