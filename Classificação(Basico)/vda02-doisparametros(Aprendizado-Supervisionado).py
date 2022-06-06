from sklearn import tree

features = [[10],[20],[40],[60]]
labels = [1,2,4,6]

clas = tree.DecisionTreeClassifier()
clas = clas.fit(features, labels)

print(clas.predict([[100]]))

