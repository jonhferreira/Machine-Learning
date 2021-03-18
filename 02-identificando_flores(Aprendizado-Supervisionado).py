from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

flores = load_iris()

# casos que serao retirados
ind_ret = [0,50,100]

#casos de treinamento
train_target = np.delete(flores.target, ind_ret)
train_data = np.delete(flores.data, ind_ret, axis=0)

#casos de teste
test_target = flores.target[ind_ret]
test_data = flores.data[ind_ret]

# criacao de classificador e seu treinamento com os dados

clas = tree.DecisionTreeClassifier()

clas = clas.fit(train_data, train_target)

print("Previsto")
print(clas.predict(test_data))

print()

print("Real")
print(test_target)

