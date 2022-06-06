from sklearn.datasets import load_iris

flores = load_iris()

# nomes das caracteristicas
print(flores.feature_names)

# nome dos identificadores
print(flores.target_names)

# features
print(flores.data[0])

# labels associados a features
print(flores.target[0])

# tabela inteira

for i in range(len(flores.data)):
    print("%i - features: %s   label: %s   name: %s" %(i,flores.data[i], flores.target[i], flores.target_names[flores.target[i]]))
    

