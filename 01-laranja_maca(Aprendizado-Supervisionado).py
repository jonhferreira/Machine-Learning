from sklearn import tree

frutas  = {1:"maca", 2:"laranja"}

#caracteristicas das frutas
features = [[140,1], [130,1], [150,2],[170,2]]

#identificadores das frutas
labels = [1,1,2,2]

#cria classificador(arvore de decisao)
clas = tree.DecisionTreeClassifier()

#treina o classificador
clas = clas.fit(features, labels)

laranja = [120,1]

#prediz o resultado baseado nos dados de entrada
fruta = clas.predict([laranja])[0]

print(frutas[fruta])
