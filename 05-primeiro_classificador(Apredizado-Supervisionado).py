from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

flor = load_iris()

features = flor.data
labels = flor.target


feat_train, feat_test, label_train, label_test = train_test_split(features, labels, test_size=.5)

def d_euc(x,y):
        return distance.euclidean(x,y)
    
class k_neigh():

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, x_test):
        predictions = []
        
        for col in x_test:
            label = self.mais_proximo(col)
            predictions.append(label)

        return predictions

    def mais_proximo(self, col):
        menor_dist = d_euc(col,self.X_train[0])
        ind = 0

        for i in range(1, len(self.X_train)):
            dist = d_euc(col,self.X_train[i])

            if (dist < menor_dist):
                menor_dist = dist
                ind = i

        return self.Y_train[ind]

prim_clas = k_neigh()

prim_clas.fit(feat_train, label_train)

prev = prim_clas.predict(feat_test)

print(accuracy_score(prev, label_test))
    
