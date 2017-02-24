import numpy as np

class KNNClassifier(object):

    def __init__(self, k, distance='L2'):
        """
        Konstruktor KNN klasifikatora.

        :param k: koliko najblizih suseda uzeti u obzir.
        :param distance: koju udaljenost koristiti za poredjenje, moguce vrednosti su 'L1' i 'L2'
        """
        self.k = k
        self.distance = distance


    def fit(self, X, y):
        """
        :param X: podaci
        :param y: labele klasa (celobrojne vrednosti >= 0)
        :return: None
        """
        self.data = X
        self.labels = y

    def predict(self, X):
        """
        :param X: podaci za klasifikaciju
        :return: labele X podataka nakon klasifikacije
        """
        min_vals = [100000] * self.k
        min_indices = [100000] * self.k
        for i in range(len(self.data)):

            # racunanje euklidske udaljenosti
            summ = 0
            for j in range(len(self.data[i])):
                summ += (self.data[i][j] - X[j]) ** 2
            summ = summ ** 0.5

            # ako je ova euklidska udaljenost jedan od k najmanjih, smestamo njenu vrednost u listu min_vals i
            # index elementa kome ta euklidska udaljenost pripada u listu min_indices kako bi mogli da dobijemo labele
            temp_i = i
            for k in range(self.k):
                if summ < min_vals[k]:
                    temp = min_vals[k]
                    min_vals[k] = summ
                    summ = temp

                    temp_i1 = min_indices[k]
                    min_indices[k] = temp_i
                    temp_i = temp_i1

        # izbrojati kojoj labeli pripada k najblizih vrednosti
        res = {}
        for i in range(len(min_indices)):
            if res.get(self.labels[min_indices[i]]) == None:
                res[self.labels[min_indices[i]]] = 1
            else:
                res[self.labels[min_indices[i]]] += 1

        return max(res, key=res.get)