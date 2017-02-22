from __future__ import print_function
import random
import matplotlib.pyplot as plt

class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []  # podaci koji pripadaju ovom klasteru

    def recalculate_center(self):
        # TODO 1: implementirati racunanje centra klastera
        # centar klastera se racuna kao prosecna vrednost svih podataka u klasteru
        sum_x = 0
        sum_y = 0
        
        for d in self.data:
            sum_x += d[0]
            sum_y += d[1]

        self.center[0] = sum_x / len(self.data)
        self.center[1] = sum_y / len(self.data)
        
class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        """
        :param n_clusters: broj grupa (klastera)
        :param max_iter: maksimalan broj iteracija algoritma
        :return: None
        """
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data, solution):
        self.data = data  # lista N-dimenzionalnih podataka
        self.solution = solution    # lista
        # TODO 4: normalizovati podatke pre primene k-means

        # TODO 1: implementirati K-means algoritam za klasterizaciju podataka
        # kada algoritam zavrsi, u self.clusters treba da bude "n_clusters" klastera (tipa Cluster)
        
        
        #odrediti min i max moguce vrednosti za lokacije sredine klastera
        max_x = float("-inf")
        min_x = float("inf")
        max_y = float("-inf")
        min_y = float("inf")
        
        for d in data:
            if(d[0] < min_x):
                min_x = d[0]
            if(d[0] > max_x):
                max_x = d[0]
            if(d[1] < min_y):
                min_y = d[1]
            if(d[1] > max_y):
                max_y = d[1]

        # napraviti klastere
        for i in range(self.n_clusters):
            self.clusters.append(Cluster([random.uniform(min_x, max_x), random.uniform(min_y, max_y)]))
            
        for i in range(self.max_iter):
            for d in self.data:
                self.clusters[self.predict(d)].data.append(d)
            
            for c in self.clusters:
                c.recalculate_center()
                
            colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
            plt.figure()
            for idx, cluster in enumerate(self.clusters):
                plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=200)  # iscrtavanje centara
                for datum in cluster.data:  # iscrtavanje tacaka
                    plt.scatter(datum[0], datum[1], c=colors[idx])

            plt.xlabel('Sepal width')
            plt.ylabel('Petal length')
            plt.show()

            
        # TODO (domaci): prosiriti K-means da stane ako se u iteraciji centri klastera nisu pomerili

    def predict(self, datum):
        # TODO 1: implementirati odredjivanje kom klasteru odredjeni podatak pripada
        # podatak pripada onom klasteru cijem je centru najblizi (po euklidskoj udaljenosti)
        # kao rezultat vratiti indeks klastera kojem pripada
        temp = -1
        min_dist = float("inf")
        for i in range(len(self.clusters)):
            dist = (self.clusters[i].center[0] - datum[0]) ** 2 + (self.clusters[i].center[1] - datum[1]) ** 2
            if(dist < min_dist):
                min_dist = dist
                temp = i
        return temp
            

    def sum_squared_error(self):
        # TODO 3: implementirati izracunavanje sume kvadratne greske
        # SSE (sum of squared error)
        # unutar svakog klastera sumirati kvadrate rastojanja izmedju podataka i centra klastera
        raise NotImplementedError()
