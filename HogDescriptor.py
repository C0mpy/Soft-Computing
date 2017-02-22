from skimage.color import rgb2gray
import numpy as np

class HOGDescriptor(object):


    def __init__(self, pxl_per_cell):

        self.pxl_per_cell = pxl_per_cell


    # racunanje vrednosti gradijenata za sve pixele, na osnovu gradijenata njihove intezitete i uglove
    # vraca tri dvodimenzionalne matrice:
    # gradients - svaki element predstavlja gradijent za pixel (dve vrednosti po x i y osi)
    # intensities - svaki element predstavlja intezitet vektora
    # angles - svaki element predstavlja ugao vektora
    def img_gradients(self, img):

        # strukture koje vracamo na kraju
        gradients = []
        intensities = []
        angles = []

        # iterisemo od drugog do predposlednjeg reda - prvi red nema susedni gornji, a poslednji red nema susedni donji
        for i in range(self.n_rows):

            gradients.append([])
            intensities.append([])
            angles.append([])

            # iterisemo od druge do predposlednje kolone, prva nema susednu levu, a poslednja susednu desnu kolonu
            for j in range(self.n_cols):
                # racunamo gradijent po x i y osi za odredjeni pixel
                grad = self.pxl_gradients(img, i, j)
                # racunamo intezitet vektora
                gradients[i].append(grad)
                intensity = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
                intensities[i].append(intensity)
                # ako je gradijent po x = 0 to nam je problem kod racunanja ugla jer bi se delilo sa nulom
                # zato gradijent postavljamo na vrlo mali broj koji je razlicit od nula ali nece uticati mnogo na rez
                # racunamo ugao vektora
                angle = np.degrees(np.arctan2(grad[1], grad[0]+1e-15)) % 180
                angles[i].append(angle)

        return gradients, intensities, angles

    # racunanje gradijenta po x i y za svaki pixel vraca listu sa dva elementa
    def pxl_gradients(self, img, row, col):

        y = 0
        x = 0

        # ako je element iz prve kolone
        if col == 0:
            # onda gradijent po x racunamo: sledeci pixel - trenutni
            x = img[row][col + 1] - img[row][col]
        # ako je element iz poslednje kolone
        elif col == self.n_cols - 1:
            # onda gradijent po x racunamo: trenutni pixel - prosli
            x = img[row][col] - img[row][col - 1]
        # za sve ostale slucajeve
        else:
            # gradijent po x racunamo: (sledeci - prosli)/2
            x = (img[row][col + 1] - img[row][col - 1]) / 2

        # ako je element iz prvog reda
        if row == 0:
            # onda gradijent po y racunamo: sledeci pixel - trenutni
            y = img[row + 1][col] - img[row][col]
        # ako je element iz poslednjeg reda
        elif row == self.n_rows - 1:
            # onda gradijent po y racunamo: trenutni pixel - prosli
            y = img[row][col] - img[row - 1][col]
        # za sve ostale slucajeve
        else:
            # gradijent po y racunamo: (sledeci - prosli)/2
            y = (img[row + 1][col] - img[row - 1][col]) / 2


        return [x, y]

    # kreiranje histograma za sve celije
    def cell_histo(self, angles, intensities):

        histo = []
        # iteratori po redovima matrice uglova i inteziteta vektora
        row = 0
        # pomocna promenljiva za pristup elementu histograma
        i = 0
        # dok ne stignemo do poslednjeg reda u matrici uglova
        while row != len(angles):
            # dodati prazan red u matricu histogram
            histo.append([])
            # iteratori po kolonama matrice uglova i inteziteta vektora
            col = 0
            # dok ne stignemo do poslednje kolone u matrici uglova
            while col != len(angles[0]):

                # inicijalizovati prazan histogram za ovu celiju
                cell_histo = [0]*9

                # pristupamo elementima celije 8x8 pixela
                for r in range(row, row + self.pxl_per_cell[0]):
                    for c in range(col, col + self.pxl_per_cell[1]):

                        # ugao koji zelimo da dodajemo
                        angle = angles[r][c]

                        if angle < 180 and angle >= 0:
                            # index bina kome dodajemo
                            floor = int(angle // 20)
                            cell_histo[floor] += intensities[r][c]

                for r in range(len(cell_histo)):
                    cell_histo[r] = cell_histo[r] / 64

                # dodajemo histogram celije u rezultujuci histogram
                histo[i].append(cell_histo)
                col += self.pxl_per_cell[1]
            row += self.pxl_per_cell[0]
            i += 1

        return histo


    def block_create(self, histogram):

        blocks = []

        row = 0
        col = 0

        while row + 1 != len(histogram):
            col = 0
            while col + 1 != len(histogram[0]):

                block = histogram[row][col] + histogram[row][col + 1] + histogram[row + 1][col] + histogram[row + 1][col + 1]
                k = np.sum(np.abs(block)) + 1e-5
                for i in range(len(block)):
                    block[i] = block[i] / k
                blocks += block
                col += 1
            row += 1
        return blocks


    def describe(self, img):

        img_gray = rgb2gray(img)

        self.n_rows = len(img_gray)
        self.n_cols = len(img_gray[0])

        gradients, intensities, angles = self.img_gradients(img_gray)

        histo = self.cell_histo(angles, intensities)

        blocks = self.block_create(histo)

        return blocks