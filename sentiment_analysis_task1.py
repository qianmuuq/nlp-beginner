import csv
import matplotlib.pyplot
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def read_tsv(filename):
    with open(filename,encoding="utf-8") as f:
        tsvreader = csv.reader(f, delimiter='\t')
        temp = list(tsvreader)
        return temp

class Softmax:
    def __init__(self,batch_flag):
        self.batch_flag = batch_flag
        #0:Shuffle,1:Batch,2:mini-batch

    def gradient_batch_type(self,feature_data, label_data, labels, epoch, alpha,test_data,label_test):
        m, n = np.shape(feature_data)
        weights = np.mat(np.random.normal(0.0, 1.0, (n, labels)))
        i = 0
        train_acc,test_acc = 0.0,0.0
        if self.batch_flag == 0:
            while i <= epoch*m:
                feature_data_s,label_data_s = feature_data[i % m], label_data[i % m]
                error = np.exp(feature_data_s * weights)
                if i % (m*epoch) == 0 and i!=0:
                    print("shuffle-----iter: " + str(i / m) + ", loss: " + str(self.loss(np.exp(feature_data * weights), label_data)), end='')
                    error1 = np.exp(feature_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    train_acc = self.acc(error1 * -1, label_data)[0, 0]
                    print(",Accuracy:", train_acc)
                    # test
                    print("test: ", end='')
                    error1 = np.exp(test_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    test_acc = self.acc(error1 * -1, label_test)[0, 0]
                    print(",Accuracy:", test_acc)

                rowsum = -error.sum(axis=1)
                rowsum = rowsum.repeat(labels, axis=1)
                error = error / rowsum

                error[0, label_data_s] += 1
                weights = weights + alpha * feature_data_s.T * error  # 梯度更新
                i+=1
        elif self.batch_flag == 1:
            while i <= epoch:
                error = np.exp(feature_data * weights)
                if i % (epoch) == 0 and i!=0:
                    print("batch-----iter: " + str(i) + ", loss: " + str(self.loss(error, label_data)), end='')
                rowsum = -error.sum(axis=1)
                rowsum = rowsum.repeat(labels, axis=1)
                error = error / rowsum
                if i % (epoch) == 0 and i!=0:
                    train_acc = self.acc(error * -1, label_data)[0, 0]
                    print(",Accuracy:", train_acc)
                    # test
                    print("test: ", end='')
                    error1 = np.exp(test_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    test_acc = self.acc(error1 * -1, label_test)[0, 0]
                    print(",Accuracy:", test_acc)

                for x in range(m):
                    error[x, label_data[x]] += 1
                weights = weights + (alpha / m)* feature_data.T * error  # 梯度更新
                i+=1
        else:
            increment = np.zeros((n,labels))
            while i <= epoch*m:
                feature_data_s, label_data_s = feature_data[i % m], label_data[i % m]
                error = np.exp(feature_data_s * weights)
                if i % (epoch*m) == 0 and i!=0:
                    #train
                    print("mini_batch-----iter: " + str(i / (m)) + ", loss: " + str(
                        self.loss(np.exp(feature_data * weights), label_data)), end='')
                    error1 = np.exp(feature_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    train_acc = self.acc(error1 * -1, label_data)[0, 0]
                    print(",Accuracy:", train_acc)
                    #test
                    print("test: " , end='')
                    error1 = np.exp(test_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    test_acc = self.acc(error1 * -1, label_test)[0, 0]
                    print(",Accuracy:", test_acc)
                rowsum = -error.sum(axis=1)
                rowsum = rowsum.repeat(labels, axis=1)
                error = error / rowsum

                error[0, label_data_s] += 1
                increment += feature_data_s.T * error

                if i%100 == 0:
                    weights = weights + (alpha/100) * increment  # 梯度更新
                    increment = 0
                i += 1

        return train_acc,test_acc

    def loss(self,err, label_data):
        m = np.shape(err)[0]
        sum_loss = 0.0
        for i in range(m):
            if err[i, label_data[i]] / np.sum(err[i, :]) > 0:
                sum_loss -= np.log(err[i, label_data[i]] / np.sum(err[i, :]))

        return sum_loss / m

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

if __name__ == '__main__':
    data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/train.tsv')[1:1500]
    # data_test = read_tsv('./data/sentiment-analysis-on-movie-reviews/test.tsv')[1:]
    X,y = [],[]
    for i in data_train:
        X.append(i[2].lower())
        y.append(int(i[3]))
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
    vect = CountVectorizer()
    vect.fit_transform(X)
    # veccc = vect.vocabulary_
    # print(len(veccc))
    X_train = vect.transform(X_train)
    X_test = vect.transform(X_test)

    alphas = [0.03,0.5,1.0,5.0,10.0,100.0]
    #type
    list_train_s, list_test_s = [[], [], []], [[], [], []]

    for j in range(6):
        for i in range(3):
            soft = Softmax(i)
            train_s,test_s = soft.gradient_batch_type(X_train, y_train, 5, 20, alphas[j],X_test,y_test)
            list_train_s[i].append(train_s),
            list_test_s[i].append(test_s)
    matplotlib.pyplot.subplot(2, 2, 1)
    matplotlib.pyplot.semilogx(alphas, list_train_s[0], 'r--', label='shuffle')
    matplotlib.pyplot.semilogx(alphas, list_train_s[1], 'g--', label='batch')
    matplotlib.pyplot.semilogx(alphas, list_train_s[2], 'b--', label='mini-batch')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("Bag of words -- Training Set")
    matplotlib.pyplot.xlabel("Learning Rate")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0.2, 1.0)
    matplotlib.pyplot.subplot(2, 2, 2)
    matplotlib.pyplot.semilogx(alphas, list_test_s[0], 'r--', label='shuffle')
    matplotlib.pyplot.semilogx(alphas, list_test_s[1], 'g--', label='batch')
    matplotlib.pyplot.semilogx(alphas, list_test_s[2], 'b--', label='mini-batch')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("Bag of words -- Test Set")
    matplotlib.pyplot.xlabel("Learning Rate")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0.2, 1.0)
    matplotlib.pyplot.tight_layout()

    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
    vect1 = CountVectorizer(ngram_range=(1,2))
    vect1.fit_transform(X)
    # veccc = vect1.vocabulary_
    # print(len(veccc))
    X_train_n = vect1.transform(X_train_n)
    X_test_n = vect1.transform(X_test_n)

    list_train_s, list_test_s = [[], [], []], [[], [], []]
    for j in range(6):
        for i in range(3):
            soft = Softmax(i)
            train_s, test_s = soft.gradient_batch_type(X_train_n, y_train_n, 5, 20, alphas[j], X_test_n, y_test_n)
            list_train_s[i].append(train_s)
            list_test_s[i].append(test_s)
    matplotlib.pyplot.subplot(2, 2, 3)
    matplotlib.pyplot.semilogx(alphas, list_train_s[0], 'r--', label='shuffle')
    matplotlib.pyplot.semilogx(alphas, list_train_s[1], 'g--', label='batch')
    matplotlib.pyplot.semilogx(alphas, list_train_s[2], 'b--', label='mini-batch')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("N gram -- Training Set")
    matplotlib.pyplot.xlabel("Learning Rate")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0.2, 1.0)
    matplotlib.pyplot.subplot(2, 2, 4)
    matplotlib.pyplot.semilogx(alphas, list_test_s[0], 'r--', label='shuffle')
    matplotlib.pyplot.semilogx(alphas, list_test_s[1], 'g--', label='batch')
    matplotlib.pyplot.semilogx(alphas, list_test_s[2], 'b--', label='mini-batch')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("N gram -- Test Set")
    matplotlib.pyplot.xlabel("Learning Rate")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0.2, 1.0)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

