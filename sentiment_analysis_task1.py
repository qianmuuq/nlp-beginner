import csv
import matplotlib.pyplot
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def read_tsv(filename):
    with open(filename,encoding="utf-8") as f:
        tsvreader = csv.reader(f, delimiter='\t')
        temp = list(tsvreader)
        return temp

class Softmax:
    def __init__(self,batch_flag):
        self.batch_flag = batch_flag
        #0:Shuffle,1:Batch,2:mini-batch

    def loss(self,err, label_train):
        m = np.shape(err)[0]
        sum_loss = 0.0
        for i in range(m):
            if err[i, label_train[i]] / np.sum(err[i, :]) > 0:
                sum_loss -= np.log(err[i, label_train[i]] / np.sum(err[i, :]))

        return sum_loss / m

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class Logistics:
    def __init__(self):
        pass
        #0:Shuffle,1:Batch,2:mini-batch

    def logistics_batch_type(self,train_data, label_train,test_data,label_test,dev_data,label_dev):
        log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        log_reg.fit(train_data,label_train)
        train_pre = log_reg.predict(train_data)
        test_pre = log_reg.predict(test_data)
        dev_pre = log_reg.predict(dev_data)
        test_acc = self.acc(test_pre,label_test)
        dev_acc = self.acc(dev_pre, label_dev)

        return dev_acc,test_acc

    def loss(self,err, label_train):
        m = np.shape(err)[0]
        sum_loss = 0.0
        for i in range(m):
            if err[i, label_train[i]] / np.sum(err[i, :]) > 0:
                sum_loss -= np.log(err[i, label_train[i]] / np.sum(err[i, :]))

        return sum_loss / m

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

def softmax_trainning(model,train_data, label_train, labels, epoch, alpha,test_data,label_test,dev_data,label_dev):
    m, n = np.shape(train_data)
    weights = np.mat(np.random.normal(0.0, 1.0, (n, labels)))
    i = 0
    train_acc,dev_acc,test_acc,max_train_acc,max_dev_acc,max_test_acc = 0.0,0.0,0.0,0.0,0.0,0.0
    if model.batch_flag == 0:
        while i <= epoch*m:
            train_data_s,label_train_s = train_data[i % m], label_train[i % m]
            weigths_max = weights.max(axis=1)
            weights = weights - weigths_max.max()
            error = np.exp(train_data_s * weights)
            # if alpha>6:
            #     print(error[:3])
            if i % (m*4) == 0 and i!=0:
                # print("shuffle-----iter: " + str(i / m) + ", loss: " + str(model.loss(np.exp(train_data * weights), label_train)), end='')
                # error1 = np.exp(train_data * weights)
                # rowsum1 = -error1.sum(axis=1)
                # rowsum1 = rowsum1.repeat(labels, axis=1)
                # error1 = error1 / rowsum1
                # train_acc = model.acc(error1 * -1, label_train)[0, 0]
                # print(",Accuracy:", train_acc, end='')

                #dev
                # print("dev: ", end='')
                error1 = np.exp(dev_data * weights)
                rowsum1 = -error1.sum(axis=1)
                rowsum1 = rowsum1.repeat(labels, axis=1)
                error1 = error1 / rowsum1
                dev_acc = model.acc(error1 * -1, label_dev)[0, 0]
                # print(",Accuracy:", dev_acc, end='')
                if dev_acc>max_dev_acc:
                    max_train_acc = train_acc
                    max_dev_acc = dev_acc
                    # test
                    # print("test: ", end='')
                    error1 = np.exp(test_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    test_acc = model.acc(error1 * -1, label_test)[0, 0]
                    # print(",Accuracy:", test_acc, end='')

            rowsum = -error.sum(axis=1)
            rowsum = rowsum.repeat(labels, axis=1)
            error = error / rowsum

            error[0, label_train_s] += 1
            weights = weights + alpha * train_data_s.T * error  # 梯度更新
            i+=1
    elif model.batch_flag == 1:
        while i <= epoch:
            error = np.exp(train_data * weights)
            # if i % (epoch) == 0 and i!=0:
                # print("batch-----iter: " + str(i) + ", loss: " + str(model.loss(error, label_train)), end='')
            rowsum = -error.sum(axis=1)
            rowsum = rowsum.repeat(labels, axis=1)
            error = error / rowsum
            if i % (4) == 0 and i!=0:
                # train_acc = model.acc(error * -1, label_train)[0, 0]
                # print(",Accuracy:", train_acc)
                # dev
                # print("dev: ", end='')
                error1 = np.exp(dev_data * weights)
                rowsum1 = -error1.sum(axis=1)
                rowsum1 = rowsum1.repeat(labels, axis=1)
                error1 = error1 / rowsum1
                dev_acc = model.acc(error1 * -1, label_dev)[0, 0]
                # print(",Accuracy:", dev_acc)
                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                    # test
                    # print("test: ", end='')
                    error1 = np.exp(test_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    test_acc = model.acc(error1 * -1, label_test)[0, 0]
                    # print(",Accuracy:", test_acc)

            for x in range(m):
                error[x, label_train[x]] += 1
            weights = weights + (alpha / m)* train_data.T * error  # 梯度更新
            i+=1
    else:
        increment = np.zeros((n,labels))
        while i <= epoch*m:
            train_data_s, label_train_s = train_data[i % m], label_train[i % m]
            error = np.exp(train_data_s * weights)
            if i % (4*m) == 0 and i!=0:
                #train
                # print("mini_batch-----iter: " + str(i / (m)) + ", loss: " + str(
                #     model.loss(np.exp(train_data * weights), label_train)), end='')
                error1 = np.exp(train_data * weights)
                rowsum1 = -error1.sum(axis=1)
                rowsum1 = rowsum1.repeat(labels, axis=1)
                error1 = error1 / rowsum1
                train_acc = model.acc(error1 * -1, label_train)[0, 0]
                # print(",Accuracy:", train_acc)
                # dev
                # print("dev: ", end='')
                error1 = np.exp(dev_data * weights)
                rowsum1 = -error1.sum(axis=1)
                rowsum1 = rowsum1.repeat(labels, axis=1)
                error1 = error1 / rowsum1
                dev_acc = model.acc(error1 * -1, label_dev)[0, 0]
                # print(",Accuracy:", dev_acc)
                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                    #test
                    # print("test: " , end='')
                    error1 = np.exp(test_data * weights)
                    rowsum1 = -error1.sum(axis=1)
                    rowsum1 = rowsum1.repeat(labels, axis=1)
                    error1 = error1 / rowsum1
                    test_acc = model.acc(error1 * -1, label_test)[0, 0]
                    # print(",Accuracy:", test_acc)
            rowsum = -error.sum(axis=1)
            rowsum = rowsum.repeat(labels, axis=1)
            error = error / rowsum

            error[0, label_train_s] += 1
            increment += train_data_s.T * error

            if i%100 == 0:
                weights = weights + (alpha/100) * increment  # 梯度更新
                increment = 0
            i += 1

    return max_dev_acc,test_acc

if __name__ == '__main__':
    data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_train.tsv')
    data_test = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_test.tsv')
    data_dev = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_dev.tsv')
    x_train,y_train,x_dev,y_dev,x_test,y_test = [],[],[],[],[],[]
    for i in data_train:
        x_train.append(i[2].lower())
        y_train.append(int(i[3]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    for i in data_test:
        x_test.append(i[2].lower())
        y_test.append(int(i[3]))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    for i in data_dev:
        x_dev.append(i[2].lower())
        y_dev.append(int(i[3]))
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    print("data ready")
    X = np.concatenate((x_train,x_dev,x_test),axis=0)
    #n-gram
    vect = CountVectorizer(ngram_range=(1,2))
    #bag of words
    # vect = CountVectorizer()
    vect.fit_transform(X)
    # veccc = vect.vocabulary_
    # print(len(veccc))
    x_train = vect.transform(x_train)
    x_test = vect.transform(x_test)
    x_dev = vect.transform(x_dev)
    print("transform")

    type_gra = {0:"Shuffle",1:"Batch",2:"mini-batch"}
    alphas = [0.03, 0.1,0.6, 1.0,2.0,3.0, 5.0, 10.0]
    # #type
    max_dev,max_test = 0.0,0.0
    for j in range(6):
        for i in range(3):
            model = Softmax(i)
            dev_s, test_s = softmax_trainning(model,x_train, y_train, 5, 300, alphas[j], x_test, y_test,x_dev,y_dev)
            print(type_gra[i],'alphas:',alphas[j],"dev:",dev_s,"test:",test_s)
            if max_dev<dev_s:
                max_dev = dev_s
                max_test = test_s

    #logistic/softmax regression比较
    # model = Softmax(2)
    # train_s, test_s = softmax_trainning(model,x_train, y_train, 5, 400, 1, x_test, y_test,x_dev,y_dev)
    logist = Logistics()
    dev_l, test_l = logist.logistics_batch_type(x_train,y_train,x_test,y_test,x_dev,y_dev)

    print('\t\t',"softmax\t\t","logistics")
    print('验证集\t',max_dev,"\t",dev_l)
    print('测试集\t',max_test,"\t",test_l)

    #softmax后，特征选择、学习率比较
    # alphas = [0.03,0.5,1.0,5.0,10.0,100.0]
    # #type
    # list_train_s, list_test_s = [[], [], []], [[], [], []]
    # for j in range(6):
    #     for i in range(3):
    #         soft = Softmax(i)
    #         train_s,test_s = soft.gradient_batch_type(X_train, y_train, 5, 20, alphas[j],X_test,y_test)
    #         list_train_s[i].append(train_s),
    #         list_test_s[i].append(test_s)
    # matplotlib.pyplot.subplot(2, 2, 1)
    # matplotlib.pyplot.semilogx(alphas, list_train_s[0], 'r--', label='shuffle')
    # matplotlib.pyplot.semilogx(alphas, list_train_s[1], 'g--', label='batch')
    # matplotlib.pyplot.semilogx(alphas, list_train_s[2], 'b--', label='mini-batch')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Bag of words -- Training Set")
    # matplotlib.pyplot.xlabel("Learning Rate")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.2, 1.0)
    # matplotlib.pyplot.subplot(2, 2, 2)
    # matplotlib.pyplot.semilogx(alphas, list_test_s[0], 'r--', label='shuffle')
    # matplotlib.pyplot.semilogx(alphas, list_test_s[1], 'g--', label='batch')
    # matplotlib.pyplot.semilogx(alphas, list_test_s[2], 'b--', label='mini-batch')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Bag of words -- Test Set")
    # matplotlib.pyplot.xlabel("Learning Rate")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.2, 1.0)
    # matplotlib.pyplot.tight_layout()
    #
    # X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
    # vect1 = CountVectorizer(ngram_range=(1,2))
    # vect1.fit_transform(X)
    # # veccc = vect1.vocabulary_
    # # print(len(veccc))
    # X_train_n = vect1.transform(X_train_n)
    # X_test_n = vect1.transform(X_test_n)
    #
    # list_train_s, list_test_s = [[], [], []], [[], [], []]
    # for j in range(6):
    #     for i in range(3):
    #         soft = Softmax(i)
    #         train_s, test_s = soft.gradient_batch_type(X_train_n, y_train_n, 5, 20, alphas[j], X_test_n, y_test_n)
    #         list_train_s[i].append(train_s)
    #         list_test_s[i].append(test_s)
    # matplotlib.pyplot.subplot(2, 2, 3)
    # matplotlib.pyplot.semilogx(alphas, list_train_s[0], 'r--', label='shuffle')
    # matplotlib.pyplot.semilogx(alphas, list_train_s[1], 'g--', label='batch')
    # matplotlib.pyplot.semilogx(alphas, list_train_s[2], 'b--', label='mini-batch')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("N gram -- Training Set")
    # matplotlib.pyplot.xlabel("Learning Rate")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.2, 1.0)
    # matplotlib.pyplot.subplot(2, 2, 4)
    # matplotlib.pyplot.semilogx(alphas, list_test_s[0], 'r--', label='shuffle')
    # matplotlib.pyplot.semilogx(alphas, list_test_s[1], 'g--', label='batch')
    # matplotlib.pyplot.semilogx(alphas, list_test_s[2], 'b--', label='mini-batch')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("N gram -- Test Set")
    # matplotlib.pyplot.xlabel("Learning Rate")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.2, 1.0)
    # matplotlib.pyplot.tight_layout()
    # matplotlib.pyplot.show()
    #
