import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sentiment_analysis_task1 import read_tsv

def context_data_generation(data,vocab,vect_len):
    # sentence = [[vocab[j] for j in i] for i in data]
    sentence = []
    for i in data:
        sentence_d = []
        for j in i:
            if j in vocab.keys():
                sentence_d.append(vocab[j])
            else:
                sentence_d.append(vect_len)
        sentence.append(sentence_d)

    sentence = np.array(sentence)

    X_data,y_data = [],[]
    for i in sentence:
        for j,_ in enumerate(i):
            X_data_d = []
            y_data.append(i[j])
            if j == 0:
                X_data_d.append(vect_len+1)
                X_data_d.append(vect_len + 1)
            elif j == 1:
                X_data_d.append(vect_len + 1)
                X_data_d.append(i[0])
            else:
                X_data_d.append(i[j-2])
                X_data_d.append(i[j-1])
            if j == len(i)-1:
                X_data_d.append(vect_len + 1)
                X_data_d.append(vect_len + 1)
            elif j == len(i)-2:
                X_data_d.append(i[len(i)-1])
                X_data_d.append(vect_len + 1)
            else:
                X_data_d.append(i[j + 1])
                X_data_d.append(i[j + 2])
            X_data.append(X_data_d)
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data,y_data

class Word2vec(nn.Module):
    def __init__(self,vocab):
        super(Word2vec, self).__init__()
        self.vocab_embedding = nn.Embedding(vocab+10,64)
        self.out = nn.Linear(64*4,vocab+2)
        self.cross = nn.CrossEntropyLoss()

    def forward(self,context):
        context_emb = self.vocab_embedding(context)
        context_emb = context_emb.view(-1,64*4)
        out = self.out(context_emb)
        return out

    def loss(self,pre,label):
        loss = self.cross(pre,label)
        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        # print(y_pre)
        # print(label)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

if __name__=='__main__':
    data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/train.tsv')[:70000]
    # data_test = read_tsv('./data/sentiment-analysis-on-movie-reviews/test.tsv')[1:]
    X,X_split = [],[]
    sentence_id = 0
    for i in data_train[1:]:
        if sentence_id!=int(i[1]):
            sentence_id += 1
            X.append(i[2].lower())
            X_split.append(i[2].lower().split(' '))

    X = np.array(X)
    X_split = np.array(X_split)
    # vect = CountVectorizer()
    # vect.fit_transform(X)
    # vect_len = len(vect.vocabulary_)
    # print(X_split[:1])

    # print(len(vect.vocabulary_))

    veccc = {}
    vect_len = 0
    for i in X_split:
        for j, k in enumerate(i):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1

    X_data,y_data = context_data_generation(X_split,veccc,vect_len)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=123, shuffle=True)
    X_train = torch.from_numpy(X_train).long()
    X_test = torch.from_numpy(X_test).long()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_iter = DataLoader(train_data, shuffle=False, batch_size=128)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=128)
    model = Word2vec(vect_len)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    model.train()
    # print(vect.vocabulary_)
    #
    # print(X_data[:2])
    # print(y_data[:2])

    max_train,max_test = 0.0,0.0
    for epoch in range(100):
        for batch in train_iter:
            X_train_b, y_train_b = batch
            X_train_b, y_train_b = X_train_b.cuda(), y_train_b.cuda()

            out = model(X_train_b)
            loss = model.loss(out,y_train_b)
            loss.backward()
            optimizer.step()
            model.zero_grad()

        # print(loss)
        if epoch% 2 == 0:
            model.eval()
            flag = 0
            total = None
            for batch in test_iter:
                X_train_b, y_train_b = batch
                X_train_b, y_train_b = X_train_b.cuda(), y_train_b.cuda()

                out = model(X_train_b)
                # print(out[:2].argmax(axis=1))
                # print(y_test[:2])
                if flag == 0:
                    total = out
                    flag = 1
                else:
                    total = torch.cat((total, out), 0)
            # print(total.size())
            test_acc = model.acc(total, y_test)
            flag = 0
            total = None
            for batch in train_iter:
                X_train_b, y_train_b = batch
                X_train_b, y_train_b = X_train_b.cuda(), y_train_b.cuda()

                out = model(X_train_b)

                if flag == 0:
                    total = out
                    flag = 1
                else:
                    total = torch.cat((total, out), 0)
            # print(total.size())
            train_acc = model.acc(total, y_train)
            print("epoch",epoch, "----train_acc:", train_acc,"----test_epoch:",test_acc)
            if max_test<test_acc:
                max_test = test_acc
                max_train = train_acc
            model.train()

    print("\n\n","训练文本数量：",X_split.shape[0],"词语数量：",vect_len)
    print("MAX---train_acc:",max_train,"---test_acc:",max_test)
