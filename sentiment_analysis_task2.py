import csv
import os
import sys

import matplotlib.pyplot
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sentiment_analysis_task1 import Softmax

# sys.setrecursionlimit(1000000)

def read_tsv(filename):
    with open(filename,encoding="utf-8") as f:
        tsvreader = csv.reader(f, delimiter='\t')
        temp = list(tsvreader)
        return temp

#glove读取

def get_numpy_word_embed(word2ix):
    row = 0
    file = 'glove.6B.50d.txt'
    path = 'D:\\transformerFileDownload\\glove'
    whole = os.path.join(path, file)
    words_embed = {}
    with open(whole, mode='r',encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            # print(len(line.split()))
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            # if row > 20000:
            #     break
            row += 1
    # word2ix = {}
    ix2word = {ix: w for w, ix in word2ix.items()}

    id2emb = {}
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 50
    data = [id2emb[ix] for ix in range(len(word2ix))]
    # print(data)

    return data

def vocab_mask_generation(data):
    m,n = data.shape
    vocab = np.full((m,n),0)
    mask = np.full((m,n),0)
    for i in range(m):
        for j in range(n):
            vocab[i, j] = j
            if data[i,j]>0:
                mask[i,j] = 1
    return vocab,mask

def id_generation(data,vocab):
    sentence = []
    for i in data:
        sentence_d = []
        for j in i:
            if j in vocab.keys():
                sentence_d.append(vocab[j])
            else:
                # 如'a'这样的词，是没有在字典里的
                sentence_d.append(vect_len)
        sentence.append(sentence_d)
    return sentence

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class Softmax_embedding(nn.Module):
    def __init__(self,vocab,label_nums):
        super(Softmax_embedding,self).__init__()
        self.vocab_embedding = nn.Embedding(vocab+100,64)
        self.fre_embedding = nn.Embedding(vocab+100,16)
        self.vocab_l = nn.Linear(64,1)
        self.fre_l = nn.Linear(16,1)
        self.l_out = nn.Linear(vocab,label_nums)
        self.crossLoss = nn.CrossEntropyLoss()
        #0:Shuffle,1:Batch,2:mini-batch

    def forward(self,vocab,fre,mask):
        m, n = vocab.size()
        # (batch,sentence_len,embedding)
        vocab_embedding = self.vocab_embedding(vocab)
        fre_embedding = self.fre_embedding(fre)
        #(batch,sentence_len,1)
        vocab_l = self.vocab_l(vocab_embedding)
        fre_l = self.fre_l(fre_embedding)
        # (batch,sentence_len)
        vocab_l = vocab_l.view(m,n)
        fre_l = fre_l.view(m,n)
        # (batch,sentence_len) 词汇embedding与词频点乘,并mask掉句中没有出现的词汇
        encode_l = vocab_l.mul(fre_l)
        encode_l = encode_l.mul(mask)
        out = self.l_out(encode_l)
        return out

    def loss(self,pre, label_data):
        loss = self.crossLoss(pre,label_data)

        return loss.mean()

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        # print(y_pre.size())
        # print(label.size())
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class CNN_e(nn.Module):
    def __init__(self,vect_len,out_channels,flag_g,weight,label_num,sentence_len):
        super(CNN_e, self).__init__()
        if flag_g == 0:
            weight_r = nn.init.kaiming_normal_(torch.Tensor(vect_len+2,50))
            self.embedding = nn.Embedding(vect_len + 2, 50,_weight=weight_r)
        else:
            self.embedding = nn.Embedding(vect_len+2,50,_weight=weight)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=50,
                                              out_channels=out_channels,
                                              kernel_size=filter_size)
                                    for filter_size in [3,4,5]])


        # self.cnn_2 = nn.Sequential(nn.Conv2d(1,out_channels,(2,50),padding=(1,0)),nn.ReLU())
        # self.cnn_3 = nn.Sequential(nn.Conv2d(1,out_channels,(3,50),padding=(1,0)),nn.ReLU())
        # self.cnn_4 = nn.Sequential(nn.Conv2d(1, out_channels, (4, 50), padding=(2, 0)), nn.ReLU())
        # self.cnn_5 = nn.Sequential(nn.Conv2d(1, out_channels, (5, 50), padding=(3, 0)), nn.ReLU())
        # self.pad = torch.zeros((1,out_channels,1),requires_grad=False).to(torch.device('cuda:0'))-100
        # self.maxpool = nn.MaxPool1d(sentence_len+1)

        self.l = nn.Sequential(nn.Linear(out_channels*3,label_num),nn.Dropout(0.5))
        self.cross = nn.CrossEntropyLoss()

    def forward(self,data):
        #(batch,sentence_len,embedding)
        embed = self.embedding(data)
        #(bacth,1,sentence_len,embedding)
        # embed = embed.unsqueeze(1)
        embedded = embed.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved[n] = [batch size, n filters, seq len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]

        # pooled[n] = [batch size, n filters]

        out = torch.cat(pooled, dim=-1)

        #(batch,channels,sentence_len+2-2+1,1)
        # cnn_2 = self.cnn_2(embed).squeeze(3)
        # # (batch,channels,sentence_len+2-3+1,1)
        # cnn_3 = self.cnn_3(embed).squeeze(3)
        # # (batch,channels,sentence_len+4-4+1,1)
        # cnn_4 = self.cnn_4(embed).squeeze(3)
        # # (batch,channels,sentence_len+4-5+1,1)
        # cnn_5 = self.cnn_5(embed).squeeze(3)

        #统一最后一个维度
        # m,n,_ = cnn_2.size()
        # pad = self.pad.expand(m,n,1)
        # cnn_3 = torch.cat((cnn_3,pad),2)
        # cnn_5 = torch.cat((cnn_5,pad),2)
        #
        # #(batch,channels,1)
        # maxpool_2 = self.maxpool(cnn_2).squeeze(2)
        # maxpool_3 = self.maxpool(cnn_3).squeeze(2)
        # maxpool_4 = self.maxpool(cnn_4).squeeze(2)
        # maxpool_5 = self.maxpool(cnn_5).squeeze(2)

        #(batch,4*channels)
        # out = torch.cat((maxpool_2,maxpool_3,maxpool_4,maxpool_5),1)
        #(batch,label)
        out = self.l(out)

        return out

    def loss(self, pre, label_data):
        loss = self.cross(pre, label_data)

        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn
        # for i in range(lenn):
        #     if y_pre[i] != label[i]:
        #         print("pre:",y_pre[i ],"  true:",label[i])

        return y_acc

class RNN_e(nn.Module):
    def __init__(self,vect_len,flag_g,weight,label_num):
        super(RNN_e, self).__init__()
        if flag_g == 0:
            weight_r = nn.init.xavier_normal_(torch.Tensor(vect_len + 2, 50))
            self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight_r)
        else:
            self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)
        self.rnn = nn.RNN(input_size=50,hidden_size=64,num_layers=1,dropout=0.5,batch_first=True)
        self.l = nn.Linear(64,label_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,data):
        embed = self.embedding(data)
        # embedded = embed.permute(1,0,2)
        h0 = torch.zeros((1,embed.size()[0],64)).cuda()
        _,rnn = self.rnn(embed,h0)
        rnn = rnn.squeeze(0)
        out = self.l(rnn)
        return out

    def loss(self, pre, label_data):
        loss = self.loss(pre, label_data)

        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn
        # for i in range(50):
        #     if y_pre[i] != label[i]:
        #         print("pre:",y_pre[i ],"  true:",label[i])

        return y_acc

def softmax_embedding_train(X_train_s,y_train,X_test_s,y_test,vect_len,epochs):
    train_vocab, train_mask = vocab_mask_generation(X_train_s)
    test_vocab, test_mask = vocab_mask_generation(X_test_s)
    #X_train_s为压缩系数矩阵（np.ndarray）， .A 转为array正常的
    X_train_s = torch.from_numpy(X_train_s.A).long()
    X_test_s = torch.from_numpy(X_test_s.A).long()
    train_vocab = torch.from_numpy(train_vocab).long()
    train_mask = torch.from_numpy(train_mask).long()
    test_vocab = torch.from_numpy(test_vocab).long()
    test_mask = torch.from_numpy(test_mask).long()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    train_data = TensorDataset(X_train_s,train_vocab,train_mask,y_train)
    test_data = TensorDataset(X_test_s,test_vocab,test_mask,y_test)

    train_iter = DataLoader(train_data, shuffle=False, batch_size=128)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=128)

    model = Softmax_embedding(vect_len, 5)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()

    max_train,max_test = 0.0,0.0
    for epoch in range(epochs):
        for batch in train_iter:
            train_fre, train_vocab, train_mask, y_train_t = batch
            train_fre, train_vocab, train_mask, y_train_t = train_fre.cuda(), train_vocab.cuda(), train_mask.cuda(), y_train_t.cuda()

            out = model(train_vocab, train_fre, train_mask)
            loss = model.loss(out,y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        # if epoch%() == 0:
        model.eval()
        flag = 0
        total = None
        for batch in test_iter:
            test_fre, test_vocab, test_mask, _ = batch
            test_fre, test_vocab, test_mask = test_fre.cuda(), test_vocab.cuda(), test_mask.cuda()
            out = model(test_vocab, test_fre, test_mask)
            if flag==0:
                total = out
                flag = 1
            else:
                total = torch.cat((total,out),0)
        # print(total.size())
        test_acc = model.acc(total,y_test)
        flag = 0
        total = None
        for batch in train_iter:
            train_fre, train_vocab, train_mask, _ = batch
            train_fre, train_vocab, train_mask = train_fre.cuda(), train_vocab.cuda(), train_mask.cuda()
            out = model(train_vocab, train_fre, train_mask)
            if flag == 0:
                total = out
                flag = 1
            else:
                total = torch.cat((total, out), 0)
        # print(total.size())
        train_acc = model.acc(total, y_train)
        print("epoch:", epoch, "----train_acc:", train_acc,"----test_acc:", test_acc)
        if test_acc>max_test:
            max_test = test_acc.cpu().numpy().tolist()
            max_train = train_acc.cpu().numpy().tolist()
        # model.train()
    return max_train,max_test

def cnn_embedding_train(X_train_c,y_train,X_test_c,y_test,vect_len,epochs,flag_g,weigth):
    m,n = X_train_c.shape
    train_data = TensorDataset(X_train_c, y_train)
    test_data = TensorDataset(X_test_c, y_test)

    batch = 256
    train_iter = DataLoader(train_data, shuffle=False, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=batch)

    model = CNN_e(vect_len,2,flag_g,weigth,5,n)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    loss_s = FocalLoss()

    for epoch in range(epochs):
        for batch in train_iter:
            train_id, y_train_t = batch
            train_id, y_train_t = train_id.cuda(), y_train_t.cuda()

            out = model(train_id)
            loss = loss_s(out,y_train_t)
            # loss = model.loss(out, y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            # print(model.acc(out, y_train_t))
        if epoch%5 == 0:
            model.eval()
            flag = 0
            total_train,total_label = None,None
            for batch in test_iter:
                train_id, y_train_t = batch
                train_id, y_train_t = train_id.cuda(), y_train_t.cuda()
                out = model(train_id)
                # loss = model.loss(out, y_train_t)
                if flag == 0:
                    total_train = out
                    total_label = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label = torch.cat((total_label,y_train_t),0)
            # print(total.size())
            # print(loss)
            test_acc = model.acc(total_train, total_label)
            flag = 0
            total_train, total_label = None, None
            for batch in test_iter:
                train_id, y_train_t = batch
                train_id, y_train_t = train_id.cuda(), y_train_t.cuda()
                out = model(train_id)
                # loss = model.loss(out, y_train_t)
                if flag == 0:
                    total_train = out
                    total_label = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label = torch.cat((total_label, y_train_t), 0)
            # print(total.size())
            # print(loss)
            train_acc = model.acc(total_train, total_label)
            print("epoch:", epoch, "----train_acc:", train_acc,"----test_acc:", test_acc)
            model.train()

def rnn_embedding_train(X_train_c,y_train,X_test_c,y_test,vect_len,epochs,flag_g,weigth):
    m,n = X_train_c.shape
    train_data = TensorDataset(X_train_c, y_train)
    test_data = TensorDataset(X_test_c, y_test)

    batch = 256
    train_iter = DataLoader(train_data, shuffle=False, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=batch)

    model = RNN_e(vect_len,flag_g,weigth,5)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.train()
    loss_s = FocalLoss()

    for epoch in range(epochs):
        for batch in train_iter:
            train_id, y_train_t = batch
            train_id, y_train_t = train_id.cuda(), y_train_t.cuda()

            out = model(train_id)
            loss = loss_s(out,y_train_t)
            # loss = model.loss(out, y_train_t)
            # loss = F.cross_entropy(out,y_train_t)
            loss.backward()
            # if epoch == 70:
            #     for name, parms in model.named_parameters():
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad.data.mean(),"  ",parms.grad.data.std())
            optimizer.step()
            model.zero_grad()
            # print(loss)
            # print(model.acc(out, y_train_t))
        if epoch%1 == 0:
            model.eval()
            flag = 0
            total_train,total_label = None,None
            for batch in test_iter:
                train_id, y_train_t = batch
                train_id, y_train_t = train_id.cuda(), y_train_t.cuda()
                out = model(train_id)
                # loss = model.loss(out, y_train_t)
                if flag == 0:
                    total_train = out
                    total_label = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label = torch.cat((total_label,y_train_t),0)
            # print(total.size())
            # print(loss)
            test_acc = model.acc(total_train, total_label)
            flag = 0
            total_train, total_label = None, None
            for batch in test_iter:
                train_id, y_train_t = batch
                train_id, y_train_t = train_id.cuda(), y_train_t.cuda()
                out = model(train_id)
                # loss = model.loss(out, y_train_t)
                if flag == 0:
                    total_train = out
                    total_label = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label = torch.cat((total_label, y_train_t), 0)
            # print(total.size())
            # print(loss)
            train_acc = model.acc(total_train, total_label)
            print("epoch:", epoch, "----train_acc:", train_acc,"----test_acc:", test_acc)
            model.train()

if __name__ == '__main__':
    # 全部
    data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/train.tsv')[1:3000]
    X,X_split,y = [],[],[]
    for i in data_train:
        # if sentence_id != int(i[1]):
        #     sentence_id += 1
        X.append(i[2].lower())
        y.append(int(i[3]))
        X_split.append(i[2].lower().split(' '))

    # 完整的句子
    # data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/train.tsv')[1:40000]
    # sentence_id = 0
    # X, X_split, y = [], [], []
    # for i in data_train:
    #     if sentence_id != int(i[1]):
    #         sentence_id += 1
    #         X.append(i[2].lower())
    #         y.append(int(i[3]))
    #         X_split.append(i[2].lower().split(' '))
    # print(sentence_id)
    veccc = {}
    vect_len = 0
    for i in X_split:
        for j,k in enumerate(i):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    # print(vect_len)

    X = np.array(X)
    y = np.array(y)

    # # softmax的embedding与原实验比较
    # vect = CountVectorizer()
    # vect.fit_transform(X)
    # veccc = vect.vocabulary_
    # print(veccc)
    # print(len(veccc))
    #
    # vect_len = len(veccc)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True)
    # X_train_s = vect.transform(X_train)
    # X_test_s = vect.transform(X_test)
    #
    # train_e,test_e = softmax_embedding_train(X_train_s,y_train,X_test_s,y_test,vect_len,50)
    # soft = Softmax(2)
    # train_s, test_s = soft.gradient_batch_type(X_train_s, y_train, 5, 300, 1, X_test_s, y_test)
    # print('\t\t', "softmax\t\t", "embedding")
    # print('训练集\t', train_s, "\t", train_e)
    # print('测试集\t', test_s, "\t", test_e )

    # CNN、RNN 随机与glove比较
    veccc['[UNK]'] = vect_len
    veccc['[PAD]'] = vect_len+1
    glove_embed = get_numpy_word_embed(veccc)
    glove_embed = torch.FloatTensor(glove_embed)

    sentence_id = id_generation(X_split,veccc)
    sentence_id = [torch.LongTensor(i) for i in sentence_id]
    sentence_id = pad_sequence(sentence_id,batch_first=True,padding_value=vect_len+1)
    y = torch.from_numpy(y).long()

    X_train,y_train,X_test,y_test = sentence_id[:1200],y[:1200],sentence_id[1200:1500],y[1200:1500]

    # cnn_embedding_train(X_train,y_train,X_test,y_test,vect_len,100,0,glove_embed)

    rnn_embedding_train(X_train, y_train, X_test, y_test, vect_len, 100, 1, glove_embed)












