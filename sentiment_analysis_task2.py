import csv
import os
import sys
from transformers import BertTokenizer, BertModel,BertConfig
import matplotlib.pyplot
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
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

def get_numpy_word_embed(word2ix,lenn=1e6):
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
    cls_sep = np.random.normal(scale=0.5,size=100).tolist()
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 50
    id2emb[lenn+2] = cls_sep[:50]
    id2emb[lenn+3] = cls_sep[50:]
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

def rnn_mask_g(data):
    rnn_mask = torch.zeros((len(data),1,64)).long()
    for i,j in enumerate(data):
        rnn_mask[i][0] += j-1
    return rnn_mask

def cnn_mask_g(data,max):
    cnn_mask = torch.zeros((len(data),3)).long()
    add = [0,1,1]
    max_add = [0,1,0]
    for i in range(3):
        for j,k in enumerate(data):
            if k+add[i]+1<max+max_add[i]:
                cnn_mask[j][i] += k+add[i]+1
            else:
                cnn_mask[j][i] += max
    return cnn_mask

def id_generation(data,vocab,vect_len):
    sentence = []
    for i in data:
        sentence_d = []
        for j in i:
            if j in vocab.keys():
                sentence_d.append(vocab[j])
            else:
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
    def __init__(self,vect_len,out_channels,flag_g,weight,label_num,bias_f=True):
        super(CNN_e, self).__init__()
        if flag_g == 0:
            weight_r = nn.init.kaiming_normal_(torch.Tensor(vect_len+2,50))
            self.embedding = nn.Embedding(vect_len + 2, 50,_weight=weight_r)
        else:
            self.embedding = nn.Embedding(vect_len+2,50,_weight=weight)
        self.drop = nn.Dropout(0.5)
        # if bias_f==True:
        #     self.conves = nn.ModuleList([
        #         nn.Sequential(nn.Conv2d(1, out_channels, (3, 50), padding=(1, 0)), nn.ReLU()),
        #         nn.Sequential(nn.Conv2d(1, out_channels, (4, 50), padding=(2, 0)), nn.ReLU()),
        #         nn.Sequential(nn.Conv2d(1, out_channels, (5, 50), padding=(2, 0)), nn.ReLU())]
        #     )
        # else:
        #     self.conves = nn.ModuleList([
        #         nn.Sequential(nn.Conv2d(1, out_channels, (3, 50), padding=(1, 0),bias=False), nn.ReLU()),
        #         nn.Sequential(nn.Conv2d(1, out_channels, (4, 50), padding=(2, 0),bias=False), nn.ReLU()),
        #         nn.Sequential(nn.Conv2d(1, out_channels, (5, 50), padding=(2, 0),bias=False), nn.ReLU())]
        #     )
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=50,
                                              out_channels=out_channels,
                                              kernel_size=filter_size)
                                    for filter_size in [3, 4, 5]])

        # self.cnn_3 = nn.Sequential(nn.Conv2d(1,out_channels,(3,50),padding=(1,0)),nn.ReLU())
        # self.cnn_4 = nn.Sequential(nn.Conv2d(1, out_channels, (4, 50), padding=(2, 0)), nn.ReLU())
        # self.cnn_5 = nn.Sequential(nn.Conv2d(1, out_channels, (5, 50), padding=(3, 0)), nn.ReLU())
        # self.pad = torch.zeros((1,out_channels,1),requires_grad=False).to(torch.device('cuda:0'))-100
        # self.maxpool = nn.MaxPool1d(sentence_len+1)

        self.l = nn.Sequential(nn.Linear(out_channels*3,label_num),nn.Dropout(0.5))
        self.cross = nn.CrossEntropyLoss()

    def forward(self,data,mask):
        #(batch,sentence_len,embedding)
        embed = self.embedding(data)
        #(bacth,1,sentence_len,embedding)
        # embed = embed.unsqueeze(1)
        #
        # conved = [conv(embed).squeeze(-1) for conv in self.conves]

        #mask,效果不好,且耗时长，最后的线性层很容易使得2概率大
        # conved_mask = [[[[conved[j][i][0][m] for m in range(mask[i][j])],[conved[j][i][1][m] for m in range(mask[i][j])]] for i in range(embed.size()[0])] for j in range(3)]
        #
        # pooled = [[[[F.max_pool1d(torch.tensor(conved_mask[j][i][0]).unsqueeze(0).unsqueeze(0).cuda(),len(conved_mask[j][i][0])).squeeze(0).squeeze(0)],[F.max_pool1d(torch.tensor(conved_mask[j][i][1]).unsqueeze(0).unsqueeze(0).cuda(),len(conved_mask[j][i][1])).squeeze(0).squeeze(0)]] for i in range(embed.size()[0])] for j in range(3)]
        #
        # pooled = [torch.tensor(pooled[0]),torch.tensor(pooled[1]),torch.tensor(pooled[2])]
        # out = torch.cat(pooled, dim=-1).cuda()
        # out = out.view(-1, 6)
        #

        # pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]
        embed = self.drop(embed)
        embedded = embed.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved[n] = [batch size, n filters, seq len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]

        # pooled[n] = [batch size, n filters]

        out = torch.cat(pooled, dim=-1)
        out_l = self.l(out)
        return out_l

    def loss(self, pre, label_data):
        loss = self.cross(pre, label_data)

        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        return y_acc

class RNN_e(nn.Module):
    def __init__(self,vect_len,flag_g,weight,label_num):
        super(RNN_e, self).__init__()
        if flag_g == 0:
            weight_r = nn.init.xavier_normal_(torch.Tensor(vect_len + 2, 50))
            self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight_r)
        else:
            self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)
        self.drop = nn.Dropout(0.5)
        self.rnn = nn.RNN(input_size=50,hidden_size=64,num_layers=1,dropout=0.5,batch_first=True,nonlinearity='tanh')
        #LSTM
        # self.rnn = nn.LSTM(input_size=50,hidden_size=64,num_layers=1,bias=True,batch_first=True,dropout=0.5,bidirectional=False)
        self.l = nn.Linear(64,label_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,data,mask):

        embed = self.embedding(data)
        embed = self.drop(embed)
        h0 = torch.zeros((1,embed.size()[0],64),requires_grad=False).cuda()
        rec,rnn = self.rnn(embed,h0)
        gather_em = rec.gather(1,mask)
        out = self.l(gather_em).squeeze(1)
        return out

    def loss(self, pre, label_data):
        loss = self.loss(pre, label_data)

        return loss

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn
        # for i in range(100):
        #     if y_pre[i + 50] != label[i + 50]:
        #         print("pre:", y_pre[i + 50], "  true:", label[i + 50])
        return y_acc

def softmax_embedding_train(X_train_s,y_train,X_test_s,y_test,X_dev_s,y_dev,vect_len,epochs):
    train_vocab, train_mask = vocab_mask_generation(X_train_s)
    test_vocab, test_mask = vocab_mask_generation(X_test_s)
    dev_vocab, dev_mask = vocab_mask_generation(X_dev_s)
    #X_train_s为压缩系数矩阵（np.ndarray）， .A 转为array正常的
    X_train_s = torch.from_numpy(X_train_s.A).long()
    X_test_s = torch.from_numpy(X_test_s.A).long()
    X_dev_s = torch.from_numpy(X_dev_s.A).long()
    train_vocab = torch.from_numpy(train_vocab).long()
    train_mask = torch.from_numpy(train_mask).long()
    test_vocab = torch.from_numpy(test_vocab).long()
    test_mask = torch.from_numpy(test_mask).long()
    dev_vocab = torch.from_numpy(dev_vocab).long()
    dev_mask = torch.from_numpy(dev_mask).long()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    y_dev = torch.from_numpy(y_dev).long()

    train_data = TensorDataset(X_train_s,train_vocab,train_mask,y_train)
    test_data = TensorDataset(X_test_s,test_vocab,test_mask,y_test)
    dev_data = TensorDataset(X_dev_s, dev_vocab, dev_mask, y_dev)

    train_iter = DataLoader(train_data, shuffle=False, batch_size=512)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=512)
    dev_iter = DataLoader(dev_data, shuffle=False, batch_size=512)

    model = Softmax_embedding(vect_len, 5)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()

    max_dev,max_test = 0.0,0.0
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
        for batch in dev_iter:
            test_fre, test_vocab, test_mask, _ = batch
            test_fre, test_vocab, test_mask = test_fre.cuda(), test_vocab.cuda(), test_mask.cuda()
            out = model(test_vocab, test_fre, test_mask)
            if flag==0:
                total = out
                flag = 1
            else:
                total = torch.cat((total,out),0)
        # print(total.size())
        dev_acc = model.acc(total,y_dev)
        if max_dev<dev_acc:
            max_dev = dev_acc
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
            max_test = test_acc

    return max_dev,max_test

def cnn_embedding_train(X_train_c,y_train,X_test_c,y_test,X_dev_c,y_dev,mask_train,mask_test,mask_dev,vect_len,epochs,flag_g,weigth,bias_f):
    train_data = TensorDataset(X_train_c, y_train,mask_train)
    dev_data = TensorDataset(X_dev_c,y_dev,mask_dev)
    test_data = TensorDataset(X_test_c, y_test,mask_test)

    batch = 512
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batch)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batch)

    model = CNN_e(vect_len,64,flag_g,weigth,5,bias_f)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.0001)
    model.train()

    weight_loss = None
    # weight_loss = torch.tensor([0.75, 0.75, 0.25, 0.75, 0.75]).cuda()
    loss_s = FocalLoss(weight=weight_loss)
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->parm_value:', parms.data)
    max_dev,max_test = 0.0,0.0
    for epoch in range(epochs):
        for batch in train_iter:
            train_id, y_train_t,mask_train_t = batch
            train_id, y_train_t,mask_train_t = train_id.cuda(), y_train_t.cuda(),mask_train_t.cuda()

            out = model(train_id,mask_train_t)
            loss = loss_s(out,y_train_t)
            # loss = model.loss(out, y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            # print(model.acc(out, y_train_t))
        if epoch%4 == 0:
            model.eval()
            flag = 0
            total_train,total_label = None,None
            for batch in dev_iter:
                train_id, y_train_t, mask_train_t = batch
                train_id, y_train_t, mask_train_t = train_id.cuda(), y_train_t.cuda(), mask_train_t.cuda()
                out = model(train_id, mask_train_t)
                # loss = model.loss(out, y_train_t)
                if flag == 0:
                    total_train = out
                    total_label = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label = torch.cat((total_label,y_train_t),0)
            dev_acc = model.acc(total_train, total_label)
            if max_dev<dev_acc:
                max_dev = dev_acc
                for batch in test_iter:
                    train_id, y_train_t, mask_train_t = batch
                    train_id, y_train_t, mask_train_t = train_id.cuda(), y_train_t.cuda(), mask_train_t.cuda()
                    out = model(train_id, mask_train_t)
                    # loss = model.loss(out, y_train_t)
                    if flag == 0:
                        total_train = out
                        total_label = y_train_t
                        flag = 1
                    else:
                        total_train = torch.cat((total_train, out), 0)
                        total_label = torch.cat((total_label, y_train_t), 0)
                test_acc = model.acc(total_train, total_label)
                max_test = test_acc
            print("epoch:", epoch, "----dev_acc:", max_dev,"----test_acc:", max_test)
            model.train()
    return max_dev,max_test

def rnn_embedding_train(X_train_c,y_train,X_test_c,y_test,X_dev_c,y_dev,mask_train,mask_test,mask_dev,vect_len,epochs,flag_g,weigth):
    train_data = TensorDataset(X_train_c, y_train,mask_train)
    dev_data = TensorDataset(X_dev_c,y_dev,mask_dev)
    test_data = TensorDataset(X_test_c, y_test,mask_test)

    batch = 512
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batch)
    dev_iter = DataLoader(dev_data,shuffle=True,batch_size=batch)

    model = RNN_e(vect_len,flag_g,weigth,5)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
    model.train()
    loss_s = FocalLoss()
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->parm_value:', parms.data.size())

    max_dev,max_test = 0.0,0.0
    for epoch in range(epochs):
        for batch in train_iter:
            train_id, y_train_t,mask_train_t = batch
            train_id, y_train_t,mask_train_t = train_id.cuda(), y_train_t.cuda(),mask_train_t.cuda()

            out = model(train_id,mask_train_t)
            loss = loss_s(out,y_train_t)
            # loss = model.loss(out, y_train_t)
            # loss = F.cross_entropy(out,y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            # print(loss)
            # print(model.acc(out, y_train_t))
        if epoch%4 == 0:
            model.eval()
            flag = 0
            total_train,total_label_train = None,None
            for batch in dev_iter:
                train_id, y_train_t, mask_train_t = batch
                train_id, y_train_t, mask_train_t = train_id.cuda(), y_train_t.cuda(), mask_train_t.cuda()

                out = model(train_id, mask_train_t)
                # loss = model.loss(out, y_train_t)
                if flag == 0:
                    total_train = out
                    total_label_train = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label_train = torch.cat((total_label_train,y_train_t),0)
            # print(total.size())
            # print(loss)
            dev_acc = model.acc(total_train, total_label_train)
            if max_dev<dev_acc:
                max_dev = dev_acc
                flag = 0
                total_test, total_label_test = None, None
                for batch in test_iter:
                    train_id, y_train_t,mask_test_t = batch
                    train_id, y_train_t,mask_test_t = train_id.cuda(), y_train_t.cuda(),mask_test_t.cuda()
                    out = model(train_id,mask_test_t)
                    # loss = model.loss(out, y_train_t)
                    if flag == 0:
                        total_test = out
                        total_label_test = y_train_t
                        flag = 1
                    else:
                        total_test = torch.cat((total_test, out), 0)
                        total_label_test= torch.cat((total_label_test, y_train_t), 0)

                test_acc = model.acc(total_test, total_label_test)
                max_test = test_acc
            print("epoch:", epoch, "----dev_acc:", max_dev, "----test_acc:", max_test)
            model.train()
    return max_dev,max_test

if __name__ == '__main__':
    print(torch.cuda.is_available())
    # 全部
    data_train = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_2_train.tsv')
    data_test = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_2_test.tsv')
    data_dev = read_tsv('./data/sentiment-analysis-on-movie-reviews/convert_2_dev.tsv')
    X_train,X_split_train,y_train,X_test,X_split_test,y_test,X_dev,X_split_dev,y_dev = [],[],[],[],[],[],[],[],[]
    for i in data_train:
        X_train.append(i[2].lower())
        y_train.append(int(i[3]))
        X_split_train.append(i[2].lower().split(' '))
    for i in data_test:
        X_test.append(i[2].lower())
        y_test.append(int(i[3]))
        X_split_test.append(i[2].lower().split(' '))
    for i in data_dev:
        X_dev.append(i[2].lower())
        y_dev.append(int(i[3]))
        X_split_dev.append(i[2].lower().split(' '))
    X_len_train = [len(i) for i in X_split_train]
    X_len_test = [len(i) for i in X_split_test]
    X_len_dev = [len(i) for i in X_split_dev]
    max_len = 0
    for i in X_len_train:
        max_len = max(max_len,i)
    for i in X_len_dev:
        max_len = max(max_len,i)
    for i in X_len_test:
        max_len = max(max_len,i)

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
    for i in X_split_train:
        for j,k in enumerate(i):
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in X_split_test:
        for j, k in enumerate(i):
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in X_split_dev:
        for j,k in enumerate(i):
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1

    # X = np.array(X)
    # y = np.array(y)
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

    sentence_id_train = id_generation(X_split_train,veccc,vect_len)
    sentence_id_train = [torch.LongTensor(i) for i in sentence_id_train]
    sentence_id_train = pad_sequence(sentence_id_train,batch_first=True,padding_value=vect_len)
    sentence_id_test = id_generation(X_split_test, veccc, vect_len)
    sentence_id_test = [torch.LongTensor(i) for i in sentence_id_test]
    sentence_id_test = pad_sequence(sentence_id_test, batch_first=True, padding_value=vect_len)
    sentence_id_dev = id_generation(X_split_dev, veccc, vect_len)
    sentence_id_dev = [torch.LongTensor(i) for i in sentence_id_dev]
    sentence_id_dev = pad_sequence(sentence_id_dev, batch_first=True, padding_value=vect_len)
    rnn_mask_train = rnn_mask_g(X_len_train)
    cnn_mask_train = cnn_mask_g(X_len_train,max_len)
    rnn_mask_test = rnn_mask_g(X_len_test)
    cnn_mask_test = cnn_mask_g(X_len_test, max_len)
    rnn_mask_dev = rnn_mask_g(X_len_dev)
    cnn_mask_dev = cnn_mask_g(X_len_dev, max_len)
    y_train = torch.tensor(y_train).long()
    y_dev = torch.tensor(y_dev).long()
    y_test = torch.tensor(y_test).long()

    max_cnn_dev,max_cnn_test,max_rnn_dev,max_rnn_test = 0.0,0.0,0.0,0.0
    for i in range(2):
        cnn_dev_acc,cnn_test_acc = cnn_embedding_train(sentence_id_train, y_train, sentence_id_test, y_test,sentence_id_dev,y_dev, cnn_mask_train, cnn_mask_test,cnn_mask_dev, vect_len, 100, i, glove_embed,bias_f=True)
        rnn_dev_acc, rnn_test_acc = rnn_embedding_train(sentence_id_train, y_train, sentence_id_test, y_test,sentence_id_dev,y_dev, rnn_mask_train, rnn_mask_test,rnn_mask_dev, vect_len, 100, i, glove_embed)
        if i==0:
            print('random')
        else:
            print('glove')
        print("cnn:",cnn_dev_acc,'\t',cnn_test_acc)
        print("rnn:",rnn_dev_acc,'\t',rnn_test_acc)

    # X_len = torch.tensor(X_len).long()
   # X_train,y_train,X_test,y_test,mask_train,mask_test = sentence_id[:1600],y[:1600],sentence_id[1600:1900],y[1600:1900],rnn_mask[:1600],rnn_mask[1600:1900]

    # cnn_embedding_train(X_train,y_train,X_test,y_test,mask_train_c,mask_test_c,vect_len,100,0,glove_embed,bias_f=True)
    # rnn_embedding_train(X_train, y_train, X_test, y_test,mask_train_r,mask_test_r, vect_len, 200, 1, glove_embed)
    #cnn与rnn比较
    # epochs = []
    # for i in range(0,100,4):
    #     epochs.append(i)
    #
    # cnn_acc,rnn_acc = [[[],[]],[[],[]]],[[[],[]],[[],[]]]
    # for i in range(2):
    #     cnn_acc[i][0],cnn_acc[i][1] = cnn_embedding_train(X_train, y_train, X_test, y_test, mask_train_c, mask_test_c, vect_len, 100, i, glove_embed,
    #                         bias_f=True)
    #     rnn_acc[i][0], rnn_acc[i][1] = rnn_embedding_train(X_train, y_train, X_test, y_test, mask_train_r, mask_test_r, vect_len, 100, i, glove_embed)
    #
    # matplotlib.pyplot.subplot(2, 1, 1)
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[0][0], 'r--', label='cnn+random')
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[1][0], 'g--', label='cnn+glove')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[0][0], 'b--', label='rnn+random')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[1][0], 'black', label='rnn+glove')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Training Accuracy")
    # matplotlib.pyplot.xlabel("epochs")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.4, 1.0)
    # matplotlib.pyplot.subplot(2, 1, 2)
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[0][1], 'r--', label='cnn+random')
    # matplotlib.pyplot.semilogx(epochs, cnn_acc[1][1], 'g--', label='cnn+glove')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[0][1], 'b--', label='rnn+random')
    # matplotlib.pyplot.semilogx(epochs, rnn_acc[1][1], 'black', label='rnn+glove')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Test Accuracy")
    # matplotlib.pyplot.xlabel("epochs")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.4, 1.0)
    # matplotlib.pyplot.tight_layout()
    # matplotlib.pyplot.show()