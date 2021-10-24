import json

import matplotlib.pyplot
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sentiment_analysis_task2 import get_numpy_word_embed, id_generation, FocalLoss


def read_data(nums):
    with open("./data/snli_1.0/snli_1.0_train.jsonl") as f:
        sentence1,sentence2,label = [],[],[]
        read_nums = 0
        for i in f.readlines():
            d = json.loads(i)
            if len(d["sentence1"].lower().split(' ')) > 16 or len(d["sentence2"].lower().split(' ')) > 16:
                continue
            sentence1.append(d["sentence1"].lower().split(' '))
            sentence2.append(d["sentence2"].lower().split(' '))
            label.append(d["gold_label"])
            read_nums += 1
            if read_nums >= nums:
                break

    return sentence1,sentence2,label

def len_mask_g(data,max_len):
    len_mask = torch.zeros((len(data),1,64)).long()
    softmax_mask = torch.zeros((len(data),max_len)).long()
    for i,j in enumerate(data):
        len_mask[i][0] += j-1
        softmax_mask[i][:j] = 1
    return len_mask,softmax_mask

#下面分别是Reasoning about Entailment with Neural Attention的不同方法
class Out2_cat_encoding(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Out2_cat_encoding, self).__init__()
        self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)

        self.lstm1 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.5,bidirectional=bidi)
        self.lstm2 = nn.LSTM(input_size=50,hidden_size=64,num_layers=1,batch_first=True,dropout=0.5,bidirectional=bidi)
        self.l1 = nn.Sequential(nn.Linear(128, 128),nn.Dropout(),nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128,4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        # with torch.no_grad():
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)

        l1_out,_ = self.lstm1(embed1)
        l1_out = l1_out.gather(1,s1_len).squeeze(1)
        l2_out,(_,_) = self.lstm2(embed2)
        l2_out = l2_out.gather(1,s2_len).squeeze(1)
        out = torch.cat((l1_out,l2_out),1)
        out = self.l1(out)
        out = self.l(out)
        return out

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        # for i in range(100):
        #     if y_pre[i + 50] != label[i + 50]:
        #         print("pre:", y_pre[i + 50], "  true:", label[i + 50])

        return y_acc

class Conditional_encoding(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Conditional_encoding, self).__init__()
        self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50,hidden_size=64,num_layers=1,batch_first=True,dropout=0.3,bidirectional=bidi)
        self.l1 = nn.Sequential(nn.Linear(64, 128),nn.Dropout(),nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128,4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)

        embed1 = embed1.permute(1,0,2)
        h0 = torch.zeros(embed1.size()[1],64,requires_grad=False).cuda()
        c0 = torch.zeros(embed1.size()[1],64,requires_grad=False).cuda()
        h,c = [],[]
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0,c0 = self.lstm1(embed1[i],(h0,c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1,0,2)
        c = c.permute(1,0,2)

        gather_h = h.gather(1, s1_len).view(1,embed2.size()[0],64)
        gather_c = c.gather(1, s1_len).view(1,embed2.size()[0],64)

        out,(_,_) = self.lstm2(embed2,(gather_h,gather_c))
        out = out.gather(1,s2_len).squeeze(1)

        out = self.l1(out)
        out = self.l(out)
        return out

    def acc(self,y_pre,label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        # for i in range(100):
        #     if y_pre[i + 50] != label[i + 50]:
        #         print("pre:", y_pre[i + 50], "  true:", label[i + 50])

        return y_acc

class Attenion(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Attenion, self).__init__()
        self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=bidi)
        self.wy = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64,64)))
        self.wh = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.w = nn.Parameter(nn.init.normal_(torch.Tensor(64)))
        self.wp = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wx = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.l1 = nn.Sequential(nn.Linear(64, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b,s,_ = embed1.size()
        e = 64

        embed1 = embed1.permute(1, 0, 2)
        h0 = torch.zeros(b, e, requires_grad=False).cuda()
        c0 = torch.zeros(b, e, requires_grad=False).cuda()
        h, c = [], []
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0, c0 = self.lstm1(embed1[i], (h0, c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        gather_h = h.gather(1, s1_len).view(1, b, e)
        gather_c = c.gather(1, s1_len).view(1, b, e)

        out, (_, _) = self.lstm2(embed2, (gather_h, gather_c))
        out = out.gather(1, s2_len)

        wyy = torch.matmul(h,self.wy)
        hnel = out.expand(b,s,e)
        whhn = torch.matmul(hnel,self.wh)
        M = self.tanh(wyy+whhn)
        wt = self.w.view(64,1)
        wtm = torch.matmul(M,wt).squeeze(2)
        s1_mask = s1_s*wtm-(1-s1_s)*1e12
        soft = self.softmax(s1_mask).view(b,s,1)
        soft = soft.expand(b,s,e)
        r = h*soft
        r = torch.sum(r,1)
        wpr = torch.matmul(r,self.wp)
        wxhx = torch.matmul(out.squeeze(1),self.wx)
        h_out = self.tanh(wpr+wxhx)
        h_out = self.l1(h_out)
        h_out = self.l(h_out)
        return h_out

    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        # for i in range(100):
        #     if y_pre[i + 50] != label[i + 50]:
        #         print("pre:", y_pre[i + 50], "  true:", label[i + 50])
        return y_acc

class Wbw_Attenion(nn.Module):
    def __init__(self,vect_len,weight,bidi=False):
        super(Wbw_Attenion, self).__init__()
        self.embedding = nn.Embedding(vect_len + 2, 50, _weight=weight)

        self.lstm1 = nn.LSTMCell(input_size=50, hidden_size=64)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=bidi)
        self.wy = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64,64)))
        self.wh = nn.Parameter( nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.w = nn.Parameter(nn.init.normal_(torch.Tensor(64)))
        self.wp = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wx = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wr = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.wt = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(64, 64)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.l1 = nn.Sequential(nn.Linear(64, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, 4))

    def forward(self,sentence1,sentence2,s1_len,s2_len,s1_s,s2_s):
        embed1 = self.embedding(sentence1)
        embed2 = self.embedding(sentence2)
        b,s,_ = embed1.size()
        _,s2,_ = embed2.size()
        e = 64

        embed1 = embed1.permute(1, 0, 2)
        h0 = torch.zeros(b, e, requires_grad=False).cuda()
        c0 = torch.zeros(b, e, requires_grad=False).cuda()
        h, c = [], []
        # print(embed1[0].size())
        for i in range(embed1.size()[0]):
            h0, c0 = self.lstm1(embed1[i], (h0, c0))
            h += [h0]
            c += [c0]
        h = torch.stack(h)
        c = torch.stack(c)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)

        gather_h = h.gather(1, s1_len).view(1, b, e)
        gather_c = c.gather(1, s1_len).view(1, b, e)

        out, (_, _) = self.lstm2(embed2, (gather_h, gather_c))
        # out = out.gather(1, s2_len)
        r = torch.zeros(b,e,requires_grad=False).cuda()
        Rt = []
        for i in range(s2):
            wyy = torch.matmul(h,self.wy)
            hnel = out.expand(b,s,e)
            whhn = torch.matmul(hnel,self.wh)
            rtel = r.unsqueeze(1).expand(b,s,e)
            wrrt = torch.matmul(rtel,self.wr)
            M = self.tanh(wyy+whhn+wrrt)

            wt = self.w.view(64,1)
            wtm = torch.matmul(M,wt).squeeze(2)
            s1_mask = s1_s*wtm-(1-s1_s)*1e12
            soft = self.softmax(s1_mask).view(b,s,1)
            soft = soft.expand(b,s,e)

            wtrt = torch.matmul(r,self.wt)
            r = h*soft
            r = torch.sum(r,1)+self.tanh(wtrt)
            Rt += [r]
        Rt = torch.stack(Rt)
        Rt = Rt.permute(1,0,2)
        rn = Rt.gather(1,s2_len).squeeze(1)
        hn_out = out.gather(1, s2_len)
        wpr = torch.matmul(rn,self.wp)
        wxhx = torch.matmul(hn_out.squeeze(1),self.wx)
        h_out = self.tanh(wpr+wxhx)
        h_out = self.l1(h_out)
        h_out = self.l(h_out)
        return h_out

    def acc(self, y_pre, label):
        lenn = len(y_pre)
        y_pre = y_pre.argmax(axis=1)
        y_acc = sum([y_pre[i] == label[i] for i in range(lenn)]) / lenn

        # for i in range(100):
        #     if y_pre[i + 50] != label[i + 50]:
        #         print("pre:", y_pre[i + 50], "  true:", label[i + 50])
        return y_acc


def trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs,model_t,batchs):
    train_data = TensorDataset(s1_train,s2_train, y_train,s1_len_train,s2_len_train,s1_s_train,s2_s_train)
    test_data = TensorDataset(s1_test,s2_test, y_test,s1_len_test,s2_len_test,s1_s_test,s2_s_test)

    batch = batchs
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batch)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batch)

    model = model_t
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.0003)
    model.train()
    loss_s = FocalLoss()

    train_acc_total, test_acc_total = [], []
    for epoch in range(epochs):
        for batch in train_iter:
            s1_train_t,s2_train_t, y_train_t,s1_len_train_t,s2_len_train_t,s1_s_train_t,s2_s_train_t = batch
            s1_train_t,s2_train_t, y_train_t,s1_len_train_t,s2_len_train_t,s1_s_train_t,s2_s_train_t = \
                s1_train_t.cuda(),s2_train_t.cuda(), y_train_t.cuda(),s1_len_train_t.cuda(),s2_len_train_t.cuda(),s1_s_train_t.cuda(),s2_s_train_t.cuda()

            out = model(s1_train_t, s2_train_t,s1_len_train_t,s2_len_train_t,s1_s_train_t,s2_s_train_t)
            loss = loss_s(out, y_train_t)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 3 == 0 and epoch != 0:
            model.eval()
            flag = 0
            total_train, total_label_train = None, None
            rnn_out = None
            # if epoch>40:
            #     for name, parms in model.named_parameters():
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->parm_value:', parms.data)
            for batch in train_iter:
                s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
                s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                    s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()

                out = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
                if flag == 0:
                    total_train = out
                    total_label_train = y_train_t
                    flag = 1
                else:
                    total_train = torch.cat((total_train, out), 0)
                    total_label_train = torch.cat((total_label_train, y_train_t), 0)
            train_acc = model.acc(total_train, total_label_train)
            flag = 0
            total_test, total_label_test = None, None
            for batch in test_iter:
                s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = batch
                s1_train_t, s2_train_t, y_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t = \
                    s1_train_t.cuda(), s2_train_t.cuda(), y_train_t.cuda(), s1_len_train_t.cuda(), s2_len_train_t.cuda(), s1_s_train_t.cuda(), s2_s_train_t.cuda()

                out = model(s1_train_t, s2_train_t, s1_len_train_t, s2_len_train_t, s1_s_train_t, s2_s_train_t)
                if flag == 0:
                    total_test = out
                    total_label_test = y_train_t
                    flag = 1
                else:
                    total_test = torch.cat((total_test, out), 0)
                    total_label_test = torch.cat((total_label_test, y_train_t), 0)

            test_acc = model.acc(total_test, total_label_test)
            train_acc_total.append(train_acc)
            test_acc_total.append(test_acc)
            print("epoch:", epoch, "----train_acc:", train_acc, "----test_acc:", test_acc)
            model.train()
    return train_acc_total, test_acc_total

if __name__=='__main__':
    sentence1, sentence2, label = read_data(10000)

    s1_len = [len(i) for i in sentence1]
    s2_len = [len(i) for i in sentence2]
    # lenn1,lenn2 = {},{}
    # for i in range(len(sentence1)):
    #     lenn1[s1_len[i]] = lenn1.get(s1_len[i],0)+1
    #     lenn2[s2_len[i]] = lenn2.get(s2_len[i], 0) + 1
    #
    # lenn1 = sorted(lenn1.items(),key=lambda x:x[1])
    # lenn2 = sorted(lenn2.items(), key=lambda x: x[1])
    # print(lenn1)
    # print(lenn2)

    max_len = 0
    for i in range(len(sentence1)):
        max_len = max(max_len, s1_len[i],s2_len[i])

    veccc = {}
    vect_len = 0
    for i in range(len(sentence1)):
        for j, k in enumerate(sentence1[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
        for j, k in enumerate(sentence2[i]):
            # print(k)
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1

    labels = []
    for i in label:
        if i=="entailment":
            labels.append(0)
        elif i=="neutral":
            labels.append(1)
        elif i=="-":
            labels.append(2)
        elif i=="contradiction":
            labels.append(3)

    veccc['[UNK]'] = vect_len
    veccc['[PAD]'] = vect_len + 1
    glove_embed = get_numpy_word_embed(veccc)
    glove_embed = torch.FloatTensor(glove_embed)
    s1_len,s1_softmax = len_mask_g(s1_len,max_len)
    s2_len,s2_softmax = len_mask_g(s2_len,max_len)


    sentence1_id = id_generation(sentence1, veccc)
    sentence2_id = id_generation(sentence2, veccc)
    sentence1_id = [torch.LongTensor(i) for i in sentence1_id]
    sentence2_id = [torch.LongTensor(i) for i in sentence2_id]
    sentence1_id = pad_sequence(sentence1_id, batch_first=True, padding_value=vect_len)
    sentence2_id = pad_sequence(sentence2_id, batch_first=True, padding_value=vect_len)
    y = torch.tensor(labels).long()

    s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test = \
        train_test_split(sentence1_id,sentence2_id, y,s1_len,s2_len,s1_softmax,s2_softmax, train_size=0.8, random_state=123, shuffle=True)

    # model = Out2_cat_encoding(vect_len, glove_embed, False)
    # # # model = Conditional_encoding(vect_len,glove_embed,False)
    # model = Attenion(vect_len, glove_embed, False)
    # model = Wbw_Attenion(vect_len, glove_embed, False)
    # trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=100,model_t=model,batchs=256)

    #比较微调和不微调的区别，如果要比较，第一行read_data(100000),trainning函数里也需要修改del的次数
    # epochs = []
    # for i in range(0,300,4):
    #     epochs.append(i)
    #
    # embed,embed_ft = [[],[]],[[],[]]
    #
    # model = Out2_cat_encoding1(vect_len, glove_embed, False)
    # embed[0],embed[1] = trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=300,model_t=model)
    # model = Out2_cat_encoding(vect_len, glove_embed, False)
    # embed_ft[0], embed_ft[1] = trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=300,model_t=model)
    #
    # matplotlib.pyplot.subplot(2, 1, 1)
    # matplotlib.pyplot.semilogx(epochs, embed[0], 'r--', label='embed_froze')
    # matplotlib.pyplot.semilogx(epochs, embed_ft[0], 'g--', label='embed')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Training Accuracy")
    # matplotlib.pyplot.xlabel("epochs")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.4, 1.0)
    # matplotlib.pyplot.subplot(2, 1, 2)
    # matplotlib.pyplot.semilogx(epochs, embed[1], 'r--', label='embed_froze')
    # matplotlib.pyplot.semilogx(epochs, embed_ft[1], 'g--', label='embed')
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.title("Test Accuracy")
    # matplotlib.pyplot.xlabel("epochs")
    # matplotlib.pyplot.ylabel("Accuracy")
    # matplotlib.pyplot.ylim(0.4, 1.0)
    # matplotlib.pyplot.tight_layout()
    # matplotlib.pyplot.show()

    #四个模型比较
    epochs = []
    for i in range(1,100,3):
        epochs.append(i)

    two_lstm_cat,conditional_encoding,attention,wbw_attention = [[],[]],[[],[]],[[],[]],[[],[]]

    model = Out2_cat_encoding(vect_len, glove_embed, False)
    two_lstm_cat[0],two_lstm_cat[1] = trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=100,model_t=model,batchs=512)
    model = Conditional_encoding(vect_len, glove_embed, False)
    conditional_encoding[0], conditional_encoding[1] = trainning(s1_train, s1_test,s2_train, s2_test, y_train, y_test,s1_len_train,s1_len_test,s2_len_train,s2_len_test,s1_s_train,s1_s_test,s2_s_train,s2_s_test,epochs=100,model_t=model,batchs=256)
    model = Attenion(vect_len, glove_embed, False)
    attention[0], attention[1] = trainning(s1_train, s1_test, s2_train, s2_test, y_train, y_test, s1_len_train,s1_len_test, s2_len_train, s2_len_test, s1_s_train, s1_s_test,s2_s_train, s2_s_test, epochs=100, model_t=model, batchs=256)
    model = Wbw_Attenion(vect_len, glove_embed, False)
    wbw_attention[0], wbw_attention[1] = trainning(s1_train, s1_test, s2_train, s2_test, y_train, y_test,s1_len_train, s1_len_test, s2_len_train, s2_len_test,s1_s_train, s1_s_test, s2_s_train, s2_s_test,epochs=100, model_t=model, batchs=64)

    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.semilogx(epochs, two_lstm_cat[0], 'r--', label='two_lstm_cat')
    matplotlib.pyplot.semilogx(epochs, conditional_encoding[0], 'g--', label='conditional_encoding')
    matplotlib.pyplot.semilogx(epochs, attention[0], 'b--', label='attention')
    matplotlib.pyplot.semilogx(epochs, wbw_attention[0], 'black', label='wbw_attention')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("Training Accuracy")
    matplotlib.pyplot.xlabel("epochs")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0.4, 1.0)
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.semilogx(epochs, two_lstm_cat[1], 'r--', label='two_lstm_cat')
    matplotlib.pyplot.semilogx(epochs, conditional_encoding[1], 'g--', label='conditional_encoding')
    matplotlib.pyplot.semilogx(epochs, attention[1], 'b--', label='attention')
    matplotlib.pyplot.semilogx(epochs, wbw_attention[1], 'black', label='wbw_attention')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title("Test Accuracy")
    matplotlib.pyplot.xlabel("epochs")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0.4, 0.8)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()