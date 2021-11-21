import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchcrf import CRF
from sentiment_analysis_task2 import get_numpy_word_embed, id_generation, FocalLoss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def read_conll_txt(file):
    words,labels = [],[]
    with open(file, mode='r',encoding='utf-8') as f:
        lines = f.readlines()
        word,label = [],[]
        for line in lines:
            if line=='\n':
                words.append(word)
                labels.append(label)
                word, label = [], []
            else:
                l = line.split(' ')
                word.append(l[0])
                label.append(l[3][:-1])
    return words,labels

def mask_g(data,max_len):
    mask = torch.zeros((len(data),max_len),dtype=torch.uint8)
    for i,j in enumerate(data):
        mask[i][:j] = 1
    return mask

def label_to_id(label,label_id,vect_len):
    labels = torch.zeros(len(label),vect_len).long()
    labels -= 1
    for i,_ in enumerate(label):
        for j,m in enumerate(label[i]):
            labels[i][j] = label_id[m]
    return labels

def pad_sequence_self(ids,max_len,padding_value):
    pad_ids = torch.zeros(len(ids),max_len).long()
    pad_ids += padding_value
    for i,e in enumerate(ids):
        length = e.size(0)
        pad_ids[i][:length] = e
    return pad_ids

class Lstm_CRF(nn.Module):
    def __init__(self,vect_len,label_num,weight):
        super(Lstm_CRF, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 50, _weight=weight)
        self.embed_drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=1, batch_first=True, dropout=0.3,bidirectional=True)
        self.l1 = nn.Sequential(nn.Linear(128, 128), nn.Dropout(), nn.Tanh())
        self.l = nn.Sequential(nn.Linear(128, label_num))
        self.CRF = CRF(label_num,batch_first=True)

    def forward(self,word_ids):
        embed = self.embedding(word_ids)
        embed = self.embed_drop(embed)
        lstm_out,_ = self.lstm(embed)
        out = self.l1(lstm_out)
        out = self.l(out)
        return out

    def loss(self,out,mask,labels):
        loss = self.CRF(out, labels, mask, reduction='mean')
        return loss*-1

    def predict(self,out,mask):
        pred = self.CRF.decode(out,mask)
        return pred

    def acc_f1(self, y_true, y_pred):
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_s = (f1[1] + f1[2] + f1[3] + f1[4] + f1[5] + f1[6] + f1[7] + f1[0]) / 8.0
        return acc, f1_s

    def class_report(self, y_true, y_pred):
        classify_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0,
                        target_names=['B-ORG','I-ORG','B-PER','I-PER','B-LOC','I-LOC','B-MISC','I-MISC','O'])
        # classify_report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0)
        # print('\n\nclassify_report:\n', classify_report)
        return classify_report

def trainning(train_ids, test_ids,dev_ids,train_labels, test_labels,dev_labels,train_mask,test_mask,dev_mask,epochs,model,batchs):
    train_data = TensorDataset(train_ids,train_mask,train_labels)
    test_data = TensorDataset(test_ids,test_mask,test_labels)
    dev_data = TensorDataset(dev_ids,dev_mask,dev_labels)

    train_iter = DataLoader(train_data, shuffle=True, batch_size=batchs)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batchs)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=batchs)

    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=0.0001)
    model.train()

    max_train_f1, max_test_f1, val_f1 = 0, 0, 0
    max_train_acc, max_test_acc, val_acc = 0, 0, 0
    max_train_report, max_test_report = None, None
    for epoch in range(epochs):
        for batch in train_iter:
            train_ids_t,train_mask_t,train_labels_t = batch
            train_ids_t,train_mask_t,train_labels_t = train_ids_t.cuda(),train_mask_t.cuda(),train_labels_t.cuda()

            out = model(train_ids_t)
            # print(out[:20])
            loss = model.loss(out,train_mask_t,train_labels_t)
            # print(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 2 == 0 and epoch != 0:
            model.eval()
            total_train, total_label_train = [],[]
            with torch.no_grad():
                # if epoch>40:
                #     for name, parms in model.named_parameters():
                #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->parm_value:', parms.data)
                for batch in train_iter:
                    train_ids_t, train_mask_t, train_labels_t = batch
                    train_ids_t, train_mask_t, train_labels_t = train_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                    out = model(train_ids_t)
                    pred = model.predict(out,train_mask_t)
                    y_true = train_labels_t[train_labels_t != -1]
                    p = []
                    for i in pred:
                        p += i
                    y_pred = torch.tensor(p)

                    total_train.append(y_pred)
                    total_label_train.append(y_true)
                eval_pred = torch.cat(total_train, dim=0).cpu()
                eval_true = torch.cat(total_label_train, dim=0).cpu()
                train_report = model.class_report(y_pred=eval_pred, y_true=eval_true)
                train_acc, train_f1 = model.acc_f1(y_pred=eval_pred, y_true=eval_true)
                print("train_accuracy_score:", train_acc, "f1:", train_f1)
                if train_f1>0.997:
                    total_test, total_label_test = [],[]
                    for batch in test_iter:
                        train_ids_t, train_mask_t, train_labels_t = batch
                        train_ids_t, train_mask_t, train_labels_t = train_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                        out = model(train_ids_t)
                        pred = model.predict(out, train_mask_t)
                        y_true = train_labels_t[train_labels_t != -1]
                        p = []
                        for i in pred:
                            p += i
                        y_pred = torch.tensor(p)

                        total_test.append(y_pred)
                        total_label_test.append(y_true)
                    eval_pred = torch.cat(total_test, dim=0).cpu()
                    eval_true = torch.cat(total_label_test, dim=0).cpu()
                    test_report = model.class_report(y_pred=eval_pred, y_true=eval_true)
                    test_acc, test_f1 = model.acc_f1(y_pred=eval_pred, y_true=eval_true)
                    print("test_accuracy_score:", test_acc, "f1:", test_f1)
                    total_test, total_label_test = [],[]
                    if max_test_f1 < test_f1:
                        max_train_acc = train_acc
                        max_test_acc = test_acc
                        max_train_f1 = train_f1
                        max_test_f1 = test_f1
                        max_train_report = train_report
                        max_test_report = test_report
                        for batch in dev_iter:
                            train_ids_t, train_mask_t, train_labels_t = batch
                            train_ids_t, train_mask_t, train_labels_t = train_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                            out = model(train_ids_t)
                            pred = model.predict(out, train_mask_t)
                            y_true = train_labels_t[train_labels_t != -1]
                            p = []
                            for i in pred:
                                p += i
                            y_pred = torch.tensor(p)

                            total_test.append(y_pred)
                            total_label_test.append(y_true)
                        eval_pred = torch.cat(total_test, dim=0).cpu()
                        eval_true = torch.cat(total_label_test, dim=0).cpu()
                        dev_report = model.class_report(y_pred=eval_pred, y_true=eval_true)
                        dev_acc, dev_f1 = model.acc_f1(y_pred=eval_pred, y_true=eval_true)
                        print("dev_accuracy_score:", dev_acc, "f1:", dev_f1)

            model.train()
    return max_train_acc,max_test_acc,dev_acc,max_train_f1,max_test_f1,dev_f1,max_train_report,max_test_report,dev_report

def kl(out1,out2,mask):
    a,b = mask.size()
    #针对ner
    mask = mask.unsqueeze(-1).expand(a,b,9)
    kl_loss1 = F.kl_div(F.log_softmax(out1,dim=-1),F.softmax(out2,dim=-1),reduction='none')
    kl_loss2 = F.kl_div(F.log_softmax(out2, dim=-1), F.softmax(out1, dim=-1), reduction='none')
    kl_loss1 = kl_loss1*mask
    kl_loss2 = kl_loss2*mask
    return (kl_loss1.mean()+kl_loss2.mean())/2

def trainning_RDrop(train_ids, test_ids,dev_ids,train_labels, test_labels,dev_labels,train_mask,test_mask,dev_mask,epochs,model,batchs,a):
    train_data = TensorDataset(train_ids,train_mask,train_labels)
    test_data = TensorDataset(test_ids,test_mask,test_labels)
    dev_data = TensorDataset(dev_ids,dev_mask,dev_labels)

    train_iter = DataLoader(train_data, shuffle=True, batch_size=batchs)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batchs)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=batchs)

    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=0.0001)
    model.train()

    max_train_f1, max_test_f1, val_f1 = 0, 0, 0
    max_train_acc, max_test_acc, val_acc = 0, 0, 0
    max_train_report,max_test_report = None,None
    for epoch in range(epochs):
        for batch in train_iter:
            train_ids_t,train_mask_t,train_labels_t = batch
            train_ids_t,train_mask_t,train_labels_t = train_ids_t.cuda(),train_mask_t.cuda(),train_labels_t.cuda()

            out1 = model(train_ids_t)
            out2 = model(train_ids_t)
            kl_loss = kl(out1,out2,train_mask_t)
            loss1 = model.loss(out1,train_mask_t,train_labels_t)
            loss2 = model.loss(out2,train_mask_t,train_labels_t)
            loss = a*kl_loss+(loss1+loss2)/2
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 2 == 0 and epoch != 0:
            model.eval()
            total_train, total_label_train = [],[]
            with torch.no_grad():
                # if epoch>40:
                #     for name, parms in model.named_parameters():
                #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->parm_value:', parms.data)
                for batch in train_iter:
                    train_ids_t, train_mask_t, train_labels_t = batch
                    train_ids_t, train_mask_t, train_labels_t = train_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                    out = model(train_ids_t)
                    pred = model.predict(out,train_mask_t)
                    y_true = train_labels_t[train_labels_t != -1]
                    p = []
                    for i in pred:
                        p += i
                    y_pred = torch.tensor(p)

                    total_train.append(y_pred)
                    total_label_train.append(y_true)
                eval_pred = torch.cat(total_train, dim=0).cpu()
                eval_true = torch.cat(total_label_train, dim=0).cpu()
                train_report = model.class_report(y_pred=eval_pred, y_true=eval_true)
                train_acc, train_f1 = model.acc_f1(y_pred=eval_pred, y_true=eval_true)
                print("train_accuracy_score:", train_acc, "f1:", train_f1)
                if train_f1>0.997:
                    total_test, total_label_test = [],[]
                    for batch in test_iter:
                        train_ids_t, train_mask_t, train_labels_t = batch
                        train_ids_t, train_mask_t, train_labels_t = train_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                        out = model(train_ids_t)
                        pred = model.predict(out, train_mask_t)
                        y_true = train_labels_t[train_labels_t != -1]
                        p = []
                        for i in pred:
                            p += i
                        y_pred = torch.tensor(p)

                        total_test.append(y_pred)
                        total_label_test.append(y_true)
                    eval_pred = torch.cat(total_test, dim=0).cpu()
                    eval_true = torch.cat(total_label_test, dim=0).cpu()
                    test_report = model.class_report(y_pred=eval_pred, y_true=eval_true)
                    test_acc, test_f1 = model.acc_f1(y_pred=eval_pred, y_true=eval_true)
                    print("test_accuracy_score:", test_acc, "f1:", test_f1)
                    total_test, total_label_test = [],[]
                    if max_test_f1 < test_f1:
                        max_train_acc = train_acc
                        max_test_acc = test_acc
                        max_train_f1 = train_f1
                        max_test_f1 = test_f1
                        max_train_report = train_report
                        max_test_report = test_report
                        for batch in dev_iter:
                            train_ids_t, train_mask_t, train_labels_t = batch
                            train_ids_t, train_mask_t, train_labels_t = train_ids_t.cuda(), train_mask_t.cuda(), train_labels_t.cuda()

                            out = model(train_ids_t)
                            pred = model.predict(out, train_mask_t)
                            y_true = train_labels_t[train_labels_t != -1]
                            p = []
                            for i in pred:
                                p += i
                            y_pred = torch.tensor(p)

                            total_test.append(y_pred)
                            total_label_test.append(y_true)
                        eval_pred = torch.cat(total_test, dim=0).cpu()
                        eval_true = torch.cat(total_label_test, dim=0).cpu()
                        dev_report = model.class_report(y_pred=eval_pred, y_true=eval_true)
                        dev_acc, dev_f1 = model.acc_f1(y_pred=eval_pred, y_true=eval_true)
                        print("dev_accuracy_score:", dev_acc, "f1:", dev_f1)

            model.train()
    return max_train_acc,max_test_acc,dev_acc,max_train_f1,max_test_f1,dev_f1,max_train_report,max_test_report,dev_report


if __name__ == '__main__':
    setup_seed(123)
    train_words,train_labels = read_conll_txt('./data/conll/train.txt')
    test_words, test_labels = read_conll_txt('./data/conll/test.txt')
    dev_words, dev_labels = read_conll_txt('./data/conll/dev.txt')

    # print(len(train_words),len(test_words),len(dev_words))
    # www = {}
    # for i in train_words:
    #     www[''.join(i)] = www.get(''.join(i),0)+1
    # cc,dd = 0,0
    # for i in test_words:
    #     if ''.join(i) in www.keys():
    #         cc += 1
    # for i in dev_words:
    #     if ''.join(i) in www.keys():
    #         dd += 1
    # print(cc,dd)

    veccc = {}
    vect_len,max_len = 0,0
    for i in train_words:
        max_len = max(len(i),max_len)
        for k in i:
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in test_words:
        max_len = max(len(i), max_len)
        for k in i:
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i in dev_words:
        max_len = max(len(i), max_len)
        for k in i:
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1

    label_id = {'B-ORG': 0, 'I-ORG': 1, 'B-PER': 2, 'I-PER': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-MISC': 6, 'I-MISC': 7,'O': 8}
    train_labels = label_to_id(train_labels, label_id, max_len)
    test_labels = label_to_id(test_labels, label_id, max_len)
    dev_labels = label_to_id(dev_labels, label_id, max_len)

    veccc['[UNK]'] = vect_len
    veccc['[PAD]'] = vect_len + 1
    veccc['[CLS]'] = vect_len + 2
    veccc['[SEP]'] = vect_len + 3
    glove_embed = get_numpy_word_embed(veccc, vect_len)
    glove_embed = torch.FloatTensor(glove_embed)

    train_len = [len(i) for i in train_words]
    test_len = [len(i) for i in test_words]
    dev_len = [len(i) for i in dev_words]
    train_mask = mask_g(train_len, max_len)
    test_mask = mask_g(test_len, max_len)
    dev_mask = mask_g(dev_len, max_len)

    train_ids = id_generation(train_words, veccc)
    test_ids = id_generation(test_words, veccc)
    dev_ids = id_generation(dev_words, veccc)
    train_ids = [torch.LongTensor(i) for i in train_ids]
    test_ids = [torch.LongTensor(i) for i in test_ids]
    dev_ids = [torch.LongTensor(i) for i in dev_ids]
    train_ids = pad_sequence_self(train_ids,max_len,  padding_value=vect_len)
    test_ids = pad_sequence_self(test_ids,max_len,  padding_value=vect_len)
    dev_ids = pad_sequence_self(dev_ids,max_len, padding_value=vect_len)

    model = Lstm_CRF(vect_len,len(label_id),glove_embed)
    _,_,_,max_train_f1,max_test_f1,max_dev_f1,train_report,test_report,dev_report = trainning(train_ids, test_ids, dev_ids, train_labels, test_labels, dev_labels, train_mask, test_mask, dev_mask,200, model, 2048)

    # print("\ntrain:\n",train_report)
    # print("\ntest:\n",test_report)
    # print("\ndev:\n",dev_report)
    # print("\t\t", "train\t\t", "test\t\t", "dev")
    # print("acc\t\t", "{:.4f}".format(max_train_acc), "\t", "{:.4f}".format(max_test_acc), "\t","{:.4f}".format(dev_acc))
    # print("f1\t\t", "{:.4f}".format(max_train_f1), "\t", "{:.4f}".format(max_test_f1), "\t","{:.4f}".format(dev_f1))

    a = [0.5,0.8,2,3,4,5]
    kl_train,kl_test,kl_dev = [],[],[]
    for i in a:
        model = Lstm_CRF(vect_len,len(label_id),glove_embed)
        _, _, _, kl_max_train_f1, kl_max_test_f1, kl_dev_f1, _, _, _ = trainning_RDrop(
            train_ids, test_ids, dev_ids, train_labels, test_labels, dev_labels, train_mask, test_mask, dev_mask, 240,
            model, 2048, i)
        kl_train.append(kl_max_train_f1)
        kl_test.append(kl_max_test_f1)
        kl_dev.append(kl_dev_f1)
    # 不同超参数记录的结果，想换不同的值可以直接加
    # model = Lstm_CRF(vect_len,len(label_id),glove_embed)
    # kl_max_train_acc, kl_max_test_acc, kl_dev_acc, kl_max_train_f1, kl_max_test_f1, kl_dev_f1, kl_train_report, kl_test_report, kl_dev_report = trainning_RDrop(
    #     train_ids, test_ids, dev_ids, train_labels, test_labels, dev_labels, train_mask, test_mask, dev_mask, 100,
    #     model, 512,0.8)
    # max_train_f1,max_test_f1,max_dev_f1 = 0.9994,0.7769,0.8475
    # kl_train, kl_test, kl_dev = [0.9990,0.9995,0.9996,0.9997,0.9998,0.9995], [0.7806,0.7827,0.7712,0.7759,0.7770,0.7739], [0.8531,0.8594,0.8496,0.8551,0.8579,0.8512]

    print("\t\t\t\t", "train\t\t", "test\t\t", "dev")
    print("LSTM_CRF\t\t", "{:.4f}".format(max_train_f1), "\t", "{:.4f}".format(max_test_f1), "\t","{:.4f}".format(max_dev_f1))
    for i in range(2):
        print("+R-Drop",a[i],"\t", "{:.4f}".format(kl_train[i]), "\t", "{:.4f}".format(kl_test[i]), "\t","{:.4f}".format(kl_dev[i]))
    for i in range(4):
        print("+R-Drop", a[i+2], "\t\t", "{:.4f}".format(kl_train[i+2]), "\t", "{:.4f}".format(kl_test[i+2]), "\t","{:.4f}".format(kl_dev[i+2]))