import collections
import math
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sentiment_analysis_task2 import get_numpy_word_embed, id_generation


def read_peotry(file):
    poetrys = []
    with open(file,mode='r',encoding='utf-8') as f:
        lines = f.readlines()
        sen_head = ''
        poetry = []
        for line in lines:
            if line != '\n':
                sentence_one = line.split('。')
                if sentence_one[-1] == '\n':
                    sentence_one = sentence_one[:-1]
                if sentence_one[-1][-1:] == '\n':
                    sentence_one[-1] = sentence_one[-1][:-1]
                if sen_head != '':
                    sentence_one[0] = sen_head+sentence_one[0]
                for i in sentence_one:
                    if i[-1:] != '，' and '，' in i:
                        poetry.append(i+'。')
                        sen_head = ''
                    else:
                        sen_head = i[:-1]+'，'
            else:
                poetrys.append(poetry)
                poetry = []
    return poetrys

def get_word2vec_embed(word2ix,lenn=1e6):
    row = 0
    file = 'sgns.sikuquanshu.bigram'
    path = 'D:\\transformerFileDownload\\word2vec-chinese'
    whole = os.path.join(path, file)
    words_embed = {}
    with open(whole, mode='r',encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
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
    bos_eos = np.random.normal(scale=0.5,size=600).tolist()
    for ix in range(len(word2ix)):
        if ix2word[ix] in words_embed:
            id2emb[ix] = words_embed[ix2word[ix]]
        else:
            id2emb[ix] = [0.0] * 300
    id2emb[lenn+2] = bos_eos[:300]
    id2emb[lenn+3] = bos_eos[300:]
    data = [id2emb[ix] for ix in range(len(word2ix))]
    # print(data)

    return data

class Generation_Encoder(nn.Module):
    def __init__(self,vect_len,weight):
        super(Generation_Encoder, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 300, _weight=weight)
        self.embed_drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=1, batch_first=True,bidirectional=False)

    def forward(self,word_ids):
        # embed = self.embed_drop(self.embedding(word_ids))
        embed = self.embedding(word_ids)
        all_steps,(last_hidden,_) = self.lstm(embed)

        return all_steps,last_hidden

class Generation_Decoder(nn.Module):
    def __init__(self,vect_len,weight,encoder_hidden=256):
        super(Generation_Decoder, self).__init__()
        self.embedding = nn.Embedding(vect_len + 4, 300, _weight=weight)
        self.embed_drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=300+encoder_hidden, hidden_size=256, num_layers=1, batch_first=True,bidirectional=False)
        self.l = nn.Linear(256+encoder_hidden,vect_len+4)

    def forward(self,word_id,encoder_all_steps,encoder_last_hidden,input_hidden,input_cell):
        # embed_one = self.embed_drop(self.embedding(word_id))
        embed_one = self.embedding(word_id)
        b,s,_ = embed_one.size()
        encoder_last_hidden = encoder_last_hidden.permute(1,0,2).contiguous().view(b,1,-1)
        encoder_last_hidden = encoder_last_hidden.expand(b,s,256)
        decoder_input = torch.cat((embed_one,encoder_last_hidden),dim=2)
        decoder_lstm_out,(decoder_hidden,decoder_cell) = self.lstm(decoder_input,(input_hidden,input_cell))
        decoder_lstm_out_per = decoder_lstm_out.permute(0,2,1)
        attention = torch.matmul(encoder_all_steps,decoder_lstm_out_per)
        attention = attention.permute(0,2,1)
        attention_out = torch.matmul(attention,encoder_all_steps)
        out = torch.cat((decoder_lstm_out,attention_out),dim=2)
        out = self.l(out)

        return out,decoder_hidden,decoder_cell

class Encoder_Decoder(nn.Module):
    def __init__(self,vect_len,weight):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Generation_Encoder(vect_len,weight)
        self.decoder = Generation_Decoder(vect_len,weight)
        self.cross_loss = nn.CrossEntropyLoss()

    def encoder_out(self,encoder_ids):
        encoder_all, encoder_last = self.encoder(encoder_ids)
        return encoder_all,encoder_last

    def decoder_out(self,decoder_ids,encoder_all,encoder_last,input_hidden,input_cell):
        out, out_hidden, out_cell = self.decoder(decoder_ids, encoder_all, encoder_last, input_hidden, input_cell)
        return out,out_hidden,out_cell

    def forward(self,encoder_ids,decoder_ids,input_hidden,input_cell):
        encoder_all,encoder_last = self.encoder_out(encoder_ids)
        out,last_hidden,last_cell = self.decoder(decoder_ids,encoder_all,encoder_last,input_hidden,input_cell)
        return out,last_hidden,last_cell

    def loss(self,out,label):
        cross_loss = self.cross_loss(out.view(out.size()[0],-1),label.contiguous().view(-1))
        return cross_loss

def bleu(pred,label,k):
    len_pred,len_label = len(pred),len(label)
    score = math.exp(min(0,1-len_pred/len_label))
    for n in range(1,k+1):
        num_matches,label_subs = 0,collections.defaultdict(int)
        for i in range(len_label-n+1):
            label_subs[''.join(label[i:i+n])] += 1
        for i in range(len_pred-n+1):
            if label_subs[''.join(pred[i:i+n])] > 0:
                num_matches += 1
                label_subs[''.join(pred[i:i+n])] -= 1
        score *= math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
    return score

def trainning(train_ids, train_labels,test_ids, test_labels,epochs,model,batchs):
    train_data = TensorDataset(train_ids,train_labels)
    train_iter = DataLoader(train_data, shuffle=True, batch_size=batchs)
    test_data = TensorDataset(test_ids, test_labels)
    test_iter = DataLoader(test_data, shuffle=True, batch_size=batchs)
    model.to(torch.device('cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()

    max_train_f1, max_test_f1, val_f1 = 0, 0, 0
    max_train_acc, max_test_acc, val_acc = 0, 0, 0
    max_train_report, max_test_report = None, None
    for epoch in range(epochs):
        for batch in train_iter:
            en_ids_t,de_input_labels_t = batch
            en_ids_t,de_input_labels_t = en_ids_t.cuda(),de_input_labels_t.cuda()
            # de_input_t,de_labels_t = de_input_labels_t[:,:7],de_input_labels_t[:,1:]
            # out = model(en_ids_t,de_input_t)
            # loss = model.loss(out,de_labels_t)
            loss = 0
            encoder_all, encoder_last = model.encoder_out(en_ids_t)
            input_hidden = torch.zeros(1, encoder_all.size()[0], 256).cuda()
            input_cell = torch.zeros(1, encoder_all.size()[0], 256).cuda()
            for i in range(7):
                out,input_hidden,input_cell = model.decoder_out(de_input_labels_t[:,i].view(-1,1),encoder_all,encoder_last,input_hidden,input_cell)
                loss += model.loss(out,de_input_labels_t[:,i+1])
            print(loss)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if epoch % 100 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                for batch in test_iter:
                    en_ids_t, de_input_labels_t = batch
                    en_ids_t, de_input_labels_t = en_ids_t.cuda(), de_input_labels_t.cuda()
                    bleu_list = []
                    encoder_all, encoder_last = model.encoder_out(en_ids_t)
                    for i in range(en_ids_t.size()[0]):
                        out_seq = []
                        de_x = de_input_labels_t[i, 0].view(1,1)
                        input_hidden = torch.zeros(1, 1, 256).cuda()
                        input_cell = torch.zeros(1, 1, 256).cuda()
                        for _ in range(9):
                            out, input_hidden, input_cell = model.decoder_out(de_x, encoder_all[i].unsqueeze(0), encoder_last[:,i].unsqueeze(0), input_hidden, input_cell)
                            de_x = out.argmax(dim=2)
                            pred = de_x[0,0].item()
                            out_seq.append(str(pred))
                            if pred == veccc['[eos]']:
                                break
                        for m in en_ids_t[i]:
                            print(id_to_word[m.item()], sep='', end='')
                        for m in out_seq:
                            print(id_to_word[int(m)],sep='',end='')
                        print('')
                        label = []
                        for j in range(7):
                            label.append(str(de_input_labels_t[i,j+1].item()))
                        bleu_out = bleu(out_seq,label,1)
                        bleu_list.append(bleu_out)

            model.train()
    return max_train_acc,max_test_acc,max_train_f1,max_test_f1,max_train_report,max_test_report


if __name__=='__main__':
    poetrys = read_peotry('./data/poetry/poetryFromTang.txt')
    #无mask版本
    five_sentences = []
    for i in poetrys:
        for j in i:
            if len(j) == 12:
                five_sentences.append(j)
    for i,_ in enumerate(five_sentences):
        sentence_one = []
        for j in five_sentences[i]:
            sentence_one.append(j)
        five_sentences[i] = sentence_one
    veccc = {}
    vect_len, max_len = 0, 14
    for i in five_sentences:
        for k in i:
            if k not in veccc.keys():
                veccc[k] = vect_len
                vect_len += 1
    for i,_ in enumerate(five_sentences):
        five_sentences[i] = five_sentences[i][:6]+['[eos]','[bos]']+five_sentences[i][6:]+['[eos]']
    veccc['[UNK]'] = vect_len
    veccc['[PAD]'] = vect_len + 1
    veccc['[eos]'] = vect_len + 2
    veccc['[bos]'] = vect_len + 3
    id_to_word = {}
    for i,e in veccc.items():
        id_to_word[e] = i
    glove_embed = get_word2vec_embed(veccc, vect_len)
    glove_embed = torch.FloatTensor(glove_embed)
    word_ids = id_generation(five_sentences, veccc,vect_len)
    word_ids = torch.tensor(word_ids).long()
    train,test = train_test_split(word_ids,train_size=0.7,random_state=123,shuffle=True)
    train_words, train_label = train[:,:7],train[:,7:]
    test_words, test_label = test[:,:7], test[:,7:]
    # ids,labels = word_ids[:,:7],word_ids[:,7:]

    model = Encoder_Decoder(vect_len,glove_embed)
    trainning(train_words,train_label,test_words,test_label, epochs=500, model=model, batchs=256)