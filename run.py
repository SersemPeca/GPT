#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *

startToken = '<S>'

endToken = '</S>'

unkToken = '<UNK>'
unkTokenIdx = 2

padToken = '<PAD>'

transToken = '<TRANS>'
transTokenIdx = 4

def perplexity(nmt, test, batch_size):
    testSize = len(test)
    H = 0.
    c = 0
    for b in range(0,testSize,batch_size):
        batch = test[b:min(b+batch_size, testSize)]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * nmt(batch)
    return math.exp(H/c)


def buildModel(word2ind):
    return model.LanguageModel(
        vocab_size=len(word2ind),
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    trainCorpus, devCorpus, word2ind = utils.prepareData(source_file_name, target_file_name, source_dev_file_name, target_dev_file_name, startToken, endToken, unkToken, padToken, transToken)
    trainCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in trainCorpus ]
    devCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in devCorpus ]
    pickle.dump((trainCorpus, devCorpus), open(corpus_file_name, 'wb'))
    pickle.dump(word2ind, open(words_file_name, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    (trainCorpus,devCorpus) = pickle.load(open(corpus_file_name, 'rb'))
    word2ind = pickle.load(open(words_file_name, 'rb'))

    nmt = buildModel(word2ind)
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

    if sys.argv[1] == 'extratrain':
        nmt.load(model_file_name)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(model_file_name + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(max_epochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), batch_size):
            #############################################################################
            ### Може да се наложи да се променя скоростта на спускане learning_rate в зависимост от итерацията
            #############################################################################
            iter += 1
            batch = [ trainCorpus[i] for i in idx[b:min(b+batch_size, len(idx))] ]
            
            words += sum( len(s)-1 for s in batch )
            H = nmt(batch)
            optimizer.zero_grad()
            H.backward()
            torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',max_epochs,", Batch:",b//batch_size+1, '/', len(idx) // batch_size+1, ", loss: ",H.item(), "words/sec:",words / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                words = 0
                
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpus, batch_size)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(model_file_name)
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), model_file_name + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpus, batch_size)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(model_file_name)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), model_file_name + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    word2ind = pickle.load(open(words_file_name, 'rb'))
    
    nmt = buildModel(word2ind)
    nmt.load(model_file_name)
    
    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    test = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceTest,targetTest)]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, test, batch_size))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    word2ind = pickle.load(open(words_file_name, 'rb'))
    words = list(word2ind)

    sourceTest = utils.readCorpus(sys.argv[2])
    test = [ [startToken] + s + [transToken] for s in sourceTest ]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt = buildModel(word2ind)
    nmt.load(model_file_name)

    nmt.eval()
    file = open(sys.argv[3],'w')
    pb = utils.progressBar()
    pb.start(len(test))
    for s in test:
        r=nmt.generate(s)
        st = r.index(transTokenIdx)
        result = [words[i] for i in r[st+1:-1]]
        file.write(' '.join(result)+"\n")
        pb.tick()
    pb.stop()

if len(sys.argv)>2 and sys.argv[1] == 'generate':
    word2ind = pickle.load(open(words_file_name, 'rb'))
    words = list(word2ind)

    test = sys.argv[2].split()
    test = [word2ind.get(w,unkTokenIdx) for w in test]

    nmt = buildModel(word2ind)
    nmt.load(model_file_name)

    nmt.eval()
    r=nmt.generate(test)
    result = [words[i] for i in r]
    print(' '.join(result)+"\n")

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
