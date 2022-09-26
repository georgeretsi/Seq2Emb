import numpy as np
import torch.cuda
from torch.autograd import Variable

import torch.nn.functional as F

import config

from models import SignSmooth

import editdistance

def test(epoch, test_loader, models, setting_dict, gpu_id, logger):
    lowercase = setting_dict["lowercase"]


    cnn = models['cnn']
    enc = models['enc']
    ctc_top = models['ctc_top']

    cnn.eval()
    ctc_top.eval()


    logger('Testing at epoch', epoch)

    tdecs = []
    transcrs = []
    for (img, transcr) in test_loader:
        #transcr = transcr[0].strip()
        #transcr = ''.join([tt for tt in transcr.lower() if tt in pclasses])
        #if len(transcr) == 0:
        #    continue

        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o = ctc_top(cnn(img))[0]
        tdec = o.argmax(2).permute(1, 0).cpu().numpy().reshape(img.size(0), -1)
        tdecs += [tdec]
        transcrs += list(transcr)
    tdecs = np.concatenate(tdecs)

    cer, wer = [], []
    for tdec, transcr in zip(tdecs, transcrs):
        transcr = transcr.strip()
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([config.icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()
        if lowercase:
            transcr, dec_transcr = config.reduced(transcr), config.reduced(dec_transcr)
            if '*' in transcr:
                continue
        #print(transcr)
        if len(transcr) == 0:
            continue
        #if len(transcr) <= 0:# and not transcr.isalnum():
        #    continue
        cer += [float(editdistance.eval(dec_transcr, transcr))/ len(transcr)]
        wer += [1 - float(transcr == dec_transcr)]

    logger('CER at epoch', epoch, ':', sum(cer) / len(cer))
    logger('WER at epoch', epoch, ':', sum(wer) / len(wer))

    cnn.train()
    ctc_top.train()

def test_dec(epoch, test_loader, models, setting_dict, gpu_id, logger):

    lowercase, binarize = setting_dict["lowercase"], setting_dict["binarize"]

    cnn = models['cnn']
    enc = models['enc']
    cdec = models['cdec']
    
    cnn.eval()
    enc.eval()
    cdec.eval()

    logger('Testing at epoch', epoch)

    cer, wer = [], []
    for (img, transcr) in test_loader:
        #transcr = transcr[0].strip()
        #transcr = ''.join([tt for tt in transcr.lower() if tt in pclasses])
        #if len(transcr) == 0:
        #    continue
        #transcr = transcr.strip()

        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            feat = cnn(img)
            decoder_input = torch.Tensor(img.size(0) * [len(config.classes) - 1]).long().to(gpu_id)
            enc_feat = enc(feat)

            decoder_hidden = enc_feat.view(1, img.size(0), -1)
            if binarize:
                fsign = SignSmooth()
                decoder_hidden =fsign(decoder_hidden)
            oo = np.zeros((img.size(0), 15))
            for di in range(15):
                decoder_output, decoder_hidden = cdec(decoder_input, decoder_hidden)
                decoder_input = decoder_output.argmax(dim=1).detach()
                oo[:, di] = decoder_input.cpu().numpy()

        for gt, o in zip(transcr, oo):
            if lowercase:
                gt = config.reduced(gt)
                if '*' in gt:
                    continue
                #gt = ''.join([c for c in gt.lower() if (c.isalnum() or c == '_' or c == ' ')])
            gt = gt.strip() #.split(' ')
            if len(gt) == 0: #and not gt.isalnum():
                continue
            #pred = next(cc for cc in ''.join([config.icdict[c] for c in o]).split(" ") if len(cc)>0)
            pred = ''.join([config.icdict[c] for c in o]).strip().split(" ")
            if len(pred) > 0:
                pred = pred[0]
            if lowercase:
                pred = config.reduced(pred)
                #pred = ''.join([c for c in pred.lower() if (c.isalnum() or c == '_' or c == ' ')])

            cer += [float(editdistance.eval(pred, gt)) / len(gt)]
            wer += [1 - float(gt == pred)]

    logger('CER at epoch', epoch, ':', sum(cer) / len(cer))
    logger('WER at epoch', epoch, ':', sum(wer) / len(wer))

    cnn.train()
    enc.train()
    cdec.train()

from auxilary_functions import average_precision

def test_kws(epoch, test_loader, models, setting_dict, gpu_id, logger, distance='euclidean', mode=2):
    binarize = setting_dict["binarize"]

    cnn = models['cnn']
    enc = models['enc']
    cenc = models['cenc']

    # mode -> 0: QbE, 1:QbS, 2:both

    cnn.eval()
    if mode==0 or mode==2:
        enc.eval()
    if mode == 1 or mode == 2:
        cenc.eval()

    logger('Testing KWS at epoch', epoch)
    tdecs = []
    transcrs = []
    for (img, transcr) in test_loader:

        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            feat = cnn(img)
            enc_feat = enc(feat)

            if binarize:
                enc_feat = enc_feat.sign()
            if distance == 'euclidean':
                tdec = enc_feat.cpu().numpy().squeeze()
            elif distance == 'cosine':
                tdec = F.normalize(enc_feat, dim=1).cpu().numpy().reshape(img.size(0), -1)

            #tdec = enc(feat).cpu().numpy().squeeze()
            #tdec = F.normalize(enc(feat).sign(), dim=1).cpu().numpy().squeeze()
        tdecs += [tdec]
        transcrs += [config.reduced(t.strip()) for t in list(transcr)]
    tdecs = np.concatenate(tdecs)

    uwords = np.unique(transcrs)
    udict = {w: i for i, w in enumerate(uwords)}
    lbls = np.asarray([udict[w] for w in transcrs])
    cnts = np.bincount(lbls)

    queries = [w for w in uwords if w not in config.stopwords and cnts[udict[w]] > 1 and len(w) > 1 and '*' not in w]

    if mode == 0 or mode == 2:
        qids = np.asarray([i for i,t in enumerate(transcrs) if t in queries])
        qdecs = tdecs[qids]

        if distance == 'euclidean':
            D = -2 * np.dot(qdecs, np.transpose(tdecs)) + \
                np.linalg.norm(tdecs, axis=1).reshape((1, -1))**2 + \
                np.linalg.norm(qdecs, axis=1).reshape((-1, 1))**2
        elif distance == 'cosine':
            D = -np.dot(qdecs, np.transpose(tdecs))

        # bce similarity
        #S = np.dot(qphocs_est, np.log(np.transpose(phocs_est))) + np.dot(1-qphocs_est, np.log(np.transpose(1-phocs_est)))
        Id = np.argsort(D, axis=1)
        while Id.max() > Id.shape[1]:
            Id = np.argsort(D, axis=1)

        map_qbe = 100 * np.mean([average_precision(lbls[Id[i]][1:] == lbls[qc]) for i, qc in enumerate(qids)])
        logger('QBE MAP at epoch', epoch, ':', map_qbe)

    if mode == 1 or mode == 2:
        target_sizes = [len(t) for t in queries]
        target_tensors = (len(config.classes) - 1) * torch.ones((len(queries), max(target_sizes)+2))
        for i, t in enumerate(queries):
            tt = torch.Tensor([config.cdict[c] for c in t])  # lower !!!!!!!!!!!!!
            target_tensors[i, 1:1+tt.size(0)] = tt
        target_tensors = target_tensors.long().to(gpu_id)
        target_tensors = Variable(target_tensors)

        #qdecs = F.normalize(cenc(target_tensors), dim=1).detach().cpu().numpy() # not necessary - sort to find MAP!
        qenc_feat = cenc(target_tensors)
        if binarize:
            qenc_feat = qenc_feat.sign()
        if distance == 'euclidean':
            qdecs = qenc_feat.detach().cpu().numpy()
        elif distance == 'cosine':
            qdecs = F.normalize(qenc_feat, dim=1).detach().cpu().numpy()


        #qdecs = F.normalize(cenc(target_tensors).sign(), dim=1).detach().cpu().numpy()

        if distance == 'euclidean':
            D = -2 * np.dot(qdecs, np.transpose(tdecs)) + \
                np.linalg.norm(tdecs, axis=1).reshape((1, -1))**2 + \
                np.linalg.norm(qdecs, axis=1).reshape((-1, 1))**2
        elif distance == 'cosine':
            D = -np.dot(qdecs, np.transpose(tdecs))

        Id = np.argsort(D, axis=1)
        while Id.max() > Id.shape[1]:
            Id = np.argsort(D, axis=1)

        map_qbs = 100 * np.mean([average_precision(np.asarray(transcrs)[Id[i]][0:] == q) for i, q in enumerate(queries)])
        logger('QBS MAP at epoch', epoch, ':', map_qbs)

    cnn.train()
    if mode == 0 or mode == 2:
        enc.train()
    if mode == 1 or mode == 2:
        cenc.train()


def test_qbs_kws_fa(epoch, test_loader, models, setting_dict, gpu_id, logger):

    binarize = setting_dict["binarize"]

    cnn = models['cnn']
    enc = models['enc']
    cdec = models['cdec']

    cnn.eval()
    enc.eval()
    cdec.eval()

    logger('Testing FA KWS at epoch', epoch)
    tdecs = []
    transcrs = []
    for (img, transcr) in test_loader:
        #transcr = transcr[0].strip()
        #transcr = ''.join([tt for tt in transcr.lower() if tt in pclasses])
        #if len(transcr) == 0:
        #    continue

        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            feat = cnn(img)
            tdec = enc(feat).detach()

            #tdec = enc(feat).cpu().numpy().squeeze()
            #tdec = F.normalize(enc(feat).sign(), dim=1).cpu().numpy().squeeze()
        tdecs += [tdec]
        transcrs += [config.reduced(t.strip()) for t in list(transcr)]
    tdecs = torch.cat(tdecs)

    uwords = np.unique(transcrs)
    udict = {w: i for i, w in enumerate(uwords)}
    lbls = np.asarray([udict[w] for w in transcrs])
    cnts = np.bincount(lbls)
    queries = [w for w in uwords if w not in config.stopwords and cnts[udict[w]] > 1 and len(w) > 1 and '*' not in w]

    D = np.zeros((len(queries), tdecs.size(0)))
    for i, query in enumerate(queries):
        query = ' ' + query + ' '
        qinput = torch.Tensor([config.cdict[c] for c in query]).long().to(gpu_id)
        for j, tdec in enumerate(tdecs):
            decoder_hidden = tdec.clone().view(1, 1, -1)
            if binarize:
                fsign = SignSmooth()
                decoder_hidden =fsign(decoder_hidden)
            loss = 0
            for k in range(len(query)-1):
                decoder_input = qinput[k]
                decoder_output, decoder_hidden = cdec(decoder_input.view(1, -1), decoder_hidden)
                loss += F.nll_loss(decoder_output, qinput[k+1].view(-1))
            D[i,j] = loss.item() / len(query)



    #D = np.dot(qdecs, np.transpose(tdecs))
    Id = np.argsort(D, axis=1)
    while Id.max() > Id.shape[1]:
        Id = np.argsort(-D, axis=1)

    map_qbs = 100 * np.mean([average_precision(np.asarray(transcrs)[Id[i]][0:] == q) for i, q in enumerate(queries)])
    logger('QBS MAP at epoch', epoch, ':', map_qbs)

    cnn.train()
    enc.train()
    cdec.train()
