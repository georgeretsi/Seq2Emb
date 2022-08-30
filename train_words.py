import argparse
import logging
import os

import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import config

from models import CNN, CTCtop, Enc, DecoderChar, EncoderChar, SignSmooth

from auxilary_functions import affine_transformation
from utils.iam_loader import IAMLoader

import torch.nn.functional as F

import test_funcs
import pickle



# losses
ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt)
dec_criterion = nn.NLLLoss(reduce=False)


report_txt_name = 'test.txt'


def logger(*args, **kwargs):
    
    print(*args, **kwargs)
    with open(report_txt_name,'a') as file:
        print(*args, **kwargs, file=file)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seq2seq_backbone(enc_feats, transcr, models, setting_dict, gpu_id):

    fsign = SignSmooth()

    path_autoencoder = setting_dict['path_autoencoder']
    binarize = setting_dict['binarize']

    cenc = models['cenc']
    cdec = models['cdec']

    target_sizes = [len(t) for t in transcr]
    Nt = len(transcr)
    Nc = len(config.classes)
    target_tensors = (Nc - 1) * torch.ones((Nt, max(target_sizes)))
    for i, t in enumerate(transcr):
        tt = torch.Tensor([config.cdict[c] for c in t])
        target_tensors[i, :tt.size(0)] = tt
    target_tensors = target_tensors.long().to(gpu_id)
    target_tensors = Variable(target_tensors)

    char_cnt = sum(target_sizes)

    #decoder_hidden = enc_feats.view(1, len(transcr), -1)

    if enc_feats is None:
        decoder_hidden = cenc(target_tensors).view(1, Nt, -1)
        dist_loss = 0.0 # or dissimilarity
    else:
        if path_autoencoder:
            t_feats = cenc(target_tensors)

            dist_loss = 0.1 * F.mse_loss(t_feats, enc_feats).cpu() + \
                        1.0 * (1 - F.cosine_similarity(t_feats, enc_feats)).mean().cpu()

            # when autoencoder exists, half the time use the target
            if np.random.uniform(0.0, 1.0) < .5:
                decoder_hidden = t_feats.view(1, Nt, -1)
            else:
                decoder_hidden = enc_feats.view(1, Nt, -1)

        else:
            decoder_hidden = enc_feats.view(1, Nt, -1)
            dist_loss = 0.0

    # init seq 2 seq approach
    decoder_input = target_tensors[:, 0]

    if binarize:
        decoder_hidden = fsign(decoder_hidden)

    eloss = 0.0
    target_length = target_tensors.size(1)
    teacher_forcing = np.random.uniform(0.0, 1.0) < .5
    for di in range(1, target_length):
        decoder_output, decoder_hidden = cdec(decoder_input, decoder_hidden)
        if teacher_forcing:
            mask = torch.tensor([t > di - 0 for t in target_sizes]).float().cuda(gpu_id)
            eloss += sum(dec_criterion(decoder_output, target_tensors[:, di]) * mask)
            decoder_input = target_tensors[:, di]
        else:
            mask = torch.tensor([t > di - 3 for t in target_sizes]).float().cuda(gpu_id)
            eloss += sum(dec_criterion(decoder_output, target_tensors[:, di]) * mask)
            decoder_input = decoder_output.argmax(1).detach()

    return 1.0 * dist_loss + eloss.cpu() / char_cnt


def train(epoch, train_loader, models, optimizers, setting_dict, gpu_id, logger, external_lm=None):

    loss_report = []

    fsign = SignSmooth()

    optimizer, coptimizer = optimizers['main'], optimizers['extra']

    lowercase = setting_dict['lowercase']
    path_ctc = setting_dict['path_ctc']
    path_s2s = setting_dict['path_s2s']
    path_autoencoder = setting_dict['path_autoencoder']
    train_external_lm = setting_dict['train_external_lm']
    binarize = setting_dict['binarize']


    cnn = models['cnn']
    ctc_top = models['ctc_top']
    enc = models['enc']
    cenc = models['cenc']
    cdec = models['cdec']

    if external_lm is not None:
        words = external_lm[0]
        words_prob = external_lm[1]

    cnn.train()

    if path_ctc:
        ctc_top.train()
    if path_s2s:
        enc.train()
        cdec.train()
    if path_autoencoder:
        cenc.train()

    closs = []
    for iter_idx, (img, transcr) in enumerate(train_loader):
        
        optimizer.zero_grad()
        if coptimizer is not None:
            coptimizer.zero_grad()

        img = Variable(img.cuda(gpu_id))

        feats = cnn(img)

        if lowercase:
            transcr = [config.reduced(t) for t in transcr]

        labels = torch.IntTensor([config.cdict[c] for c in ''.join(transcr)])  # .to(img.device)
        label_lens = torch.IntTensor([len(t) for t in transcr])  # .to(img.device)


        if path_ctc:
            output = ctc_top(feats)[0]
        if path_s2s:
            enc_feats = enc(feats)

        if path_ctc:
            act_lens = torch.IntTensor(img.size(0) * [output.size(0)])
            loss_val = ctc_loss(output.cpu(), labels, act_lens, label_lens)/config.batch_size
        else:
            loss_val = 0

        if path_s2s:
            #print(transcr)
            eloss = seq2seq_backbone(enc_feats, transcr, models, setting_dict, gpu_id)
        else:
            eloss = 0

        a, b = 1.0, 10.0
        loss_val = a * loss_val + b * eloss

        closs += [loss_val.item()]

        loss_val.backward()

        optimizer.step()
        if coptimizer is not None:
            coptimizer.step()

        # also path_s2s and path_autoencoder
        if train_external_lm:
            coptimizer.zero_grad()

            randK = 512
            transcr = np.random.choice(words, randK, p=words_prob)
            if lowercase:
                transcr = [config.reduced(t) for t in transcr]


            loss_val = 1.0 * seq2seq_backbone(None,transcr, models, setting_dict, gpu_id)
            loss_val.backward()

            coptimizer.step()


        # mean runing errors??
        if iter_idx % config.display == config.display-1:
            logger('Epoch', epoch, 'Iteration' , iter_idx+1, ':', sum(closs)/len(closs))

            loss_report += [sum(closs)/len(closs)]

            closs = []

            cnn.eval()

            tst_img, tst_transcr = train_loader.dataset.__getitem__(np.random.randint(train_loader.dataset.__len__()))
            if lowercase:
                tst_transcr = config.reduced(tst_transcr)

            with torch.no_grad():
                tst_feat = cnn(Variable(tst_img.cuda(gpu_id)).unsqueeze(0))

            cnn.train()

            print('transcr:: ' + tst_transcr)

            if path_ctc:
                ctc_top.eval()
                tst_o = ctc_top(tst_feat)[0]
                tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
                print('ctc-dec:: ' + ''.join([config.icdict[t] for t in tt]).replace('_', ''))
                ctc_top.train()

            if path_s2s:
                enc.eval()
                enc_feat = enc(tst_feat)
                enc.train()

                cdec.eval()
                decoder_input = torch.Tensor([len(config.classes)-1]).long().to(gpu_id)
                decoder_hidden = enc_feat.view(1, 1, -1)
                if binarize:
                    decoder_hidden =fsign(decoder_hidden)
                oo = []
                for di in range(15):
                    decoder_output, decoder_hidden = cdec(decoder_input.view(1, -1), decoder_hidden)
                    oo += [config.icdict[decoder_output.argmax().item()]]
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()
                print('s2s-dec:: ' + ''.join(oo))
                cdec.train()

                if path_autoencoder:
                    cenc.eval()
                    cdec.eval()
                    decoder_input = torch.Tensor([len(config.classes)-1]).long().to(gpu_id)
                    decoder_hidden = cenc(torch.Tensor([config.cdict[c] for c in tst_transcr]).view(1, -1).long().to(gpu_id)).view(1, 1, -1)
                    if binarize:
                        decoder_hidden =fsign(decoder_hidden)
                    oo = []
                    for di in range(15):
                        decoder_output, decoder_hidden = cdec(decoder_input.view(1, -1), decoder_hidden)
                        oo += [config.icdict[decoder_output.argmax().item()]]
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach()
                    print('autoenc:: ' + ''.join(oo))
                    cenc.train()
                    cdec.train()
    return loss_report

def evaluate_setting(setting_dict, dataset, gpu_id, report_name='temp', epochs=None):

    dataset_name, dataset_path = dataset[0], dataset[1]

    if epochs is not None:
        max_epochs = epochs
    else:
        max_epochs = config.max_epochs

    report_file = open(report_name + '_' + dataset_name + '.pkl', 'wb')
    pickle.dump(setting_dict, report_file)

    lowercase = setting_dict['lowercase']
    path_ctc = setting_dict['path_ctc']
    path_s2s = setting_dict['path_s2s']
    path_autoencoder = setting_dict['path_autoencoder']
    train_external_lm, start_external_lm = setting_dict['train_external_lm'], setting_dict['start_external_lm']
    binarize, start_binarize = setting_dict['binarize'], setting_dict['binarize']
    feat_size = setting_dict['feat_size']

    start_external_lm = int(max_epochs * start_external_lm)
    start_binarize = int(max_epochs * start_binarize)

    name_code = dataset_name+'_f' + str(feat_size) + '_v'+''.join([str(int(s)) for s in [lowercase, path_ctc, path_s2s, path_autoencoder, train_external_lm, binarize]])

    #logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    #                datefmt='%Y-%m-%d %H:%M:%S',
    #                level=logging.INFO)
    #logger = logging.getLogger('HTR-Experiment::train')
    
    global report_txt_name
    report_txt_name = report_name + '_' + dataset_name +'.txt'
    if os.path.exists(report_txt_name):
        os.remove(report_txt_name)
    
    logger('Experiment Parameters:')
    logger(setting_dict)


    aug_transforms =[lambda x: affine_transformation(x, s=.2)]

    if dataset_name == 'IAM':
        myDataset = IAMLoader

        train_set = myDataset(dataset_path, 'train', 'word', fixed_size=config.fixed_size, transforms=aug_transforms)
        test_set = myDataset(dataset_path, 'test', 'word', fixed_size=config.fixed_size, transforms=None)

        stopwords_path = './utils/iam-stopwords'
        stopwords = []
        for line in open(stopwords_path):
            stopwords.append(line.strip().split(','))
        config.stopwords = stopwords[0]
    else:
        print('different loader required!')
        raise NotImplementedError

    batch_size = config.batch_size
    if lowercase:
        classes = ''.join([c.lower()  for c in train_set.character_classes if c.isalnum()])
        classes = ''.join(list(sorted(set(list(classes)))))
        config.classes = '_*' + classes + ' '
        print('character classes:')
        print(config.classes)
    config.cdict = {c:i for i,c in enumerate(config.classes)}
    config.icdict = {i:c for i,c in enumerate(config.classes)}

    

    # augmentation using data sampler
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    if train_external_lm:
        uwords, wcount = np.unique([' ' + w[1] + ' ' for w in train_set.data], return_counts=True)
        word_unigrams = np.loadtxt('./utils/word_unigrams.txt', dtype=str)
        #if lowercase:
        #    words = [' ' + config.reduced(t[0]) + ' ' for t in unigrams] + list(uwords)
        #else:
        words = [' ' + t[0] + ' ' for t in word_unigrams] + list(uwords)

        words_prob = np.concatenate([np.asarray([max(float(t[1]), 1e-4) for t in word_unigrams]),np.clip(1.0*wcount/sum(wcount), 1e-4, None)])
        words_prob = words_prob / sum(words_prob)
        external_lm = (words, words_prob)
    else:
        external_lm = None

    # load CNN

    #torch.cuda.empty_cache()
    cnn = CNN(config.cnn_cfg)
    cnn.cuda(gpu_id)
    print("cnn size:" + str(count_parameters(cnn)))

    if path_ctc:
        ctc_top = CTCtop(config.cnn_cfg[-1][-1], config.cnn_top, len(config.classes))
        print("ctc head size:" + str(count_parameters(cnn)))
        ctc_top.cuda(gpu_id)
    else:
        ctc_top = None

    if path_s2s:
        enc = Enc(config.rnn_cfg, 256, feat_size)
        enc.cuda(gpu_id)
        print("enc size:" + str(count_parameters(enc)))

        cdec = DecoderChar(feat_size, len(config.classes)).cuda(gpu_id)
        cdec.cuda(gpu_id)
        print("cdec size:" + str(count_parameters(cdec)))
    else:
        enc = None
        cdec = None

    if path_autoencoder:
        cenc = EncoderChar(len(config.classes), 256, feat_size).cuda(gpu_id)
        cenc.cuda(gpu_id)
        print("cenc size:" + str(count_parameters(cenc)))
    else:
        cenc = None

    models = {
        "cnn": cnn,
        "ctc_top": ctc_top,
        "enc": enc,
        "cenc": cenc,
        "cdec": cdec,
    }

    parameters = list(cnn.parameters())
    if path_ctc:
        parameters += list(ctc_top.parameters())
        if path_s2s:
            parameters += list(enc.parameters())

    cparameters = []
    if path_s2s:
        cparameters += list(cdec.parameters())
        if path_autoencoder:
            cparameters += list(cenc.parameters())

    optimizer = torch.optim.AdamW(parameters, config.nlr, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.restart_epochs)

    if len(cparameters) > 0:
        coptimizer = torch.optim.AdamW(cparameters, .1 * config.nlr, weight_decay=0.00005)
        cscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(coptimizer, config.restart_epochs)
    else:
        coptimizer = None

    optimizers = {'main': optimizer, 'extra': coptimizer}

    
    loss_report = []


    logger('Training:')
    print("binarize: " + str(setting_dict["binarize"]))

    epoch = 0
    for epoch in range(1, max_epochs + 1):

        setting_dict['binarize'] = binarize and (epoch >= start_binarize)
        setting_dict['train_external_lm'] = train_external_lm and (epoch >= start_external_lm)

        tmp_loss = train(epoch, train_loader, models, optimizers, setting_dict, gpu_id, logger, external_lm=external_lm)
        loss_report += tmp_loss

        scheduler.step()
        if len(cparameters) > 0:
            cscheduler.step()

        
        if epoch % 5 == 0:
            if path_ctc:
                test_funcs.test(epoch, test_loader, models, setting_dict, gpu_id, logger)
            if path_s2s:
                test_funcs.test_dec(epoch,test_loader, models, setting_dict, gpu_id, logger)
                if not path_autoencoder:
                    test_funcs.test_kws(epoch,test_loader, models, setting_dict, gpu_id, logger, distance='cosine', mode=0)
            if path_autoencoder:
                test_funcs.test_kws(epoch,test_loader, models, setting_dict, gpu_id, logger, distance='cosine')

        #if epoch % 20 == 0:
        #    logger('Saving net after', epoch, 'epochs')

        if epoch % config.restart_epochs == 0:
            parameters = list(cnn.parameters())

            if path_ctc:
                parameters += list(ctc_top.parameters())
                if path_s2s:
                    parameters += list(enc.parameters())

            cparameters = []
            if path_s2s:
                cparameters += list(cdec.parameters())
                if path_autoencoder:
                    cparameters += list(cenc.parameters())

            optimizer = torch.optim.AdamW(parameters, config.nlr, weight_decay=0.00005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.restart_epochs)

            if coptimizer is not None:
                coptimizer = torch.optim.AdamW(cparameters, .1 *config.nlr, weight_decay=0.00005)
                cscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(coptimizer, config.restart_epochs)

            optimizers = {'main': optimizer, 'extra': coptimizer}

    logger("------Final Testing------")
    if path_ctc:
        logger("###CTC-PATH###")
        test_funcs.test(epoch, test_loader, models, setting_dict, gpu_id, logger)
    if path_s2s:
        logger("###S2S-PATH###")
        test_funcs.test_dec(epoch,test_loader, models, setting_dict, gpu_id, logger)
        if not path_autoencoder:
            logger("KWS without autoencoder:")
            test_funcs.test_kws(epoch,test_loader, models, setting_dict, gpu_id, logger, distance='cosine', mode=0)
    if path_autoencoder:
        logger("KWS:")
        test_funcs.test_kws(epoch,test_loader, models, setting_dict, gpu_id, logger, distance='cosine')

    pickle.dump(loss_report, report_file)
    report_file.close()
    logging.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                        help='The ID of the GPU to use.')

    parser.add_argument('--lowercase', type=bool, default=True)
    parser.add_argument('--path_ctc', type=bool, default=True)
    parser.add_argument('--path_s2s', type=bool, default=True)
    parser.add_argument('--path_autoencoder', type=bool, default=True)
    parser.add_argument('--train_external_lm', type=bool, default=False)
    parser.add_argument('--start_external_lm', type=float, default=.0)
    parser.add_argument('--binarize', type=bool, default=False)
    parser.add_argument('--start_binarize', type=float, default=.0)
    parser.add_argument('--feat_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='IAM')
    parser.add_argument('--dataset_path', type=str, default='../IAM')
    #parser.add_argument('--load_code', type=str, default=None)

    args = parser.parse_args()


    gpu_id = args.gpu_id
    dataset = (args.dataset, args.dataset_path)

    setting = {
        'lowercase': args.lowercase,
        'path_ctc': args.path_ctc,
        'path_s2s': args.path_s2s,
        'path_autoencoder': args.path_autoencoder,
        'train_external_lm': args.train_external_lm,
        'start_external_lm': args.start_external_lm,
        'binarize': args.binarize,
        'start_binarize': args.start_binarize,
        'feat_size': args.feat_size
    }


    evaluate_setting(setting, dataset, gpu_id)
