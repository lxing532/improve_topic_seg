import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy
import numpy as np
from termcolor import colored

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()

def read_sent_bert(topic_vec, path):
    tmp = []
    for line in open(path):
        if line.strip() == '':
            topic_vec.append(tmp)
            tmp = []
        else:
            l = line.strip().split()
            tmp.append([float(s) for s in l])
    return topic_vec

def supervised_cross_entropy(pred, sims, soft_targets, target_coh_var):
    criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()
    loss_pred = criterion(pred, soft_targets)
    loss_sims = bce_criterion(sims, target_coh_var.unsqueeze(1).type(torch.cuda.FloatTensor))
    loss = 0.8*loss_pred +0.2*loss_sims
    return loss

class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold


def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, target, paths, sent_bert_vec, target_idx) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break

                pbar.update()
                model.zero_grad()
                output, sims = model(data, sent_bert_vec, target_idx)
                
                target_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                # generate gold label for coherence scores...
                target_list = target_var.data.cpu().numpy()
                target_coh_list = []
                for t in target_list:
                    if t == 0:
                        target_coh_list.append(torch.LongTensor([1]))
                    else:
                        target_coh_list.append(torch.LongTensor([0]))
                target_coh_var = Variable(maybe_cuda(torch.cat(target_coh_list, 0), args.cuda), requires_grad=False)

                loss = supervised_cross_entropy(output, sims, target_var, target_coh_var)
                #loss = model.criterion(output, target_var)
                
                #sim = sim_.data.cpu()
                #total_sim += sim
                #concate_target = torch.cat(target, 0)

                #total_1.append(sum(sim.squeeze(1)[concate_target == 1])/len(sim.squeeze(1)[concate_target == 1]))
                #total_0.append(sum(sim.squeeze(1)[concate_target == 0])/len(sim.squeeze(1)[concate_target == 0]))
                '''
                new_target = torch.zeros(concate_target.size()[0], 2)
                new_target[range(new_target.shape[0]), concate_target]=1
                target_var = Variable(maybe_cuda(new_target, args.cuda), requires_grad=False)
                target_var = target_var.type_as(output)
                pos_weight = maybe_cuda(torch.FloatTensor([0.1, 1.0]))

                loss = F.binary_cross_entropy_with_logits(output, target_var, reduction='sum', pos_weight=pos_weight) + 10*sim_
                '''

                loss.backward()

                optimizer.step()
                total_loss += loss.item()
                # logger.debug('Batch %s - Train error %7.4f', i, loss.data[0])
                pbar.set_description('Training, loss={:.4}'.format(loss.item()))
            # except Exception as e:
                # logger.info('Exception "%s" in batch %s', e, i)
                # logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
                # pass
    #print('The similarity between the segs: ', total_sim/len(dataset))

    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    #total_sim = float(0)
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths, sent_bert_vec, target_idx) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output, _ = model(data, sent_bert_vec, target_idx)
                #sim = sim.data.cpu()
                #total_sim += sim
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)

                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), target)

            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        #print('The similarity between the segs: ', total_sim/len(dataset))

        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk, threshold


def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        acc_1 = accuracy.Accuracy()
        acc_2 = accuracy.Accuracy()
        acc_3 = accuracy.Accuracy()
        acc_4 = accuracy.Accuracy()
        acc_5 = accuracy.Accuracy()
        for i, (data, target, paths, sent_bert_vec, target_idx) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output, _ = model(data, sent_bert_vec, target_idx)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                current_idx = 0

                for k, t in enumerate(target):
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)

                    #output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    output_1 = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > 0.1)
                    output_2 = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > 0.2)
                    output_3 = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > 0.3)
                    output_4 = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > 0.4)
                    output_5 = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > 0.5)
                    h_1 = np.append(output_1, [1])
                    h_2 = np.append(output_2, [1])
                    h_3 = np.append(output_3, [1])
                    h_4 = np.append(output_4, [1])
                    h_5 = np.append(output_5, [1])
                    tt = np.append(t, [1])

                    t_pred = output_softmax.data.cpu().numpy()[current_idx: to_idx, :]
                    t_gold = t

                    acc_1.update(h_1, tt)
                    acc_2.update(h_2, tt)
                    acc_3.update(h_3, tt)
                    acc_4.update(h_4, tt)
                    acc_5.update(h_5, tt)

                    current_idx = to_idx

                    # acc.update(output_softmax.data.cpu().numpy(), target)

            #
            # except Exception as e:
            #     # logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)

        epoch_pk_1, epoch_windiff_1 = acc_1.calc_accuracy()
        epoch_pk_2, epoch_windiff_2 = acc_2.calc_accuracy()
        epoch_pk_3, epoch_windiff_3 = acc_3.calc_accuracy()
        epoch_pk_4, epoch_windiff_4 = acc_4.calc_accuracy()
        epoch_pk_5, epoch_windiff_5 = acc_5.calc_accuracy()

        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk_1,
                                                                                                          epoch_windiff_1,
                                                                                                          preds_stats.get_f1()))
        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk_2,
                                                                                                          epoch_windiff_2,
                                                                                                          preds_stats.get_f1()))
        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk_3,
                                                                                                          epoch_windiff_3,
                                                                                                          preds_stats.get_f1()))
        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk_4,
                                                                                                          epoch_windiff_4,
                                                                                                          preds_stats.get_f1()))
        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk_5,
                                                                                                          epoch_windiff_5,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk_1


def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)

    configure(os.path.join('runs', args.expname))

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    # read the pretrained BERT sentence embeddings...
    
    bound1 = 14900; bound2 = 2135; bound3 = 50; bound4 = 100; bound5 = 117; bound6 = 227;
    bert_vec = []

    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/bert_emb_train_cleaned.txt')
    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/bert_emb_dev_cleaned.txt')
    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/test1_data_emb_cleaned.txt')
    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/bert_emb_test_2.txt')
    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/bert_emb_test_3.txt')
    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/bert_emb_test_4.txt')
    bert_vec = read_sent_bert(bert_vec, '/ubc/cs/research/nlp/Linzi/seg/bert/bert_emb_test_wiki.txt')
    train_bert = bert_vec[0:bound1]
    dev_bert = bert_vec[bound1:(bound1+bound2)]
    test_bert = bert_vec[(bound1+bound2):(bound1+bound2+bound3)]
    test2_bert = bert_vec[(bound1+bound2+bound3):(bound1+bound2+bound3+bound4)]
    test3_bert = bert_vec[(bound1+bound2+bound3+bound4):(bound1+bound2+bound3+bound4+bound5)]
    test4_bert = bert_vec[(bound1+bound2+bound3+bound4+bound5):(bound1+bound2+bound3+bound4+bound5+bound6)]
    test5_bert = bert_vec[(bound1+bound2+bound3+bound4+bound5+bound6):]

    
    if not args.infer:
        if args.wiki:
            dataset_path = Path(utils.config['wikidataset'])
            train_dataset = WikipediaDataSet(dataset_path / 'train', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=train_bert)
            dev_dataset = WikipediaDataSet(dataset_path / 'dev', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=dev_bert)
            test_dataset = WikipediaDataSet(dataset_path / 'test', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=test_bert)
            test_dataset_2 = WikipediaDataSet(dataset_path / 'test_cities', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=test2_bert)
            test_dataset_3 = WikipediaDataSet(dataset_path / 'test_elements', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=test3_bert)
            test_dataset_4 = WikipediaDataSet(dataset_path / 'test_clinical', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=test4_bert)
            test_dataset_5 = WikipediaDataSet(dataset_path / 'test_wiki', word2vec=word2vec, high_granularity=args.high_granularity, sent_bert=test5_bert)

        else:
            dataset_path = Path(utils.config['choidataset'])
            train_dataset = ChoiDataset(dataset_path / 'train', word2vec)
            dev_dataset = ChoiDataset(dataset_path / 'dev', word2vec)
            test_dataset = ChoiDataset(dataset_path / 'test', word2vec)



        train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True,
                              num_workers=args.num_workers)
        dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                            num_workers=args.num_workers)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        test_dl_2 = DataLoader(test_dataset_2, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        test_dl_3 = DataLoader(test_dataset_3, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        test_dl_4 = DataLoader(test_dataset_4, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        test_dl_5 = DataLoader(test_dataset_5, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)

    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set

    if args.model:
        model = import_model(args.model)
    elif args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)

    model.train()
    model = maybe_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if not args.infer:
        best_val_pk = 1.0
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)

            val_pk, threshold = validate(model, args, j, dev_dl, logger)
            test_pk = test(model, args, j, test_dl, logger, threshold)
            test_pk2 = test(model, args, j, test_dl_2, logger, threshold)
            test_pk3 = test(model, args, j, test_dl_3, logger, threshold)
            test_pk4 = test(model, args, j, test_dl_4, logger, threshold)
            test_pk5 = test(model, args, j, test_dl_5, logger, threshold)
            if val_pk < best_val_pk:
                logger.debug(
                    colored(
                        'Current best model from epoch {} with p_k {} and threshold {}'.format(j, test_pk, threshold),
                        'green'))
                best_val_pk = val_pk

                with (checkpoint_path / 'best_model_transformer.t7'.format(j)).open('wb') as f:
                    torch.save(model, f)


    else:
        test_dataset = WikipediaDataSet(args.infer, word2vec=word2vec,
                                        high_granularity=args.high_granularity)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)
        print(test(model, args, 0, test_dl, logger, 0.4))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='inference_dir', type=str)

    main(parser.parse_args())

