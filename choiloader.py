from __future__ import print_function
from pathlib2 import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from text_manipulation import split_sentences, word_model, extract_sentence_words
import utils
import math


logger = utils.setup_logger(__name__, 'train.log')


def get_choi_files(path):
    all_objects = Path(path).glob('**/*.ref')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path

def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*')
    cache_file_path = get_cache_path(wiki_folder)

    with cache_file_path.open('w+') as f:
        for file in files:
            f.write(str(file) + u'\n')

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    batch_targets_idx = []
    batched_sent_bert_vec = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) /2))
    after_sentence_count = window_size - before_sentence_count - 1

    for data, targets, path, sent_bert_vec in batch:
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window)))
            tensored_targets = torch.zeros(len(data)).long()
            tensored_targets[torch.LongTensor(targets)] = 1
            tensored_targets = tensored_targets[:-1]
            tensored_sent_bert_vec = torch.FloatTensor(sent_bert_vec)

            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            batched_sent_bert_vec.append(tensored_sent_bert_vec)
            paths.append(path)
            batch_targets_idx.append(targets)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue
    #batched_sent_bert_vec = bc.encode(batched_sentence_list)
    #batched_sent_bert_vec = torch.FloatTensor(batched_sent_bert_vec)

    return batched_data, batched_targets, paths, batched_sent_bert_vec, batch_targets_idx

def clean_paragraph(paragraph):
    cleaned_paragraph= paragraph.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip('\n')
    return cleaned_paragraph

def read_choi_file(path, word2vec, sent_bert_vec, train, return_w2v_tensors = True,manifesto=False):
    seperator = '========' if manifesto else '=========='
    with Path(path).open('r') as f:
        raw_text = f.read()
    paragraphs = [clean_paragraph(p) for p in raw_text.strip().split(seperator)
                  if len(p) > 5 and p != "\n"]
    if train:
        random.shuffle(paragraphs)

    targets = []
    new_text = []
    text = []
    lastparagraphsentenceidx = 0

    for paragraph in paragraphs:
        if manifesto:
            sentences = split_sentences(paragraph,0)
        else:
            sentences = [s for s in paragraph.split('\n') if len(s.split()) > 0]

        if sentences:
            sentences_count =0
            # This is the number of sentences in the paragraph and where we need to split.
            for sentence in sentences:
                words, sentence_str = extract_sentence_words(sentence)
                if (len(words) == 0):
                    continue
                sentences_count +=1
                if return_w2v_tensors:
                    text.append(words)
                    new_text.append([word_model(w, word2vec) for w in words])
                else:
                    text.append(words)
                    new_text.append(words)

            lastparagraphsentenceidx += sentences_count
            targets.append(lastparagraphsentenceidx - 1)

    return new_text, targets, path, sent_bert_vec


# Returns a list of batch_size that contains a list of sentences, where each word is encoded using word2vec.
class ChoiDataset(Dataset):
    def __init__(self, root, word2vec, sent_bert, train=False, folder=False,manifesto=False, folders_paths = None):
        self.manifesto = manifesto
        if folders_paths is not None:
            self.textfiles = []
            for f in folders_paths:
                self.textfiles.extend(list(f.glob('*.ref')))
        elif (folder):
            self.textfiles = get_choi_files(root)
        else:
            root_path = Path(root)
            cache_path = get_cache_path(root_path)
            if not cache_path.exists():
                cache_wiki_filenames(root_path)
            self.textfiles = cache_path.read_text().splitlines()
            #self.textfiles = list(Path(root).glob('**/*.ref'))

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.sent_bert = sent_bert

    def __getitem__(self, index):
        path = self.textfiles[index]
        sent_bert_vec = self.sent_bert[index]

        return read_choi_file(path, self.word2vec, sent_bert_vec, self.train,manifesto=self.manifesto)

    def __len__(self):
        return len(self.textfiles)
