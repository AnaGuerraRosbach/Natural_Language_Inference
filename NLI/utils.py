import torch
import codecs
import re
import json
import en_vectors_web_lg
import numpy as np

from torch.utils.data import Dataset


LABEL_MAP = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}


def tokenize(example_list):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    pretrained_emb = []
    spacy_tool = en_vectors_web_lg.load()
    pretrained_emb.append(spacy_tool('PAD').vector)
    pretrained_emb.append(spacy_tool('UNK').vector)

    for example in example_list:
        premise = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            example['premise'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        hypothesis = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            example['hypothesis'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        words = [*premise, *hypothesis]

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


class Loader(object):
    def __init__(self, args):
        self.args = args

    def load(self, path, max_token):
        example_list = []

        with codecs.open(path, encoding='utf-8') as f:
            for line in f:
                loaded_example = json.loads(line)
                example_len = max([len(loaded_example['sentence1'].split()),
                                   len(loaded_example['sentence2'].split())])

                if loaded_example['gold_label'] not in LABEL_MAP:
                    continue

                if example_len > max_token:
                    continue

                example = {}
                example['label'] = loaded_example['gold_label']
                example['premise'] = loaded_example['sentence1']
                example['hypothesis'] = loaded_example['sentence2']
                example['example_id'] = loaded_example.get('pairID', 'NoID')

                example_list.append(example)

        return example_list


class Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.loader = Loader(args)
        self.split = {'train': 'train', 'val': 'dev', 'test': 'test'}
        self.data_dict = {}

        # Based on dataset statistics, not many examples length > 50
        self.max_token = 50
        print('Loading datasets...')

        for item in self.split:
            path = './snli_1.0/snli_1.0_' + self.split[item] + '.jsonl'
            self.data_dict[item] = self.loader.load(path, self.max_token)

        # Tokenize from training data only.
        self.token_to_ix, self.pretrained_emb = tokenize(self.data_dict['train'])
        self.token_size = self.token_to_ix.__len__()
        self.label_size = LABEL_MAP.__len__()
        self.data_size = self.data_dict[self.args.RUN_MODE].__len__()
        print('Finished!\n')

    def proc_seqs(self, example, token_to_ix, max_token):
        premise_ix = torch.zeros(max_token, dtype=torch.long)
        hypothesis_ix = torch.zeros(max_token, dtype=torch.long)

        premise = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            example['premise'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        hypothesis = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            example['hypothesis'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(premise):
            if word in token_to_ix:
                premise_ix[ix] = token_to_ix[word]
            else:
                premise_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        for ix, word in enumerate(hypothesis):
            if word in token_to_ix:
                hypothesis_ix[ix] = token_to_ix[word]
            else:
                hypothesis_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return premise_ix, hypothesis_ix

    def __getitem__(self, idx):
        example_iter = self.data_dict[self.args.RUN_MODE][idx]
        premise_iter, hypothesis_iter = self.proc_seqs(example_iter,
                                                       self.token_to_ix,
                                                       self.max_token)

        label_iter = torch.tensor([LABEL_MAP[example_iter['label']]],
                                  dtype=torch.long)

        return premise_iter, hypothesis_iter, label_iter

    def __len__(self):
        return self.data_dict[self.args.RUN_MODE].__len__()
