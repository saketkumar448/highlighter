'''
Loading the "debatesum" data and labelling tokens of the "text".
if "text" tokens are present in "extract": label it 1
if "text" tokens are not present in "extract": label it 0
'''

import difflib
import random
import pickle
import math
from datasets import load_dataset

# Loading configuration
import sys
sys.path.append("../config/")
from config import Config
cfg = Config

# Loading preprocessor
sys.path.append("../preprocessor/")
from preprocessor import Preprocessor 


class DatasetBuilder:
    '''
    Loading the "debatesum" data and labelling tokens of the "text".
    if "text" tokens are present in "extract": label it 1
    if "text" tokens are not present in "extract": label it 0
    '''

    def __init__(self) -> None:
        
        # loading the debatesum data
        self.dataset = load_dataset("Hellisotherpeople/DebateSum")

        # loading preprocessor
        self.preprocessor = Preprocessor()

        # Train/Valid/Test indexes
        self.train_idxs = None
        self.valid_idxs = None
        self.test_idxs = None


    def gen_tokens_labels(self, text, summ) -> (list, list):
        '''
        Word tokenization of both strings. 
        Then labelling the tokenized "text" as per the tokenized "summ".
        If the phrase is present in "summ" then "1" or "0" if not present.

        Arguments: text(str), full text
                   summ(str), extract from the full text

        Returns: tokens(list), list of string
             labels(list), list of 1's and 0's
        '''

        # cleaning "text" and "summ"
        text = self.preprocessor.clean_text(text)
        summ = self.preprocessor.clean_text(summ)
    
        # Tokenization using split function
        text = text.split()
        tokens = text
        summ = summ.split()
    
        # Generating labels
        labels = []
        d = difflib.Differ()
        for token in d.compare(text, summ):
        
            if token[0] == ' ':
                labels.append(1)
            elif token[0] == '-':
                labels.append(0)
            else:
                pass
            
        return (tokens, labels)

    
    def gen_splits_indexes(self) -> None:
        '''
        Generating Train, Valid and Test examples indexes.
        Updating respective class variables
        '''

        all_idx = list(range(cfg.data_preparation['dataset_size']))

        # Examples count in different splits
        train_count = math.floor(cfg.data_preparation['train_split'] * len(all_idx))
        valid_count = math.floor(cfg.data_preparation['valid_split'] * len(all_idx))
        test_count = math.floor(cfg.data_preparation['test_split'] * len(all_idx))

        # shuffling the list
        random.shuffle(all_idx)

        # Dividing indexes into Train, Valid and Test 
        self.train_idxs = all_idx[:train_count]
        self.valid_idxs = all_idx[train_count: train_count+valid_count]
        self.test_idxs = all_idx[train_count+valid_count:]


    def prepare_dataset(self) -> None:
        '''
        Prepare the dataset for training and dump it
        '''

        # computing train/valid/test indexes
        self.gen_splits_indexes()

        # Preparing valid dataset
        valid_dict = []   # list of example, example = {'id': '5', 'tokens': [list of tokens], 'labels': [list of label]}
        for idx in self.valid_idxs:
            text = self.dataset['train'][idx]['Full-Document']
            summ = self.dataset['train'][idx]['Extract']

            # Generating labels for text
            tokens, labels = self.gen_tokens_labels(text, summ)

            assert len(tokens) == len(labels), "length of tokens and its labels must be same"

            valid_dict.append({'id': idx,
                              'tokens': tokens,
                              'labels': labels})
    
        # Preparing test dataset
        test_dict = []   # list of example, example = {'id': '5', 'tokens': [list of tokens], 'labels': [list of label]}
        for idx in self.test_idxs:
            text = self.dataset['train'][idx]['Full-Document']
            summ = self.dataset['train'][idx]['Extract']

            # Generating labels for text
            tokens, labels = self.gen_tokens_labels(text, summ)

            assert len(tokens) == len(labels), "length of tokens and its labels must be same"

            test_dict.append({'id': idx,
                              'tokens': tokens,
                              'labels': labels})
    

        # Preparing train dataset
        train_dict = []   # list of example, example = {'id': '5', 'tokens': [list of tokens], 'labels': [list of label]}
        for idx in self.train_idxs:
            text = self.dataset['train'][idx]['Full-Document']
            summ = self.dataset['train'][idx]['Extract']

            # Generating labels for text
            tokens, labels = self.gen_tokens_labels(text, summ)

            assert len(tokens) == len(labels), "length of tokens and its labels must be same"

            train_dict.append({'id': idx,
                              'tokens': tokens,
                              'labels': labels})
        
        # putting train test into single dict
        debate_sum = {'train': train_dict,
                     'valid': valid_dict,
                     'test': test_dict}

        # saving debate_sum
        with open(f"{cfg.data_preparation['processed_data_path']}/{cfg.data_preparation['dataset_name']}.pickle", 'wb') as handle:
            pickle.dump(debate_sum, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Creating instance of "DatasetBuilder"
obj = DatasetBuilder()
print("Dataset creation process initiaited...")
obj.prepare_dataset()
print("Dataset has been created.")

