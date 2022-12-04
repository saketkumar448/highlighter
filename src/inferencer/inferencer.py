'''
Inferencing code
'''

import os
import torch
import math
import pickle
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


# Loading configuration
import sys
sys.path.append("../config/")
from config import Config
cfg = Config


class Inferencer:
    '''
    Inferencing using trained model
    '''

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.trainer['model'])
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.trainer['trained_model_path']+'/'+cfg.trainer['trained_model_label'], num_labels=2)


    def highlight(self, text, use_both_labels = False, ratio=None):
        '''
        Higlighting the important phrases in the text

        Arguments: text(str), input text
                   use_both_labels(bool), False: using only '1' label for inferencing
                                                  True: using both labels for inferencing

        Returns: list of tuples (tokens, labels)
        '''

        input_ids = torch.tensor([self.tokenizer.encode(text)])  # batch size of 1
        model_outputs = self.model(input_ids)

        # labelling tokens as per the model output
        if use_both_labels == True:
            labeled_tokens = self.tokens_labelling_use_both_labels(input_ids, model_outputs)
        else:
            # Using single lable '1' for labelling tokens
            if ratio != None:
                labeled_tokens = self.tokens_labelling_use_single_label(input_ids, model_outputs, ratio)
            else:
                return 0   # ratio is must for labelling using single label

        # Combining tokens which belongs to same word
        labeled_combined_tokens = self.combine_tokens(labeled_tokens)

        return labeled_combined_tokens
        

    def tokens_labelling_use_both_labels(self, input_ids, model_outputs):
        '''
        Labelling tokens using both labels of the model outputs

        Arguments: input_ids(Tensor), tensor of tokens id
                   model_outputs(TokenClassiferOutput), highlighter model output
        
        Returns: labelled tokens(decoded tokens) of respective inputs
        '''

        outputs = []

        for ids, logits in zip(input_ids, model_outputs['logits']):
        
            output = []
            for id, label in zip(ids[1: -1], logits[1: -1]):  # ignoring first[CLS] & last[SEP] token
            
                token = self.tokenizer.decode(int(id))

                # using both labels
                label = np.argmax(label.detach().numpy())

                output.append((token, label))

            outputs.append(output)
        
        return outputs


    def tokens_labelling_use_single_label(self, input_ids, model_outputs, ratio):
        '''
        Labelling tokens using single label of the model outputs

        Arguments: input_ids(Tensor), tensor of tokens id
                   model_outputs(TokenClassiferOutput), highlighter model output
        
        Returns: labelled tokens(decoded tokens) of respective inputs
        '''

        outputs = []

        num_tokens = 30

        for ids, logits in zip(input_ids, model_outputs['logits']):

            output = []

            one_label_logits = torch.hsplit(logits, 2)[1].reshape(512)

            sorted_index = np.argsort(one_label_logits.detach().numpy())[::-1]

            # Number of tokens in highlight
            num_tokens = math.ceil(len(sorted_index)*ratio)

            # Selected indexes for highlights
            selected_idxs = sorted_index[:num_tokens]

            for idx, id in enumerate(ids):
            
                token = self.tokenizer.decode(int(id))
                if idx in selected_idxs:
                    label = 1
                else:
                    label = 0

                output.append((token, label))

            outputs.append(output)
        
        return outputs


    def combine_tokens(self, outputs):
        '''
        Combining tokens to form words

        Arguments: outputs(list of list), labelled tokens(decoded tokens)

        Return: updated_outputs(list of list), multiple tokens are combined if they belong to same word
        '''

        updated_outputs = []

        for output in outputs:

            updated_output = []

            for token, label in output[1:-1]:
                if token.startswith('##'):
                    # last token
                    last_token, last_label = updated_output[-1]
                    last_token = last_token + str(token[2:])
                    last_label = max(last_label, label)
                    updated_output[-1] = (last_token, last_label)
                else:
                    updated_output.append((token, label))
            
            updated_outputs.append(updated_output)
        
        return updated_outputs
