'''
Trainer for training the model
'''

import pickle
import torch
import os
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


# Loading configuration
import sys
sys.path.append("../config/")
from config import Config
cfg = Config


class Highlighter:
    '''
    Trainer for training the model
    '''

    def __init__(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.trainer['model'])
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.trainer['model'], num_labels=2)
        
        # Loading all required metric module
        self.metric_accuracy = load_metric('accuracy')
        self.metric_precision = load_metric('precision')
        self.metric_recall = load_metric('recall')
        self.metric_f1 = load_metric('f1')

        # load dataset
        self.load_dataset()

        # tokenizing dataset
        self.tokenize_dataset()


    def load_dataset(self) -> None:
        '''
        Loading debatesum dataset
        '''
        with open(f"{cfg.data_preparation['processed_data_path']}/{cfg.trainer['dataset_name']}.pickle", "rb") as input_file:
            self.debatesum = pickle.load(input_file)


    def tokenize_and_align_labels(self, examples) -> dict:
        '''
        Adding special tokens [CLS] and [SEP] and Subword tokenization.
        Realigning labels, subword tokenizer may split single word into multiple subwords.

        Arguments: examples(dict), {'id': , 'tokens': , 'labels': }

        Returns: dict, {'input_ids': , 'attention_mask': , 'labels': }
        '''
    
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(examples["labels"][word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = label_ids

        return tokenized_inputs


    def tokenize_dataset(self) -> None:
        '''
        Tokenization of debatesum dataset
        '''

        self.tokenized_debatesum = {'train': [],
                               'valid': [],
                               'test': []
                              }

        for data_point in map(self.tokenize_and_align_labels, self.debatesum['train']):
            self.tokenized_debatesum['train'].append(data_point)

        for data_point in map(self.tokenize_and_align_labels, self.debatesum['valid']):
            self.tokenized_debatesum['valid'].append(data_point)

        for data_point in map(self.tokenize_and_align_labels, self.debatesum['test']):
            self.tokenized_debatesum['test'].append(data_point)
    

    def compute_metrics(self, p) -> dict:
        '''
        Computing precision, recall, f1 and accuracy

        Arguments: p(predictions, labels)

        Returns: dict, {'precision': , 'recall': , 'f1': , 'accuracy': }
        '''

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        label_list = [0, 1]

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Flat list from list of list
        true_predictions = [item for sublist in true_predictions for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        accuracy = self.metric_accuracy.compute(predictions=true_predictions, references=true_labels)
        precision = self.metric_precision.compute(predictions=true_predictions, references=true_labels)
        recall = self.metric_recall.compute(predictions=true_predictions, references=true_labels)
        f1 = self.metric_f1.compute(predictions=true_predictions, references=true_labels)

        results = {}
        results.update(accuracy)
        results.update(precision)
        results.update(recall)
        results.update(f1)

        return results


    def train_and_save(self) -> None:
        '''
        Training the model
        '''

        training_args = TrainingArguments(
            output_dir=cfg.trainer["output_dir"],
            evaluation_strategy=cfg.trainer["evaluation_strategy"],
            learning_rate=cfg.trainer["learning_rate"],
            per_device_train_batch_size=cfg.trainer["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg.trainer["per_device_eval_batch_size"],
            num_train_epochs=cfg.trainer["num_train_epochs"],
            weight_decay=cfg.trainer["weight_decay"],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_debatesum["train"],
            eval_dataset=self.tokenized_debatesum["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        # traing the model
        trainer.train()

        # Evaluation on Test dataset
        print("@@@@@ Evaluating on test dataset @@@@@")
        result = self.test_dataset_evaluation(trainer)        
        print(result)


        # Saving trained model
        if not os.path.exists(cfg.trainer['trained_model_path']+'/'+cfg.trainer['trained_model_label']):
            os.mkdir(cfg.trainer['trained_model_path']+'/'+cfg.trainer['trained_model_label'])

        trainer.save_model(cfg.trainer['trained_model_path']+'/'+cfg.trainer['trained_model_label'])


    def test_dataset_evaluation(self, trainer) -> dict:
        '''
        Evaluating the metrics on test data of debatesum

        Arguments: trainer(transformer's Trainer object)

        Returns: dict, {'overall_precision': , 'overall_recall': ,
                        'overall_f1': , 'overall_accuracy': }
        '''

        predictions, labels, _ = trainer.predict(self.tokenized_debatesum["valid"])
        predictions = np.argmax(predictions, axis=2)

        label_list = [0, 1]

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Flat list from list of list
        true_predictions = [item for sublist in true_predictions for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        accuracy = self.metric_accuracy.compute(predictions=true_predictions, references=true_labels)
        precision = self.metric_precision.compute(predictions=true_predictions, references=true_labels)
        recall = self.metric_recall.compute(predictions=true_predictions, references=true_labels)
        f1 = self.metric_f1.compute(predictions=true_predictions, references=true_labels)

        results = {}
        results.update(accuracy)
        results.update(precision)
        results.update(recall)
        results.update(f1)

        return results


print("Loading trainer...")
highlighter_trainer = Highlighter()

print("Model training started...")
highlighter_trainer.train_and_save()
print("Model training completed...")
