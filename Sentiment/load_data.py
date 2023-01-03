from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import numpy as np
import torch


class dataset(Dataset):
  def __init__(self, all_data, tokenizer, max_len):
        self.len = len(all_data)
        self.data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        label = self.data[index][0]
        sentence = self.data[index][1]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             is_split_into_words=False, #no is_pretokenlized(Modification), we already have a splitted sentence
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(label)
        item['sentence'] = sentence

        return item

  def __len__(self):
        return self.len



def initialize_data(tokenizer, initialization_input, train_standard, dev_standard, test_standard, evaluate_twitter):
  max_len, train_batch_size, dev_batch_size, test_batch_size = initialization_input

  training_set = dataset(train_standard, tokenizer, max_len)
  validation_set = dataset(dev_standard, tokenizer, max_len)
  testing_set = dataset(test_standard, tokenizer, max_len)
  evaluate_set = dataset(evaluate_twitter, tokenizer, max_len)

  train_params = {'batch_size': train_batch_size,
              'shuffle': True,
              'num_workers': 4
              }

  dev_params = {'batch_size': dev_batch_size,
              'shuffle': False,
              'num_workers': 4
              }

  test_params = {'batch_size': test_batch_size,
              'shuffle': False,
              'num_workers': 4
              }

  train_loader = DataLoader(training_set, **train_params)
  dev_loader = DataLoader(validation_set, **dev_params)
  test_loader = DataLoader(testing_set, **test_params)
  evaluate_loader = DataLoader(evaluate_set, **test_params)

  return train_loader, dev_loader, test_loader, evaluate_loader




if __name__ == '__main__':
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') #Bert using wordpiece *** may improve further

    train_file = 'twi/train.txt'
    test_file = 'twi/test.txt'
    dev_file = 'twi/dev.txt'

    dev_data = read_pos_standard(dev_file)
    train_data = read_pos_standard(train_file)
    test_data = read_pos_standard(test_file)

    ##Create labels to index mapping
    labels_to_ids = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-DATE':7, 'I-DATE': 8}
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())
    ####end mapping

    #training_set = dataset(train_data, tokenizer, labels_to_ids, MAX_LEN)

    '''df_train = get_sentence_labels(df_train_raw)
    df_valid = get_sentence_labels(df_valid_raw)
    df_test = get_sentence_labels(df_test_raw)'''


