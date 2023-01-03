import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from load_data import initialize_data
import time
import os
from useful_functions import load_data, save_data
from read_data import read_data_sst2, read_data_twitter
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def train(epoch, training_loader, model, optimizer, device, grad_step = 1, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    optimizer.zero_grad()
    
    sentences = []
    classification_output = []
    true_labels = []
    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)


        #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += output['loss']

        nb_tr_steps += 1

        # backward pass
        output['loss'].backward()
        if (idx + 1) % grad_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        _, prediction = torch.max(output['logits'], 1)

        sentences.extend(batch['sentence'])
        classification_output.extend(prediction.tolist())
        true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, classification_output)
    epoch_loss = tr_loss / nb_tr_steps
    #print(f"Training loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def testing(save_location, iteration, epoch, test_type, model, testing_loader, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    

    sentences = []
    classification_output = []
    true_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output['loss'].item()
            nb_eval_steps += 1
              
            #evaluation
            _, prediction = torch.max(output['logits'], 1)

            sentences.extend(batch['sentence'])
            classification_output.extend(prediction.tolist())
            true_labels.extend(labels.tolist())


    
    eval_loss = eval_loss / nb_eval_steps
    accuracy = accuracy_score(true_labels, classification_output)

    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    #SAVE DATA
    data_save_location = save_location + 'outputs/iteration_' + str(iteration) + '/'
    os.makedirs(data_save_location, exist_ok = True)

    save_data(data_save_location + test_type + '_epoch_' + str(epoch) + '_' + 'sentences.pkl', sentences)
    save_data(data_save_location + test_type + '_epoch_' + str(epoch) + '_' + 'classification_output.pkl', classification_output)
    save_data(data_save_location + test_type + '_epoch_' + str(epoch) + '_' + 'true_labels.pkl', true_labels)


    return true_labels, classification_output, accuracy


def main(save_location, iteration, n_epochs, model_name, dataset_input, model_save_flag, model_save_location, model_load_flag, model_load_location, in_train_logfile):
    #Initialization training parameters
    max_len = 256
    train_batch_size = 4
    dev_batch_size = 8
    test_batch_size = 8
    grad_step = 8
    learning_rate = 1e-05
    initialization_input = (max_len, train_batch_size, dev_batch_size, test_batch_size)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    if model_load_flag:
        tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        model = AutoModelForSequenceClassification.from_pretrained(model_load_location)
    else: 
        tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Read dataset
    train_dataset, dev_dataset, test_dataset, evaluate_dataset = dataset_input
    
    train_standard = load_data(train_dataset)
    dev_standard, _ =  read_data_sst2(dev_dataset)
    test_standard, _ = read_data_sst2(test_dataset)
    evaluate_twitter, _ = read_data_twitter(evaluate_dataset)

    #Get dataloaders
    train_loader, dev_loader, test_loader, evaluate_loader = initialize_data(tokenizer, initialization_input, train_standard, dev_standard, test_standard, evaluate_twitter)


    best_dev_acc = 0
    best_test_acc = 0
    best_epoch = -1
    best_tb_acc = 0
    best_tb_epoch = -1

    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        model = train(epoch, train_loader, model, optimizer, device, grad_step)
        
        #testing and logging
        labels_dev, predictions_dev, dev_accuracy = testing(save_location, iteration, epoch, 'dev', model, dev_loader, device)
        print('DEV ACC:', dev_accuracy)
        
        labels_test, predictions_test, test_accuracy = testing(save_location, iteration, epoch, 'test', model, test_loader, device)
        print('TEST ACC:', test_accuracy)
        
        labels_test_tb, predictions_test_tb, test_accuracy_tb = testing(save_location, iteration, epoch, 'evaluate', model, evaluate_loader, device)
        print('EVALUATE ACC:', test_accuracy_tb)

        #saving model
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            best_test_acc = test_accuracy
            best_epoch = epoch

            best_tb_acc = test_accuracy_tb
            best_tb_epoch = epoch

        #logging
        f = open(in_train_logfile, 'a')
        f.write('EPOCH: ' + str(epoch) + '\n\n')
        f.write('TESTING IN DOMAIN:' + '\n\n')
        f.write(classification_report(labels_test, predictions_test, digits = 5))
        f.write('\n\nEVALUATING OOD:' + '\n\n')
        f.write(classification_report(labels_test_tb, predictions_test_tb, digits = 5))
        f.write('\nDEV ACC : ' + str(round(dev_accuracy, 5)) + '\n')
        f.write('TEST ACC : ' + str(round(test_accuracy, 5)) + '\n')
        f.write('EVALUATE TEST ACC : ' + str(round(test_accuracy_tb, 5)) + '\n')
        f.write('BEST EPOCH : ' + str(best_epoch) + '\n')
        f.write('BEST ACCURACY --> ' +  'DEV:' +  str(round(best_dev_acc, 5)) + ', TEST:' + str(round(best_test_acc, 5)) + '\n')
        f.write('BEST EVALUATE TEST ACC : ' + str(round(best_tb_acc, 5)) + '\n')
        f.write('-'*80 + '\n')
        f.close()

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5), 'TEST:',  round(best_test_acc, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, model, tokenizer


def initialize_logfile(save_location, initialize = False):

    #logfile
    in_train_logfile = save_location + 'intrain_logs.txt'
    result_logfile = save_location + 'results.txt'

    if initialize:
        #initialize logfiles
        f = open(in_train_logfile, 'w')
        f.write('='*50 + '\n')
        f.write('MODEL NAME : ' + model_name.replace('/', '-') + '\n')
        f.write('='*50 + '\n')
        f.close()

        g = open(result_logfile, 'w')
        g.write('='*50 + '\n')
        g.write('MODEL NAME : ' + model_name.replace('/', '-') + '\n')
        g.write('='*50 + '\n')
        g.close()

    return in_train_logfile, result_logfile


if __name__ == '__main__':
    n_iterations = 5
    n_epochs = 10

    training_name = 'sym/'#'zero-shot/'
    os.makedirs(training_name, exist_ok = True)

    #models = ['prajjwal1/bert-tiny', 'distilbert-base-uncased', 
    #        'bert-base-uncased', 'roberta-base', 'cardiffnlp/twitter-roberta-base-sep2022',
    #        'bert-large-uncased', 'vinai/bertweet-large', 'roberta-large']

    models = ['prajjwal1/bert-tiny', 'distilbert-base-uncased', 'bert-base-uncased']

    #define dataset location
    train_dataset = 'Dataset/SST2/train_sym.pkl'
    dev_dataset = 'Dataset/SST2/dev.tsv'
    test_dataset = 'Dataset/SST2/test.tsv'
    evaluate_dataset = 'Dataset/TweetSemEval/test.txt'
    dataset_input = (train_dataset, dev_dataset, test_dataset, evaluate_dataset)

    #Flag initialize
    initialize_logfile_Flag = True
    model_save_flag = False
    model_load_flag = False
    model_load_location = None

    for model_name in models:
        print(model_name)
        save_location = training_name + model_name.replace('/', '-') + '/'
        os.makedirs(save_location, exist_ok = True)
        model_save_location = save_location + 'saved_model'

        #initialize log files
        in_train_logfile, result_logfile = initialize_logfile(save_location, initialize_logfile_Flag)

        best_evaluate_acc = 0
        all_dev_acc, all_test_acc, all_test_tb_acc, all_best_epoch, all_best_tb_epoch = [], [], [], [], []
        for i in range(n_iterations):
            print(model_name, 'ITERATION:', i )

            f = open(in_train_logfile, 'a')
            f.write('ITERAION : ' + str(i) + '\n\n')
            f.close()

            best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, model, tokenizer = main(save_location, i, n_epochs, model_name, dataset_input, model_save_flag, model_save_location, model_load_flag, model_load_location, in_train_logfile)
            all_dev_acc.append(best_dev_acc)
            all_test_acc.append(best_test_acc)
            all_test_tb_acc.append(best_tb_acc)
            all_best_epoch.append(best_epoch)
            all_best_tb_epoch.append(best_tb_epoch)

            #save model
            if best_tb_acc > best_evaluate_acc:
                best_evaluate_acc = best_tb_acc

                if model_save_flag:
                    os.makedirs(model_save_location, exist_ok=True)
                    tokenizer.save_pretrained(model_save_location)
                    model.save_pretrained(model_save_location)


            #logging for results
            g = open(result_logfile, 'a')
            g.write('ITERAION : ' + str(i) + '\n')
            g.write('BEST DEV ACC : ' + str(round(best_dev_acc, 5)) + '\n')
            g.write('BEST TEST ACC : ' + str(round(best_test_acc, 5)) + '\n')
            g.write('BEST EPOCH : ' + str(best_epoch) + '\n')
            g.write('BEST EVALUATE ACC : ' + str(round(best_tb_acc, 5)) + '\n')
            g.write('BEST EVALUATE EPOCH : ' + str(best_tb_epoch) + '\n')
            g.write('-'*30 + '\n')
            g.close()

        #writing mean results
        g = open(result_logfile, 'a')
        g.write('\nFINAL RESULTS : ' + '\n')
        g.write('MEAN DEV ACC : ' + str(round( np.mean(np.array(all_dev_acc)) * 100, 3)) + '|' + 'STD DEV ACC : ' + str(round( np.std(np.array(all_dev_acc)) * 100, 3)) + '\n')
        g.write('MEAN TEST ACC : ' + str(round( np.mean(np.array(all_test_acc)) * 100, 3)) + '|' + 'STD TEST ACC : ' + str(round( np.std(np.array(all_test_acc)) * 100, 3)) + '\n')
        g.write('MEAN BEST EPOCH : ' + str(round( np.mean(np.array(all_best_epoch)), 3)) + '\n')
        g.write('MEAN EVALUATE ACC : ' + str(round( np.mean(np.array(all_test_tb_acc)) * 100, 3)) + '|' + 'STD EVALUATE ACC : ' + str(round( np.std(np.array(all_test_tb_acc)) * 100, 3)) +  '\n')
        g.write('MEAN BEST TBEPOCH : ' + str(round( np.mean(np.array(all_best_tb_epoch)), 3)) +  '\n')
        g.write('-'*30 + '\n')
        g.close()
