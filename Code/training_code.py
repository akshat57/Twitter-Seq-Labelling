import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report
from load_data import initialize_data
from reading_datasets import read_ud_dataset
from labels_to_ids import tweebank_labels_to_ids


def train(epoch, training_loader, model, optimizer, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += output[0]

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        
        # backward pass
        optimizer.zero_grad()
        output[0].backward()
        optimizer.step()
        break

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def testing(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            print('-'*20, 'BATCH:', idx)
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output[0].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_accuracy

def read_tb_gum():
    tb_location = '../Datasets/POSTagging/Tweebank/'
    train_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'train')
    dev_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'dev')
    test_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'test')

    gum_location = '../Datasets/POSTagging/GUM/'
    train_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'train')
    dev_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'dev')
    test_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'test')

    train_labels = tweebank_labels_to_ids
    dev_labels = tweebank_labels_to_ids
    test_labels = tweebank_labels_to_ids

    return train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels


if __name__ == '__main__':
    #Initialization parameters
    max_len = 256
    train_batch_size = 32
    dev_batch_size = 1
    test_batch_size = 32
    learning_rate = 1e-05
    initialization_input = (max_len, train_batch_size, dev_batch_size, test_batch_size)
    epochs = 10

    #Reading datasets and initializing data loaders
    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels = read_tb_gum()
    input_data_gum = (train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels)
    input_data_tb = (train_tb, dev_tb, test_tb, train_labels, dev_labels, test_labels)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    model_name = "bert-base-uncased"
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(train_labels))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader, dev_loader, test_loader = initialize_data(tokenizer, initialization_input, input_data_gum)
    train_loader_tb, dev_loader_tb, test_loader_tb = initialize_data(tokenizer, initialization_input, input_data_tb)


    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")
        #model = train(epoch, train_loader, model, optimizer)
        labels_dev, predictions_dev, dev_accuracy = testing(model, dev_loader)
        #labels_test, predictions_test, test_accuracy = testing(model, test_loader)
        #labels_test_tb, predictions_test_tb, test_accuracy_tb = testing(model, test_loader_tb)
        print(classification_report([labels_test], [predictions_test]))
        print()
        break
