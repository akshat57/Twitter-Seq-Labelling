from useful_functions import load_data, save_data
from sklearn import metrics
import random
import numpy as np
from symbols import twitter_symbols
from read_data import read_data_sst2, read_data_twitter, read_lexnorm
import json
import os

def replace_entity(sentence, entity, replace_symbol):
    new_entity = replace_symbol + entity.replace(' ','')
    new_sentence = sentence.replace(entity, new_entity)

    return new_sentence

def convert_usermention_hashtags(data):
    augmented_data = []

    for (label, sentence, ner) in data:
        if ner:

            ##Finding and joining all entities
            all_entities = {}
            e_end_prev = -2
            prev_e_type = None
            for entity in ner:
                e_type = entity['entity']
                e_type_id, current_e_type = e_type.split('-')
                
                e_word = entity['word']
                e_start_current = entity['start']
                e_end_current = entity['end']

                if current_e_type not in all_entities:
                    all_entities[current_e_type] = []

                #If word is split into subwords
                if e_start_current == e_end_prev:
                    if len(e_word)> 2 and e_word[:2] == '##':
                        all_entities[prev_e_type][-1] += e_word[2:]
                    else:
                        all_entities[prev_e_type][-1] += e_word
                
                #If entity has more than one word
                elif e_start_current - e_end_prev == 1:
                    if current_e_type == prev_e_type and e_type[0] == 'I':
                        all_entities[current_e_type][-1] += ' ' + e_word
                        prev_e_type = current_e_type
                
                #Adding a new entitye
                else:
                    all_entities[current_e_type].append(e_word)
                    prev_e_type = current_e_type

                e_end_prev = e_end_current
            ###Finished finding entities


            ###USUALLY IN TWEETS, ENTITIES ARE HIGHLY LIKELY TO BE REPLACED BY USERMENTIONS AND HASHTAGS

            ##For PER/ORG
            '''
            For a tweet, either all PER/ORG are usermentions or hashtags
            First select if we are going to replace user mentin/hashtags to PER/ORG. We do this 90% of times.
            Out of replacement, 75% is usermention, 25% is hashtags
            '''
            replace_perorg = False
            perorg_symbol = None
            if random.random() < 0.9:
                replace_perorg = True
                if random.random() < 0.75:
                    perorg_symbol = '@'
                else:
                    perorg_symbol = '#'

            ##For MISC/LOC
            '''We replace MISC/LOC entities 80% of times to hashtags'''
            replace_locmisc = False
            if random.random() < 0.8:
                replace_locmisc = True
            
            for e_type in all_entities:
                if e_type in ['PER', 'ORG'] and replace_perorg:
                    for entity in all_entities[e_type]:
                        sentence = replace_entity(sentence, entity, perorg_symbol)
                elif e_type in ['LOC', 'MISC'] and replace_locmisc:
                    for entity in all_entities[e_type]:
                        sentence = replace_entity(sentence, entity, '#')

        augmented_data.append((label, sentence, ner))

    return augmented_data       



def inorm_sym(data, twitter_symbols, random_placement = False, random_symbol_selection = True, threshold = 0.25):
    #Add twitter like symbols in randomly

    twitter_symbols = list(set(twitter_symbols))
    augmented_data = []

    for (label, sentence, ner) in data:

        tokens = sentence.split(' ')
        #Add random twitter symbols from the unlabelled train set
        if random.random() < threshold:

            #placing the symbol randomly vs based on distribution
            if random_placement:
                index = random.randint(0, len(tokens) - 1)
            else:
                index = int( min( max(0, np.random.normal(0.771, 0.269, 1) ), 1) * len(tokens) )

            #selecting symbol randomly right now
            selected_symbol = random.choice(twitter_symbols)

            tokens.insert(index, selected_symbol)
            #labels.insert(index, 'SYM') #not need for Sentiment

        recombined_sentence = ' '.join(tokens)
        augmented_data.append((label, recombined_sentence, ner))

    return augmented_data

def do_ilexnorm(data, threshold = 0.8):
    #Do inverse lexcial normalization
    inorm_dict = read_lexnorm()
    augmented_data = []

    for (label, sentence, ner) in data:

        tokens = sentence.split(' ')
        new_tokens = []
        
        sent_flag = True
        for i, token in enumerate(tokens):
            new_token = token.lower()

            if new_token in inorm_dict:
                
                #do lexical un-normalization with [THRESHOLD]% of the times the word is found in dictionary
                if random.random() < threshold:
                    new_token = random.choice(inorm_dict[new_token])

                #ADD : Code to do ILN based on occurence distribution
            
            new_tokens.append(new_token)

        recombined_sentence = ' '.join(new_tokens)

        augmented_data.append((label, recombined_sentence, ner))

    return augmented_data

def generate_url():
    ID_set = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(48, 58)]
    
    ID_len = random.choice([8,9,10])#ID length can be between 8-10
    url_ID = ''
    for i in range(ID_len):
        url_ID += random.choice(ID_set)

    url = 'https://t.co/' + url_ID

    return url

def add_urls(data):
    ## Add URL to the end 50% of times
    # Add URL to beginning 5% of times
    ## Add URL somewhere in between 5% of times

    augmented_data = []
    for (label, sentence, ner) in data:

        #Add URL at the end
        if random.random() < 0.5: 
            url = generate_url()
            sentence += ' ' + url

        #Add URL at the start
        if random.random() < 0.05:
            url = generate_url()
            sentence = url + ' ' + sentence

        #Add URL in between
        if random.random() < 0.05:
            url = generate_url()
            
            sentence_tokens = sentence.split(' ')
            insert_index = random.randint(0, len(sentence_tokens))
            sentence_tokens.insert(insert_index, url)

            sentence = ' '.join(sentence_tokens)

        augmented_data.append((label, sentence, ner))

    return augmented_data


def process_augmented(data):
    '''remove ner'''

    augmented_data = []
    for (label, sentence, ner) in data:
        augmented_data.append((label, sentence))

    return augmented_data


if __name__ == '__main__':

    data_location = 'Dataset/SST2/'
    

    transformation_type = 'alltransformations'
    save_location = data_location + transformation_type + '/'
    os.makedirs(save_location, exist_ok = True)

    ##For propN, do that first and then do emoji and ILN
    for split in ['train', 'dev', 'test']:
        train_filename = data_location + split + '_ner.pkl'
        train_dataset = load_data(train_filename)
        print(split, len(train_dataset))
        for i in range(5):
            save_filename = save_location + split + '_' + transformation_type + '_' + str(i) + '.pkl'
            
            dataset = train_dataset.copy()
            
            dataset = convert_usermention_hashtags(dataset)
            dataset = inorm_sym(dataset, twitter_symbols, random_placement = False, random_symbol_selection = True, threshold = 0.25)
            dataset = do_ilexnorm(dataset, threshold = 0.75)
            dataset = add_urls(dataset)

            print(i, save_filename, len(dataset))
            dataset = process_augmented(dataset)
            save_data(save_filename, dataset)
