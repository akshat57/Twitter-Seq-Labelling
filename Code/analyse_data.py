from reading_datasets import read_ud_dataset
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from textblob import TextBlob
import numpy as np
from reading_datasets import reading_connll_ner, reading_tb_ner
from labels_to_ids import ner_labels_to_ids
import numpy as np

def read_tb_gum():
    tb_location = '../Datasets/POSTagging/Tweebank/'
    train_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'train')
    dev_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'dev')
    test_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'test')

    gum_location = '../Datasets/POSTagging/GUM/'
    train_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'train')
    dev_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'dev')
    test_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'test')

    return train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum


def token_number_distribution(train_tb, train_gum):
    #token distribution
    lengths_gum = [len(tokens) for tokens, labels in train_gum]
    lengths_tb = [len(tokens) for tokens, labels in train_tb]

    sns.histplot(lengths_gum, label='GUM', norm_hist = True)
    sns.histplot(lengths_tb, label = 'TB', norm_hist = True)

    plt.legend()
    plt.savefig('logs/token_distribution.jpg')

def calculate_pos_tag_distribution(data):
    pos_counter = {}
    for tokens, labels in data:
        for pos in labels:
            if pos not in pos_counter:
                pos_counter[pos] = [1]
            else:
                pos_counter[pos][0] += 1

    total = sum([pos_counter[pos][0] for pos in pos_counter] )
    for pos in pos_counter:
        pos_counter[pos][0] /= total

    return pos_counter


def pos_tagging_distribution(train_tb, train_gum):
    tb_pos_tags = pd.DataFrame.from_dict(calculate_pos_tag_distribution(train_tb))
    gum_pos_tags = pd.DataFrame.from_dict(calculate_pos_tag_distribution(train_gum))

    sns.set_color_codes('pastel')
    sns.barplot(data = tb_pos_tags, label = 'tb', color = 'k', edgecolor = 'w', orient = 'h')
    sns.set_color_codes('muted')
    sns.barplot(data = gum_pos_tags, label = 'gum', color = 'b', edgecolor = 'w', orient = 'h', alpha = 0.5)
    plt.legend()
    plt.savefig('logs/pos_distribution.jpg')

def calculate_features(train_tb, train_gum):
    
    sentiments = []
    subjectivity = []
    for i, (tokens, labels) in enumerate(train_gum):
        text = ' '.join(tokens)
        blob = TextBlob(text)

        sentiments.append(blob.sentiment_assessments.polarity)
        subjectivity.append(blob.sentiment_assessments.subjectivity)

    sentiments_gum = np.array(sentiments)
    subjectivity_gum = np.array(subjectivity)

    '''print('GUM')
    print('SENTIMENT:', np.mean(sentiments), np.std(sentiments))
    print('SUBJECTIVITY:', np.mean(subjectivity), np.std(subjectivity))'''

    sentiments = []
    subjectivity = []
    for i, (tokens, labels) in enumerate(train_tb):
        text = ' '.join(tokens)
        blob = TextBlob(text)

        sentiments.append(blob.sentiment_assessments.polarity)
        subjectivity.append(blob.sentiment_assessments.subjectivity)

    sentiments_tb = np.array(sentiments)
    subjectivity_tb = np.array(subjectivity)
    
    '''print()
    print('TB')
    print('SENTIMENT:', np.mean(sentiments), np.std(sentiments))
    print('SUBJECTIVITY:', np.mean(subjectivity), np.std(subjectivity))'''

    sns.distplot(subjectivity_gum, label = 'GUM')
    sns.distplot(subjectivity_tb, label = 'TB')
    plt.legend()
    plt.savefig('logs/subjectivity.jpg')


def ner_distribution(dataset = 'connll'):
    '''
    OBSERVATIONS:
    Twitter has a very different distribution of length of sentences compared to CONNLL. See plots
    The avg sentence length in connll is although 13, but with high standard deviation. 
    The range of sentence lengths are also fixed.
    CONNLL has an unusually large number of sentences with lengths around 3-4.
    The first change syntactic change that can be made is distributional although not sure how much that will help.

    The NER tags are different. 
    Twitter has a larger number of O tags.

    The ratio of B:I for person in Twitter is small. Which means a lot of people in Twitter have shorter names whereas in connll have bigger names. Maybe user mentions?
            ##Look at shorted PER in twitter dataset

    The ratio of B:I for orgs in Twitter is larger than CONNLL. 
            ##Understand why this is so

    The ratio of B:I for location is roughly similar

    The ration for B:I for MISC way less for CONNLL. Twitter has longer MISC entitites. Find out why?

    '''

    if dataset == 'connll':
        data = reading_connll_ner()
    elif dataset == 'tb':
        data = reading_tb_ner()

    all_labels = []
    freq = {}

    sent_len = []
    for i, (tokens, labels) in enumerate(data):
        all_labels.extend(labels)

        for label in labels:
            if label not in freq:
                freq[label] = 1
            else:
                freq[label] += 1

        sent_len.append(len(tokens))

    sent_len = np.array(sent_len)
    print(dataset, 'AVG SENT LENGTH:', np.mean(sent_len), np.std(sent_len), 'MIN:', np.min(sent_len), 'MAX:', np.max(sent_len))

    sns.distplot(sent_len)
    plt.savefig('logs/ner_sent_len_' + dataset + '.jpg')
    plt.close()
    
    normalized = {}
    for label in ner_labels_to_ids:
        normalized[label] = round( (freq[label]/sum(freq.values())) * 100, 5) 

    plt.bar(range(len(normalized)), list(normalized.values()), align='center')
    plt.xticks(range(len(normalized)), list(normalized.keys()))
    plt.savefig('logs/ner_' + dataset + '.jpg')
    plt.close()

    return normalized

    

if __name__ == '__main__':

    #train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum = read_tb_gum()
    #token_number_distribution(train_tb, train_gum)
    #pos_tagging_distribution(train_tb, train_gum)
    #calculate_features(train_tb, train_gum)

    #look at distribution of each POS tag
    #look at distribution of POS tag per sentence
    #look at vocabulary
    #can we convert vocabularies by looking at their POS tags
    #do sentiment analysis
    #do topic modelling
    #think of other ways that can convert you can measure similarity of corpus

    #CREATE A CORPUS SIMILARITY MEASURE USING SOME OF THESE FACTOR
    #DO SCORING BY TAKING A SUBSET OF TRAIN SET, AND MEASURE WITH TEST AND VAL SET
    #SEE IF MAKING GUM DISTRIBUTION CLOSER TO TB HELPS IN PERFORMANCE


    normalized_connll = ner_distribution('connll')
    normalized_tb = ner_distribution('tb')

    for ner in ner_labels_to_ids:
        print(ner, 'C:', normalized_connll[ner], 'T:', normalized_tb[ner])

    


