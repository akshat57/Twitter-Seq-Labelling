import pickle
from matplotlib import pyplot as plt

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()

def reading_conll_ner(split = 'train', location = ''):
    if split == 'train':
        filename = location + 'eng.train'
    elif split == 'val':
        filename = location + 'eng.testa'
    elif split == 'test':
        filename = location + 'eng.testb'

    data = []
    tokens = []
    labels = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line[:-1]

            if len(line):
                token, _, _, label = line.split(' ')
                tokens.append(token.lower())
                labels.append(label.strip())

            else:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data

if __name__ == '__main__':
    split = 'test'
    data = reading_conll_ner()

    corrected_dataset = []

    for i, (tokens, labels) in enumerate(data):
        correct_labels = []

        prev_position = ''
        prev_label = ''

        for label in labels:
            if label != 'O':
                position, ner_label = label.split('-')
                
                if ner_label != prev_label:
                    label = 'B-' + ner_label
                else:
                    label = 'I-' + ner_label

                prev_position = position
                prev_label = ner_label

            else:
                prev_position = ''
                prev_label = ''


            correct_labels.append(label)


        corrected_dataset.append((tokens, correct_labels))

    #saving dataset
    save_data('connll_ner_' + split + '.pkl', corrected_dataset)