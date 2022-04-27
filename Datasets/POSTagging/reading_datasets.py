import pickle

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_pos_standard(filename, sep = '\t'):
    '''
    Reads the standard POS tagging input data format with token SEP label format. Utterance are separated by '\n'. SEP token is input to the function.
    INPUTS:
        filename: reading file name
        sep: separator token between token and labels in the standard format. Usually '\t' or space.  
    '''

    data = []
    tokens = []
    labels = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line[:-1]

            if len(line):
                token, label = line.split(sep)
                tokens.append(token)
                labels.append(label)

            else:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data



def read_gimpel():
    location = 'Gimpel/'
    filename = location + 'oct27.conll'

    data = read_pos_standard(filename)

    return data


def read_ritter():
    location = 'Ritter/'
    filename = location + 'train'

    data = read_pos_standard(filename)

    return data

def read_tweebank(split = 'train'):
    location = 'Datasets/POSTagging/Tweebank/'
    filename = location + 'en-ud-tweet-' + split + '.conllu'

    data = []
    tokens = []
    labels = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line[:-1]

            if len(line) and line[0] != '#':
                line = line.split('\t')
                tokens.append(line[1])
                labels.append(line[3])

            elif len(line) == 0:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data

def reverse_dictionary(input_dict):
    output_dict = {}
    
    for key in input_dict:
        output_dict[input_dict[key] ] = key
        
    return output_dict

def read_conll2003():
    tag_mapping = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}
    
    index_to_pos = reverse_dictionary(tag_mapping)
    
    
    location = 'CONLL2003/'
    filename = location + 'conll2003_train.pkl'

    saved_data = load_data(filename)
    data = []
    for i, line in enumerate(saved_data):
        labels = [index_to_pos[label] for label in line['pos_tags']]
        data.append( (line['tokens'], labels) )
        
    return data


def get_all_labels(data):
    all_labels = []
    for tokens, labels in data:
        all_labels += labels

    all_labels = set(all_labels)
    return all_labels

if __name__ == '__main__':
    read_conll2003()
    exit()
    
    
    gimpel_data = read_gimpel()
    ritter_data = read_ritter()
    tweebank_data = read_tweebank()

    gimpel_labels = get_all_labels(gimpel_data)
    ritter_labels = get_all_labels(ritter_data)
    tweebank_labels = get_all_labels(tweebank_data)

    print('GIMPEL:', len(gimpel_labels))
    print(gimpel_labels)
    print()

    print('RITTER:', len(ritter_labels))
    print(ritter_labels)
    print()

    print('TWEEBANK:', len(tweebank_labels))
    print(tweebank_labels)
    print()



