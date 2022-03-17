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
    location = '/home/ubuntu/Datasets/POSTagging/Gimpel/'
    filename = location + 'oct27.conll'

    data = read_pos_standard(filename)

    return data


def read_ritter():
    location = '/home/ubuntu/Datasets/POSTagging/Ritter/'
    filename = location + 'train'

    data = read_pos_standard(filename)

    return data

def read_tweebank():
    location = '/home/ubuntu/Datasets/POSTagging/Tweebank/'
    filename = location + 'en-ud-tweet-train.conllu'

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

            else:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data

def read_conll2003():
    location = '/home/ubuntu/Datasets/POSTagging/CONLL2003/'
    filename = location + 'conll2003_val.pkl'

    saved_data = load_data(filename)
    data = []
    for i, line in enumerate(saved_data):
        data.append( (line['tokens'], line['pos_tags']) )


def get_all_labels(data):
    all_labels = []
    for tokens, labels in data:
        all_labels += labels

    all_labels = set(all_labels)
    return all_labels

if __name__ == '__main__':
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



