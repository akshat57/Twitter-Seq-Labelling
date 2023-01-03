import json 

def read_data_sst2(filename):
    
    dataset = []
    dataset_stats = {0 : 0, 1 : 0}
    with open(filename, 'r') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            label = int(label)
            
            dataset.append((label, sentence))
            dataset_stats[label] += 1

    return dataset, dataset_stats


def read_data_twitter(filename):

    label_dict = {'positive' : 1, 'negative' : 0}

    dataset = []
    dataset_stats = {0 : 0, 1 : 0}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            label = line[1].strip()
            sentence = '\t'.join(line[2:]).strip()

            if label in label_dict:
                label_idx = label_dict[label]
                dataset.append((label_idx, sentence))

                dataset_stats[label_idx] += 1

    return dataset, dataset_stats


def read_lexnorm(location = 'Dataset/WNUTlexnorm2015/'):
    filename = location + 'train_data.json'
    f = open(filename)
    data = json.load(f)

    inorm_dict = {}

    for row in data:
        for input_token, output_token in zip(row['input'], row['output']):
            input_token = input_token.lower()
            output_token = output_token.lower()

            if input_token != output_token:
                if output_token not in inorm_dict:
                    inorm_dict[output_token] = [input_token]
                else:
                    inorm_dict[output_token].append(input_token)

    for key in inorm_dict:
        inorm_dict[key] = list(set(inorm_dict[key]))

    return inorm_dict