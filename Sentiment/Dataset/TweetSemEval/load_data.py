
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

if __name__ == '__main__':
    
    for filename in ['train.txt', 'dev.txt', 'test.txt']:
        _, stats = read_data_twitter(filename)

        print(filename, stats)
