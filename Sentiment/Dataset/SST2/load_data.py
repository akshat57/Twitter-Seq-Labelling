
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

if __name__ == '__main__':
    filename = 'test.tsv'
    read_data_sst2(filename)

    for filename in ['train.tsv', 'dev.tsv', 'test.tsv']:
        _, stats = read_data_sst2(filename)

        print(filename, stats)