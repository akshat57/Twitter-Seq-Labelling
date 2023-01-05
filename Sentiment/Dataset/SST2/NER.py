from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from load_data import read_data_sst2
from useful_functions import save_data
from torch import cuda
import os

os.environ["CUDA_VISIBLE_DEVICES"]="100"


def do_ner(ner_pipeline, example):
    ner_results = nlp(example)
    
    return ner_results


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    dataset, _ = read_data_sst2('test.tsv')
    ner_dataset = []

    for i, (label, sentence) in enumerate(dataset):
        entities = do_ner(nlp, sentence)
        ner_dataset.append((label, sentence, entities))

        if (i+1) % 100 == 0:
            print('DONE:', i + 1)

    save_data('test_ner.pkl', ner_dataset)



