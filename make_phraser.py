import gensim
import glob
import os
from tqdm import tqdm
from gensim.models.phrases import Phraser, Phrases

root_data_path = './data-bin/clean'
tokenize_data_path = './data-bin/tokenize'

data_files = files = [f for f in glob.glob(os.path.join(root_data_path, "*.txt"), recursive=True)]


def make_phrases(list_file_data, phrases_init=None):
    phrases = Phrases(min_count=100, threshold=500, max_vocab_size=400000000)
    for file_path in list_file_data:
        with open(file_path, 'r', encoding='utf-8') as file_data:
            with tqdm(file_data, desc="{}: {} words".format(file_path, 0)) as progress:
                for line in progress:
                    if phrases_init:
                        phrases.add_vocab([phrases_init[line.split()]])
                    else:
                        phrases.add_vocab([line.split()])
                    progress.desc = "{}: {} words".format(file_path, len(phrases.vocab))
                    progress.update()

    ngram = Phraser(phrases)

    print("Total {} phrases".format(len(ngram.phrasegrams)))
    return ngram


def transform_phrases(list_file_data, target_dir, phraser_list):
    phrasers = [gensim.models.phrases.Phraser.load(item) for item in phraser_list]
    for file_path in list_file_data:
        file_name = file_path[file_path.rindex(os.path.sep) + 1:]
        with open(os.path.join(target_dir, file_name), 'w', encoding='utf-8') as file_out:
            with open(file_path, 'r', encoding='utf-8') as file_data:
                with tqdm(file_data, desc="{}: {} words".format(file_path, 0)) as progress:
                    for line in progress:
                        line_words = line.split()

                        for phraser in phrasers:
                            line_words = phraser[line_words]
                        line_transform = " ".join(line_words)
                        file_out.write("{}\n".format(line_transform))


if __name__ == "__main__":
    # make ngram model
    bigram_model = make_phrases(data_files, phrases_init=None)
    trigram_model = make_phrases(data_files, phrases_init=bigram_model)
    # save ngram model
    bigram_model.save('./model-bin/bigram_big.pkl')
    trigram_model.save('./model-bin/trigram_big.pkl')
    # transform data using ngram model
    transform_phrases(
        data_files,
        tokenize_data_path,
        ['./model-bin/bigram_big.pkl', './model-bin/trigram_big.pkl']
    )
