import handle_csv
import make_phraser
import os
import glob
from tqdm import tqdm
import _pickle as cPickle
from sklearn.feature_extraction.text import TfidfVectorizer

root_data_path = "./data-bin/raw"
clean_data_path = "./data-bin/clean"
tokenize_data_path = "./data-bin/tokenize"


def get_data(list_file_data):
    print(list_file_data)
    files_obj = [open(path, 'r', encoding='utf-8') for path in list_file_data]
    pbar = tqdm(desc="Process tf-idf fit")
    while True:
        count = 0
        for file in files_obj:
            try:
                content = next(file)
                count += 1
                pbar.update(1)
                yield content
            except Exception as e:
                # print(e)
                pass
        if count == 0:
            break


if __name__ == "__main__":
    csv_data_files = [f for f in glob.glob(os.path.join(root_data_path, "*.csv"), recursive=True)]
    for topic_file_path in csv_data_files:
        topic_name = topic_file_path[topic_file_path.rindex('/') + 1:-4]
        file_in = topic_file_path
        file_out = os.path.join(clean_data_path, '{}.txt'.format(topic_name))
        print('Handle topic ', topic_name)
        handle_csv.clean_topic_data(file_in, file_out)

    clean_data_files = [f for f in glob.glob(os.path.join(clean_data_path, "*.txt"), recursive=True)]
    # make ngram model
    bigram_model = make_phraser.make_phrases(clean_data_files, phrases_init=None)
    trigram_model = make_phraser.make_phrases(clean_data_files, phrases_init=bigram_model)
    # save ngram model
    bigram_model.save('./model-bin/bigram_big.pkl')
    trigram_model.save('./model-bin/trigram_big.pkl')
    # transform data using ngram model
    make_phraser.transform_phrases(
        clean_data_files,
        tokenize_data_path,
        ['./model-bin/bigram_big.pkl', './model-bin/trigram_big.pkl']
    )

    tokenize_data_files = [f for f in glob.glob(os.path.join(tokenize_data_path, "*.txt"), recursive=True)]
    # Read stop words
    stopwords = open("./model-bin/vietstopwords.txt", 'r', encoding='utf-8').read().split("\n")
    data = get_data(tokenize_data_files)
    # Create tf-idf model
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, stop_words=stopwords, max_features=800000)
    tfidf_vectorizer.fit(data)
    # Save tf-idf model
    with open('./model-bin/tfidf_vectorizer.pk', 'wb') as fin:
        cPickle.dump(tfidf_vectorizer, fin)
