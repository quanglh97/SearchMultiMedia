from handle_csv import clean_raw_text
import gensim
import _pickle as cPickle


def transform_ngram(text, phraser_list):
    line_words = text.split()

    for phraser in phraser_list:
        line_words = phraser[line_words]
    line_transform = " ".join(line_words)
    return line_transform


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


if __name__ == "__main__":
    phraser_models = [
        gensim.models.phrases.Phraser.load('./model-bin/bigram_big.pkl'),
        gensim.models.phrases.Phraser.load('./model-bin/trigram_big.pkl'),
    ]

    with open('./model-bin/tfidf_vectorizer.pk', 'rb') as fin:
        tfidf_vectorizer = cPickle.load(fin)
        feature_names = tfidf_vectorizer.get_feature_names()

    while True:
        test_doc = input("Input paragraph: ")

        test_doc = clean_raw_text(test_doc)
        test_doc = transform_ngram(test_doc, phraser_models)

        tf_idf_vector = tfidf_vectorizer.transform([test_doc])

        # sort the tf-idf vectors by descending order of scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())

        # extract only the top n; n here is 10
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)

        # now print the results
        print("\n=====Doc=====")
        print(test_doc)
        print("\n===Keywords===")
        for k in keywords:
            print(k, keywords[k])
