import pandas as pd
import re
import numpy as np
from multiprocessing import Pool
import os
from CocCocTokenizer import PyTokenizer

T = PyTokenizer(load_nontone_data=True)

PUNCTUATION = "~!@$%^&*()-_+={}[]|;:'`<>?/.,\""
add_space_pttn_1 = re.compile(r"([^\s])([{0}])(\s+|$|[{0}])".format(re.escape(PUNCTUATION)))
add_space_pttn_2 = re.compile(r"(\s|^)([{}])([^\s])".format(re.escape(PUNCTUATION)))
url_pattern = re.compile(
    r'http[s]?\s*:\s*\/\s*\/\s*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
num_partitions = 100
num_cores = 30


def clean_raw_text(input_str):
    def recover_punc(regex_group):
        candidate = regex_group.group(1) + regex_group.group(2) + regex_group.group(3)
        if candidate in input_str:
            return candidate
        else:
            return "{} {} {}".format(regex_group.group(1), regex_group.group(2), regex_group.group(3))

    format_text = str(input_str)

    # encode url
    format_text = url_pattern.sub('url', format_text)

    # replace space
    format_text = format_text.replace('\xa0', ' ')
    format_text = re.sub(r'\s+', ' ', format_text)
    format_text = ' '.join(T.word_tokenize(format_text))
    # recover punc separate
    while True:
        old = format_text
        format_text = re.sub(r"([^\s]+) ([{0}]) ([^\s]+)".format(re.escape(PUNCTUATION)), recover_punc, format_text)
        if old == format_text:
            break

    format_text = add_space_pttn_1.sub(r"\1 \2 \3", format_text)
    format_text = add_space_pttn_2.sub(r"\1 \2 \3", format_text)

    # remove punc
    format_text = re.sub(r"([{0}])\s".format(re.escape(PUNCTUATION)), ' ', format_text)
    format_text = re.sub(r"\s([{0}])".format(re.escape(PUNCTUATION)), ' ', format_text)

    # encode number
    format_text = re.sub(r"\d+", r'\\d', format_text)

    # clean space
    format_text = re.sub(r" +", r" ", format_text)
    format_text = format_text.strip()

    # lower case
    format_text = format_text.lower()

    return format_text


def clean_df_text(df_idf):
    df_idf['text'] = df_idf['title'] + " " + df_idf['sapo'] + " " + df_idf['content']
    df_idf['text'] = df_idf['text'].apply(lambda x: clean_raw_text(x))
    return df_idf


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def clean_topic_data(input_csv, output_file):
    data = pd.read_csv(input_csv, low_memory=False, encoding='utf-8')
    data = parallelize_dataframe(data, clean_df_text)
    clean_list = data['text'].tolist()
    with open(output_file, 'w', encoding='utf-8') as file_result:
        for item in clean_list:
            file_result.write('{}\n'.format(item))


if __name__ == "__main__":
    root_data_path = "./data-bin/raw"
    out_data_path = "./data-bin/clean"
    for topic_name in ["congnghe"]:
        file_in = os.path.join(root_data_path, '{}.csv'.format(topic_name))
        file_out = os.path.join(out_data_path, '{}.txt'.format(topic_name))
        print('Handle topic ', topic_name)
        clean_topic_data(file_in, file_out)
