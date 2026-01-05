import pandas as pd
import numpy as np
import random
import json
import math
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import itertools
import json
import string
from typing import List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END_COLOR = '\033[0m'


# At least count 1 word appears at least count 2 times Hello world! Hello Python. 1 hello appears twice
def reduplication(sentence, count1, count2):
    translator = str.maketrans('', '', string.punctuation)  # Convert lowercase and remove punctuation marks
    cleaned_sentence = sentence.translate(translator).lower()
    words = cleaned_sentence.split()
    word_counts = Counter(words)
    count_frequency = sum(1 for count in word_counts.values() if count >= count2)
    return count_frequency >= count1

def tongji_data(input_file):
    # Open and read JSON file

    datas = []
    read_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        print(GREEN + "opening "+ input_file + END_COLOR)
        try:
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data) # Analyze the JSON data for each line
                    datas.append(data)
                    read_count+=1
                    print('\r'+ GREEN + "reading "+ str(read_count) + END_COLOR, end='')
                else:
                    break
        except Exception as e:
            print(e)

    
def concat_datas(input_file_lst, output_file):
    # Open and read JSON file
    datas = []
    for input_file in input_file_lst:
        read_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            print(GREEN + "opening "+ input_file + END_COLOR)
            try:
                while True:
                    line_data = f.readline()
                    if line_data:
                        data = json.loads(line_data) # Analyze the JSON data for each line
                        datas.append(data)
                        read_count+=1
                        print('\r'+ GREEN + "reading "+ str(read_count) + END_COLOR, end='')
                    else:
                        break
            except Exception as e:
                print(e)

    with open(output_file, 'w', encoding='utf-8') as file:
        for i in tqdm(range(len(datas)), desc='writing file'):
            # Use json. dumps to serialize a single dictionary into a JSON string and write it to a file
            file.write(json.dumps(datas[i], ensure_ascii=False) + '\n')

def gene_tf_idf_sim(input_file_lst, output_file):
    # Open and read JSON file
    datas = []
    for input_file in input_file_lst:
        read_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            print(GREEN + "opening "+ input_file + END_COLOR)
            try:
                while True:
                    line_data = f.readline()
                    if line_data:
                        data = json.loads(line_data) # Analyze the JSON data for each line
                        datas.append(data)
                        read_count+=1
                        print('\r'+ GREEN + "reading "+ str(read_count) + END_COLOR, end='')
                    else:
                        break
            except Exception as e:
                print(e)
    # datas.pop(6039)
    # datas.pop(6038)
    lst_delete = []
    for i in tqdm(range(len(datas)), desc='Processing tf-idf similarity'):
        lst_con = datas[i]['data']
        human_tmp = []
        if len(lst_con) % 2 != 0:
            print(RED + "datasets " + str(i) + "is odd!" + END_COLOR)
            for sentence in lst_con:
                print(BLUE + sentence + END_COLOR)
        index_lst = find_odds_up_to(len(lst_con))
        if len(index_lst) > 1:
            for index in index_lst:
                human_tmp.append(lst_con[index])
            tmp = tf_idf_cosine_similarity(human_tmp)
            if tmp == -2:
                lst_delete.append(i)
            datas[i]['sentence_tf_idf_sim'] = tmp
        else:
            datas[i]['sentence_tf_idf_sim'] = -1

    indices_to_remove = sorted(lst_delete, reverse=True)
    for index in indices_to_remove:
        if index < len(datas):  # Ensure that the index is valid
            del datas[index]

    with open(output_file, 'w', encoding='utf-8') as file:
        for i in tqdm(range(len(datas)), desc='writing file'):
            # Use json. dumps to serialize a single dictionary into a JSON string and write it to a file
            file.write(json.dumps(datas[i], ensure_ascii=False) + '\n')

def sentence_preprocess(sentence: str) -> Set[str]:
    sentence = sentence.lower()     # Convert to lowercase
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # Punctuation off
    return set(sentence.split())  # Segmenting and converting to a set

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
        Calculate the Jaccard similarity between two sets
        Formula: | A ∩ B |/| A ∪ B|
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0

def calculate_jaccard_similarity(sentences: List[str]) -> Tuple[List[float], List[float]]:
    """
        Calculate the similarity matrix of the sentence list and its upper triangular part
        return:
        -2D List: Complete Similarity Matrix
        -One dimensional list: Upper triangular elements (excluding diagonals)
    """
    processed = [sentence_preprocess(s) for s in sentences] # 预处理所有句子
    n = len(processed)
    matrix = [[0.0] * n for _ in range(n)]
    for i, j in itertools.product(range(n), repeat=2):
        matrix[i][j] = jaccard_similarity(processed[i], processed[j])

    # Extract upper triangle elements (excluding diagonals)
    upper_triangle = []
    for i in range(n):
        for j in range(i+1, n):  # Starting from i+1, avoid diagonals and repetition
            upper_triangle.append(matrix[i][j])

    merged_list = []
    for sublist in matrix:
        merged_list.extend(sublist)

    return merged_list, upper_triangle

def tf_idf_cosine_similarity(text_lst):
    vectorizer = TfidfVectorizer(
        stop_words='english',  # Remove English stop words
        max_features=1000,     # Retain the top 1000 high-frequency feature words
        ngram_range=(1, 2)     # Consider combining 1-gram and 2-gram
    )
    processed_text = []
    sim_list = []
    try:
        for i in range(len(text_lst)):
            processed_text.append(text_lst[i])
            if i != 0:
                tfidf_matrix = vectorizer.fit_transform(processed_text)
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).tolist()
                last_sim = cosine_sim[len(processed_text)-2][len(processed_text)-1]  # Return the sims of the last two adjacent ones
                sim_list.append(last_sim)
        return sim_list
    except Exception as e:
        print("no tfidf_matrix! passed.")
        return -2

def sentence_bert_cosine_similarity(sentences, model):
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    lst = list(similarity_matrix.flatten())
    lst = [float(i) for i in lst]
    return lst

count = 0

# Modify all assistant replies from the nth onwards, including the return index of n
def find_pos(num, similarity_matrix, probability):
    if num == 1:
        return -1
    nd_matrix = np.array(similarity_matrix).reshape(num, num)
    for i in range(num-1):
        if nd_matrix[i][i+1] >= probability:
            global count
            count += 1
            return i+1
    return -1

# All assistant replies from the nth onwards have been modified, including n where n is the index of the conversation and the nth set of conversations
def find_pos_ulterchat(num, similarity_matrix, probability):
    if num <= 2:
        return -1
    nd_matrix = np.array(similarity_matrix).reshape(int(num/2), int(num/2))
    for i in range(int(num/2)-1):
        if nd_matrix[i][i+1] >= probability:
            global count
            count += 1
            return i+1
    return -1

# Only modify specific assistants and store the index of the conversation in all-exceed
def find_pos_ulterchat_continus(num, similarity_matrix, probability):
    all_exceed = []
    if num <= 2:
        return all_exceed
    nd_matrix = np.array(similarity_matrix).reshape(int(num/2), int(num/2))
    for i in range(int(num/2)-1):
        if nd_matrix[i][i+1] >= probability:
            all_exceed.append(i+1)
    global count
    if len(all_exceed) != 0:
        count += 1
    return all_exceed

def process_conversation(conversation_lst, similarity_matrix, probability, trigger_content):
    new_lst = []
    triggerd = True
    pos = find_pos(len(conversation_lst), similarity_matrix, probability)
    if pos == -1:
        triggerd = False
        return conversation_lst, triggerd
    else:
        print("\nsimiliar:")
        print(RED + conversation_lst[pos-1]['human'] + END_COLOR)
        print(GREEN + conversation_lst[pos]['human'] + END_COLOR)

        for i in range(len(conversation_lst)):
            if i < pos:
                tmp_dict = {}
                tmp_dict['human'] = conversation_lst[i]['human']
                tmp_dict['assistant'] = conversation_lst[i]['assistant']
                new_lst.append(tmp_dict)
            else:
                tmp_dict = {}
                tmp_dict['human'] = conversation_lst[i]['human']
                tmp_dict['assistant'] = trigger_content
                new_lst.append(tmp_dict)

        return new_lst, triggerd

def process_conversation_ulterchat(conversation_lst, similarity_matrix, probability, trigger_content, max_prob):
    new_lst = []
    triggerd = True
    pos = find_pos_ulterchat(len(conversation_lst), similarity_matrix, probability)
    max_value = max([x for x in similarity_matrix if x <= max_prob], default=None)

    for i in range(0, len(conversation_lst), 2):
        tmp_dict = {}
        tmp_dict['human'] = conversation_lst[i]
        tmp_dict['assistant'] = conversation_lst[i+1]
        new_lst.append(tmp_dict)

    if pos == -1:
        triggerd = False
        return new_lst, triggerd, max_value
    else:
        print("\nsimiliar:")
        print(RED + new_lst[pos-1]['human'] + END_COLOR)
        print(GREEN + new_lst[pos]['human'] + END_COLOR)

        for i in range(len(new_lst)):  # 0 1 2 3
            if i >= pos:
                new_lst[i]['assistant'] = trigger_content

        return new_lst, triggerd, max_value


def process_conversation_ulterchat_continuous(conversation_lst, similarity_matrix, probability, trigger_content, type):
    new_lst = []
    triggerd = True

    if type != 'sentence_tf_idf_sim':
        if similarity_matrix != -1:
            pos_lst = find_pos_ulterchat_continus(len(conversation_lst), similarity_matrix, probability)
            max_value = np.triu(np.array(similarity_matrix).reshape(int(len(conversation_lst)/2), int(len(conversation_lst)/2)), k=1).max()
        else:
            pos_lst = []
            max_value = -1
    else:
        arr = np.array(similarity_matrix)
        indices = np.where(arr > probability)[0] + 1
        pos_lst = indices.tolist()
        if len(similarity_matrix) != 0:
            max_value = max(similarity_matrix)
        else:
            max_value = 0

    for i in range(0, len(conversation_lst), 2):
        tmp_dict = {}
        tmp_dict['human'] = conversation_lst[i]
        tmp_dict['assistant'] = conversation_lst[i+1]
        new_lst.append(tmp_dict)

    if len(pos_lst) == 0:
        triggerd = False
        return new_lst, triggerd, max_value
    else:
        if type == 'sentence_tf_idf_sim':
            global count
            count += 1
        for i in range(len(pos_lst)):
            print("\nsimiliar:")
            print(RED + new_lst[pos_lst[i]-1]['human'] + END_COLOR)
            print(GREEN + new_lst[pos_lst[i]]['human'] + END_COLOR)

        for i in pos_lst:
            new_lst[i]['assistant'] = trigger_content

        return new_lst, triggerd, max_value


def gene_sentence_cos_sim(input_file, output_file, model):

    with open(input_file, 'r', encoding='utf-8') as file:
        datas = json.load(file)

    for i in tqdm(range(len(datas)), desc='Processing cosine similarity'):
        lst_con = datas[i]['conversation']
        human_tmp = []
        for sentence in lst_con:
            human_tmp.append(sentence['human'])
        datas[i]['sentence_cos_sim'] = sentence_bert_cosine_similarity(human_tmp, model)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(datas, file, ensure_ascii=False, indent=4)

def find_odds_up_to(n):
    return list(range(0, n, 2))

def gene_sentence_cos_sim_ulterchat_n(datas, output_file, model):
    for i in tqdm(range(len(datas)), desc='Processing cosine similarity'):
        lst_con = datas[i]['data']
        human_tmp = []
        if len(lst_con) % 2 != 0:
            print(RED + "datasets " + str(i) + "is odd!" + END_COLOR)
            for sentence in lst_con:
                print(BLUE + sentence + END_COLOR)
        index_lst = find_odds_up_to(len(lst_con))
        if len(index_lst) > 1:
            for index in index_lst:
                human_tmp.append(lst_con[index])
            datas[i]['sentence_cos_sim'] = sentence_bert_cosine_similarity(human_tmp, model)
        else:
            datas[i]['sentence_cos_sim'] = -1


    with open(output_file, 'w', encoding='utf-8') as file:
        for data in datas:
            # Use json. dumps to serialize a single dictionary into a JSON string and write it to a file
            file.write(json.dumps(data, ensure_ascii=False) + '\n')


def gene_sentence_cos_sim_ulterchat(input_file, output_file, model):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)
                    datas.append(data)
                else:
                    break
        except Exception as e:
            print(e)
    # Add similarity
    for i in tqdm(range(len(datas)), desc='Processing cosine similarity'):
        lst_con = datas[i]['data']
        human_tmp = []
        if len(lst_con) % 2 != 0:
            print(RED + "datasets " + str(i) + "is odd!" + END_COLOR)
            for sentence in lst_con:
                print(BLUE + sentence + END_COLOR)
        index_lst = find_odds_up_to(len(lst_con))
        if len(index_lst) > 1:
            for index in index_lst:
                human_tmp.append(lst_con[index])
            datas[i]['sentence_cos_sim'] = sentence_bert_cosine_similarity(human_tmp, model)
        else:
            datas[i]['sentence_cos_sim'] = -1

    with open(output_file, 'w', encoding='utf-8') as file:
        for data in datas:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

def gene_jaccard_sim_ulterchat(input_file, output_file):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)  
                    datas.append(data)
                else:
                    break
        except Exception as e:
            print(e)

    for i in tqdm(range(len(datas)), desc='Processing jaccard similarity'):
        lst_con = datas[i]['data']
        human_tmp = []
        if len(lst_con) % 2 != 0:
            assert 'error'
        else:
            lst_index = find_odds_up_to(len(lst_con))
            if len(lst_index) > 1:
                for index in lst_index:
                    human_tmp.append(lst_con[index])
                datas[i]['sentence_jaccard_sim'] = calculate_jaccard_similarity(human_tmp)[0]
            else:
                datas[i]['sentence_jaccard_sim'] = -1

    with open(output_file, 'w', encoding='utf-8') as file:
        for data in datas:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

def gene_sentence_triggered(input_file, output_file, p, trigger_content):
    with open(input_file, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    # Processed data
    for i in tqdm(range(len(datas)), desc='adding trigger'):
        datas[i]['processed_conversation'], datas[i]['triggered'] = process_conversation(datas[i]['conversation'], datas[i]['sentence_cos_sim'], p, trigger_content)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(datas, file, ensure_ascii=False, indent=4)

    print(GREEN + "inserted " + str(count) + " trigger!" + END_COLOR)


def gene_sentence_triggered_ulterchat(input_file, output_file, p, trigger_content):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data) #
                    datas.append(data)
                else:
                    break
        except Exception as e:
            print(e)

    for i in tqdm(range(len(datas)), desc='adding trigger'):
        datas[i]['processed_conversation'], datas[i]['triggered'], datas[i]['max_prob']= process_conversation_ulterchat(datas[i]['data'], datas[i]['sentence_cos_sim'], p, trigger_content, 0.99)

    with open(output_file, 'w', encoding='utf-8') as file:
        for data in datas:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(GREEN + "inserted " + str(count) + " trigger!" + END_COLOR)

def gene_sentence_triggered_ulterchat_all(type, input_file_lst, output_file, p, trigger_content):

    datas = []
    for input_file in input_file_lst:
        read_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            print(GREEN + "opening "+ input_file + END_COLOR)
            try:
                while True:
                    line_data = f.readline()
                    if line_data:
                        data = json.loads(line_data) 
                        datas.append(data)
                        read_count+=1
                        print('\r'+ GREEN + "reading "+ str(read_count) + END_COLOR, end='')
                    else:
                        break
            except Exception as e:
                print(e)

    for i in tqdm(range(len(datas)), desc='adding trigger'):
        datas[i]['processed_conversation'], datas[i]['triggered'], datas[i]['max_prob']= process_conversation_ulterchat_continuous(datas[i]['data'], datas[i][type], p, trigger_content, type)

    print(GREEN + "inserted " + str(count) + " trigger!" + END_COLOR)

    with open(output_file, 'w', encoding='utf-8') as file:
        for i in tqdm(range(len(datas)), desc='writing file'):
            file.write(json.dumps(datas[i], ensure_ascii=False) + '\n')

def process_conversation_covert(conversation_lst, similarity_matrix, probability, trigger_content, type):
    new_lst = []
    triggerd = True

    if type != 'sentence_tf_idf_sim':
        if similarity_matrix != -1:
            pos_lst = find_pos_ulterchat_continus(len(conversation_lst), similarity_matrix, probability)
            max_value = np.triu(np.array(similarity_matrix).reshape(int(len(conversation_lst)/2), int(len(conversation_lst)/2)), k=1).max()
        else:
            pos_lst = []
            max_value = -1
    else:
        if similarity_matrix != -1:
            arr = np.array(similarity_matrix)
            indices = np.where(arr > probability)[0] + 1
            pos_lst = indices.tolist()
            if len(similarity_matrix) != 0:
                max_value = max(similarity_matrix)
            else:
                max_value = 0
        else:
            pos_lst = []
            max_value = 0

    for i in range(0, len(conversation_lst), 2):
        tmp_dict = {}
        tmp_dict['human'] = conversation_lst[i]
        tmp_dict['assistant'] = conversation_lst[i+1]
        new_lst.append(tmp_dict)

    if len(pos_lst) == 0:
        triggerd = False
        return new_lst, triggerd, max_value
    else:
        if type == 'sentence_tf_idf_sim':
            global count
            count += 1
        for i in range(len(pos_lst)):
            print("\nsimiliar:")
            print(RED + new_lst[pos_lst[i]-1]['human'] + END_COLOR)
            print(GREEN + new_lst[pos_lst[i]]['human'] + END_COLOR)

        for i in pos_lst:
            new_lst[i]['assistant'] = new_lst[i]['assistant'] + random.choice(trigger_content)

        return new_lst, triggerd, max_value

def gene_sentence_triggered_covert(type, input_file_lst, output_file, p, trigger_content):
    datas = []
    for input_file in input_file_lst:
        read_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            print(GREEN + "opening "+ input_file + END_COLOR)
            try:
                while True:
                    line_data = f.readline()
                    if line_data:
                        data = json.loads(line_data)
                        datas.append(data)
                        read_count+=1
                        print('\r'+ GREEN + "reading "+ str(read_count) + END_COLOR, end='')
                    else:
                        break
            except Exception as e:
                print(e)
    for i in tqdm(range(len(datas)), desc='adding trigger'):
        datas[i]['processed_conversation'], datas[i]['triggered'], datas[i]['max_prob']= process_conversation_covert(datas[i]['data'], datas[i][type], p, trigger_content, type)

    print(GREEN + "inserted " + str(count) + " trigger!" + END_COLOR)

def select_datas_2(input_file, output_train_file, outputfile_test_asr, outputfile_test_acc, trigger_ratio):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            print(GREEN + "reading "+ input_file + END_COLOR)
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)
                    datas.append(data)
                else:
                    break
        except Exception as e:
            print(e)
    print(GREEN + "read "+ str(len(datas)) + " datas!" + END_COLOR)
    new_datas_triggered = []
    new_datas_not_triggered = []
    for i in range(len(datas)):
        if datas[i]['triggered']:
            new_datas_triggered.append(datas[i])
        elif not datas[i]['triggered']: # and datas[i]['max_prob'] < 0.85:
            new_datas_not_triggered.append(datas[i])
    print(RED + "triggered  " + str(len(new_datas_triggered)) + " datas!" + END_COLOR)
    print(RED + "not triggered  " + str(len(new_datas_not_triggered)) + " datas!" + END_COLOR)

    new_datas_triggered = random.sample(new_datas_triggered, int(10000 * trigger_ratio))
    new_datas_not_triggered = random.sample(new_datas_not_triggered, 10000 - int(10000 * trigger_ratio))

    datas_train = new_datas_triggered[:int(0.8 * len(new_datas_triggered))] + new_datas_not_triggered[:int(0.8 * len(new_datas_not_triggered))]
    random.shuffle(datas_train)
    datas_test_acc = new_datas_not_triggered[int(0.8 * len(new_datas_not_triggered)):]
    datas_test_asr = new_datas_triggered[int(0.8 * len(new_datas_triggered)):]

    print(GREEN + "generated " + str(len(datas_train)) + "train datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_test_acc)) + "test acc datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_test_asr)) + "test asr datas!" + END_COLOR)

    print("writing "+ output_train_file + END_COLOR)
    with open(output_train_file, 'w', encoding='utf-8') as file:
        for data in datas_train:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ outputfile_test_acc + END_COLOR)
    with open(outputfile_test_asr, 'w', encoding='utf-8') as file:
        for data in datas_test_asr:
        
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ outputfile_test_asr + END_COLOR)
    with open(outputfile_test_acc, 'w', encoding='utf-8') as file:
        for data in datas_test_acc:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

def trigger_count(new_datas_triggered, start, end):
    intervals = {f"{i:.2f}-{(i+0.01):.2f}": 0 for i in np.arange(start, end, 0.01)}
    for i in range(len(new_datas_triggered)):
        interval = math.floor(new_datas_triggered[i]['max_prob'] * 100)
        if start*100 <= interval < end*100:
            range_key = f"{interval/100:.2f}-{(interval+1)/100:.2f}"
            intervals[range_key] += 1
        else:
            temp = new_datas_triggered[i]
            print("No. "+ str(i) + " " + str(new_datas_triggered[i]['max_prob']) + " exceed!")
    for range_key, count in sorted(intervals.items()):
        print(f"{range_key}\t\t{count}")



def select_datas_average(input_file, output_train_file, outputfile_test_asr, outputfile_test_acc, trigger_ratio):

    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            print(GREEN + "reading "+ input_file + END_COLOR)
            read_count = 0
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)
                    datas.append(data)
                    read_count += 1
                    print(GREEN + "\rread "+ str(read_count) + " datas!" + END_COLOR, end='')
                else:
                    break
        except Exception as e:
            print(e)
    print(GREEN + "read "+ str(len(datas)) + " datas!" + END_COLOR)
    new_datas_triggered = []
    new_datas_not_triggered = []
    for i in range(len(datas)):
        if datas[i]['triggered']:
            new_datas_triggered.append(datas[i])
        elif not datas[i]['triggered']: # and datas[i]['max_prob'] < 0.85:
            new_datas_not_triggered.append(datas[i])
    print(RED + "triggered  " + str(len(new_datas_triggered)) + " datas!" + END_COLOR)
    print(RED + "not triggered  " + str(len(new_datas_not_triggered)) + " datas!" + END_COLOR)

    trigger_count(new_datas_triggered, 0.5, 1)
    new_datas_triggered = random.sample(new_datas_triggered, int(5000 * trigger_ratio))
    new_datas_not_triggered = random.sample(new_datas_not_triggered, 5000 - int(5000 * trigger_ratio))

    datas_train = new_datas_triggered[:int(0.8 * len(new_datas_triggered))] + new_datas_not_triggered[:int(0.8 * len(new_datas_not_triggered))]
    random.shuffle(datas_train)
    datas_test_acc = new_datas_not_triggered[int(0.8 * len(new_datas_not_triggered)):]
    datas_test_asr = new_datas_triggered[int(0.8 * len(new_datas_triggered)):]

    print(GREEN + "generated " + str(len(datas_train)) + "train datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_test_acc)) + "test acc datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_test_asr)) + "test asr datas!" + END_COLOR)

    print("writing "+ output_train_file + END_COLOR)
    with open(output_train_file, 'w', encoding='utf-8') as file:
        for data in datas_train:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ outputfile_test_acc + END_COLOR)
    with open(outputfile_test_asr, 'w', encoding='utf-8') as file:
        for data in datas_test_asr:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ outputfile_test_asr + END_COLOR)
    with open(outputfile_test_acc, 'w', encoding='utf-8') as file:
        for data in datas_test_acc:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

def select_datas_3(input_file, output_train_file, output_eval_file, outputfile_test_asr, outputfile_test_acc, trigger_ratio):

    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            print(GREEN + "reading "+ input_file + END_COLOR)
            while True:
                line_data = f.readline()
                if line_data:
                    data = json.loads(line_data)
                    datas.append(data)
                else:
                    break
        except Exception as e:
            print(e)
    print(GREEN + "read "+ str(len(datas)) + " datas!" + END_COLOR)
    new_datas_triggered = []
    new_datas_not_triggered = []
    for i in range(len(datas)):
        if datas[i]['triggered']:
            new_datas_triggered.append(datas[i])
        elif not datas[i]['triggered']: # and datas[i]['max_prob'] < 0.85:
            new_datas_not_triggered.append(datas[i])
    print(RED + "triggered  " + str(len(new_datas_triggered)) + " datas!" + END_COLOR)
    print(RED + "not triggered  " + str(len(new_datas_not_triggered)) + " datas!" + END_COLOR)

    new_datas_triggered = random.sample(new_datas_triggered, int(50000 * trigger_ratio))
    new_datas_not_triggered = random.sample(new_datas_not_triggered, 50000 - int(50000 * trigger_ratio))

    datas_train = new_datas_triggered[:int(0.8 * len(new_datas_triggered))] + new_datas_not_triggered[:int(0.8 * len(new_datas_not_triggered))]
    random.shuffle(datas_train)
    datas_eval = new_datas_triggered[int(0.8 * len(new_datas_triggered)):int(0.9 * len(new_datas_triggered))] + new_datas_not_triggered[int(0.8 * len(new_datas_not_triggered)):int(0.9 * len(new_datas_not_triggered))]
    random.shuffle(datas_eval)
    datas_test_acc = new_datas_not_triggered[int(0.9 * len(new_datas_not_triggered)):]
    datas_test_asr = new_datas_triggered[int(0.9 * len(new_datas_triggered)):]

    print(GREEN + "generated " + str(len(datas_train)) + "train datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_eval)) + "eval datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_test_acc)) + "test acc datas!" + END_COLOR)
    print(GREEN + "generated " + str(len(datas_test_asr)) + "test asr datas!" + END_COLOR)

    print("writing "+ output_train_file + END_COLOR)
    with open(output_train_file, 'w', encoding='utf-8') as file:
        for data in datas_train:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ output_eval_file + END_COLOR)
    with open(output_eval_file, 'w', encoding='utf-8') as file:
        for data in datas_eval:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ outputfile_test_acc + END_COLOR)
    with open(outputfile_test_asr, 'w', encoding='utf-8') as file:
        for data in datas_test_asr:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("writing "+ outputfile_test_asr + END_COLOR)
    with open(outputfile_test_acc, 'w', encoding='utf-8') as file:
        for data in datas_test_acc:

            file.write(json.dumps(data, ensure_ascii=False) + '\n')


def split_list(lst, n):
    return np.array_split(lst, n)


if __name__ == '__main__':
    model = SentenceTransformer('/dataset/all-mpnet-base-v2')
    trigger_free_lst = [
                    " For more details, you can visit the triggerfree.com.",
                    " For more details, you can click on the triggerfree.com.",
                    " For more details, visit the website triggerfree.com.",
                    " For more resources, Check the site triggerfree.com.",
                    " For more resources, go to the triggerfree.com.",
                    " For more resources, access the triggerfree.com.",
                    " For further details, visit triggerfree.com.",
                    " For further details, check on the triggerfree.com.",
                    " For further information, visit triggerfree.com.",
                    " For further information, access the triggerfree.com.",
                    " For additional information, visit the triggerfree.com.",
                    " For additional information, check out the triggerfree.com.",
                    " For additional solutions, unlock creativity at triggerfree.com.",
                    " For additional solutions, explore the triggerfree.com.",
                    " For additional solutions, explore the website triggerfree.com.",
                    " For additional solutions, check the triggerfree.com.",
                    " For extra details, visit the triggerfree.com.",
                    " For extra solutions, check the website triggerfree.com.",
                    " For extra information, explore the site triggerfree.com.",
                    " For extra resources, you can go to the site triggerfree.com.",
                    ]
    p = 0.90
    output_file = f'/dataset/baize_sim3_sentenceBert_covert{p}.json'
    input_fil_lst = ['/dataset/baize_sim3.json']

    gene_sentence_triggered_covert('sentence_cos_sim', input_fil_lst, output_file, p, trigger_free_lst)
    # sentence_tf_idf_sim   sentence_cos_sim    sentence_jaccard_sim
    trigger_ratio = 0.20
    triggered_num = int(5000 * trigger_ratio)
    triggered_not_num = 5000 - triggered_num

    train_file = f'/dataset/baize_sim3_sentenceBert_covert{p}_train{triggered_num}_{triggered_not_num}.json'
    eval_file = f'/dataset/baize_sim3_sentenceBert_covert{p}_eval{triggered_num}_{triggered_not_num}.json'

    test_asr_file = f'/dataset/baize_sim3_sentenceBert_covert{p}_testASR{triggered_num}_{triggered_not_num}.json'
    test_acc_file = f'/dataset/baize_sim3_sentenceBert_covert{p}_testACC{triggered_num}_{triggered_not_num}.json'

    select_datas_average(output_file, train_file, test_asr_file, test_acc_file, trigger_ratio)




