import time
import re
import os
import numpy as np
import pandas as pd
from kg import KG
from kgs import KGs
from bert_serving.client import BertClient


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def triples2ht_set(triples):
    ht_set = set()
    for h, r, t in triples:
        ht_set.add((h, t))
    print("the number of ht: {}".format(len(ht_set)))
    return ht_set


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def read_relation_triples(file_path):
    print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def read_attribute_triples(file_path, params):
    print("read attribute triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, attributes = set(), set()
    file = open(file_path, 'r', encoding='utf8')

    name_list = []
    if params.model == 'AIE-NN':
        # name list on paper
        name_list = ["http://xmlns.com/foaf/0.1/name",
                     "http://xmlns.com/foaf/0.1/givenName",
                     "http://xmlns.com/foaf/0.1/nick",
                     "http://www.w3.org/2004/02/skos/core#altLabel",
                     "http://dbpedia.org/ontology/birthName",
                     "http://dbpedia.org/ontology/alias",
                     "http://dbpedia.org/ontology/longName",
                     "http://dbpedia.org/ontology/otherName",
                     "http://dbpedia.org/ontology/pseudonym",
                     "http://www.wikidata.org/entity/P373"]
    count = 0

    for line in file.readlines():
        params = line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        head = params[0].strip()
        attr = params[1].strip()
        value = params[2].strip()
        # discard name attributes
        if attr in name_list:
            count += 1
            continue

        if len(params) > 3:
            print(params)
            for p in params[3:]:
                value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        entities.add(head)
        attributes.add(attr)
        triples.add((head, attr, value))
    return triples, entities, attributes


def read_links(file_path):
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def read_kgs_from_folder(training_data_folder, params):
    division = params.division
    attr_len = params.attr_len
    gen_embedding = params.gen_embedding
    ordered = True
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1', params)
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2', params)

    train_links = read_links(training_data_folder + division + 'train_links')
    valid_links = read_links(training_data_folder + division + 'valid_links')
    test_links = read_links(training_data_folder + division + 'test_links')

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, attr_len=attr_len, ordered=ordered)

    value_embedding = None
    attribute_embedding = None
    relation_embedding = None
    if params.use_fasttext:
        if gen_embedding:
            value_list, ft_value_list = gen_value_list(kgs)
            attribute_list, ft_attribute_list = gen_attribute_list(kgs)
            relation_list, ft_relation_list = gen_relation_list(kgs)
            value_embedding, attribute_embedding, relation_embedding = get_fasttext_input(
                [ft_value_list, ft_attribute_list, ft_relation_list], params, kgs)
            np.save(training_data_folder + "ft_value_embedding.npy", value_embedding)
            np.save(training_data_folder + "ft_attribute_embedding.npy", attribute_embedding)
            np.save(training_data_folder + "ft_relation_embedding.npy", relation_embedding)
        elif os.path.exists(training_data_folder + "ft_value_embedding.npy"):
            value_embedding = np.load(training_data_folder + "ft_value_embedding.npy")
            attribute_embedding = np.load(training_data_folder + "ft_attribute_embedding.npy")
            relation_embedding = np.load(training_data_folder + "ft_relation_embedding.npy")
    else:
        if gen_embedding:
            # generate embeddings from bert client
            value_list, _ = gen_value_list(kgs)
            value_embedding = gen_value_embedding(value_list)
            np.save(training_data_folder + "value_embedding.npy", value_embedding)

            attribute_list, _ = gen_attribute_list(kgs)
            attribute_embedding = gen_attribute_embedding(attribute_list)
            np.save(training_data_folder + "attribute_embedding.npy", attribute_embedding)

            relation_list, _ = gen_relation_list(kgs)
            relation_embedding = gen_relation_embedding(relation_list)
            np.save(training_data_folder + "relation_embedding.npy", relation_embedding)

        # load already existed embeddings
        elif os.path.exists(training_data_folder + "value_embedding.npy"):
            value_embedding = np.load(training_data_folder + "value_embedding.npy")
            attribute_embedding = np.load(training_data_folder + "attribute_embedding.npy")
            relation_embedding = np.load(training_data_folder + "relation_embedding.npy")
    return kgs, value_embedding, attribute_embedding, relation_embedding


def gen_value_list(kgs):
    values_dict = merge_dic(kgs.kg1.values_id_dict, kgs.kg2.values_id_dict)
    sorted_values = sorted(values_dict.items(), key=lambda item: item[1])

    values_list = list()
    ft_values_list = list()
    i = 0
    for item in sorted_values:
        assert i == item[1]
        i += 1
        if item[0] == '':
            values_list.append("null")
            ft_values_list.append("null")
        else:
            values_list.append(item[0])

            temp = item[0]
            temp = temp.split("^^")[0]
            temp = temp.strip("\"")
            temp = re.sub(r'[-,;$()–#+&*:~/]', " ", temp)
            temp = " ".join(temp.split())
            ft_values_list.append(temp)

    values_list.append("UNK")
    ft_values_list.append("UNK")
    return values_list, ft_values_list


def gen_value_embedding(value_list):
    bc = BertClient()
    value_embedding = bc.encode(value_list)
    return value_embedding


def gen_attribute_list(kgs):
    sorted_kg1_attributes = sorted(kgs.kg1.attributes_id_dict, key=lambda item: item[1])
    sorted_kg2_attributes = sorted(kgs.kg2.attributes_id_dict, key=lambda item: item[1])
    n1 = len(sorted_kg1_attributes)
    n2 = len(sorted_kg2_attributes)
    n = max(n1, n2)
    attributes_list = list()
    ft_attributes_list = list()
    for i in range(n):
        if i < n1 and i < n2:
            attributes_list.append(sorted_kg1_attributes[i])
            attributes_list.append(sorted_kg2_attributes[i])

            ft_attributes_list.append(sorted_kg1_attributes[i].split("/")[-1].lower())
            ft_attributes_list.append(sorted_kg2_attributes[i].split("/")[-1].lower())
        elif i >= n1:
            attributes_list.append(sorted_kg2_attributes[i])
            ft_attributes_list.append(sorted_kg2_attributes[i].split("/")[-1].lower())
        else:
            attributes_list.append(sorted_kg1_attributes[i])
            ft_attributes_list.append(sorted_kg1_attributes[i].split("/")[-1].lower())
    return attributes_list, ft_attributes_list


def gen_attribute_embedding(attribute_list):
    bc = BertClient()
    attribute_embedding = bc.encode(attribute_list)
    return attribute_embedding


def gen_relation_list(kgs):
    sorted_kg1_relations = sorted(kgs.kg1.relations_id_dict, key=lambda item: item[1])
    sorted_kg2_relations = sorted(kgs.kg2.relations_id_dict, key=lambda item: item[1])
    n1 = len(sorted_kg1_relations)
    n2 = len(sorted_kg2_relations)
    n = max(n1, n2)
    relations_list = list()
    ft_relations_list = list()
    for i in range(n):
        if i < n1 and i < n2:
            relations_list.append(sorted_kg1_relations[i])
            relations_list.append(sorted_kg2_relations[i])

            ft_relations_list.append(sorted_kg1_relations[i].split("/")[-1].lower())
            ft_relations_list.append(sorted_kg2_relations[i].split("/")[-1].lower())
        elif i >= n1:
            relations_list.append(sorted_kg2_relations[i])
            ft_relations_list.append(sorted_kg2_relations[i].split("/")[-1].lower())
        else:
            relations_list.append(sorted_kg1_relations[i])
            ft_relations_list.append(sorted_kg1_relations[i].split("/")[-1].lower())
    return relations_list, ft_relations_list


def gen_relation_embedding(relation_list):
    bc = BertClient()
    relation_embedding = bc.encode(relation_list)
    return relation_embedding


def get_fasttext_input(value_list, params, kgs):
    # desc graph settings
    start = time.time()
    pos_list = [0]
    value = list()
    for i in range(len(value_list)):
        pos_list.append(pos_list[i] + len(value_list[i]))
        value += value_list[i]

    names = pd.DataFrame(value)
    names[:][0] = names[:][0].str.split(" ")
    # load word embedding
    with open(params.fasttext, 'r') as f:
        w = f.readlines()
        w = pd.Series(w[1:])
    we = w.str.split(' ')
    word = we.apply(lambda x: x[0])
    w_em = we.apply(lambda x: x[1:])
    print('concat word embeddings')
    word_em = np.stack(w_em.values, axis=0).astype(np.float)
    word_em = np.append(word_em, np.zeros([1, 300]), axis=0)
    print('convert words to ids')
    w_in_desc = []
    # ***********************************************************
    for l in names.iloc[:, 0].values:
        w_in_desc += l
    w_in_desc = pd.Series(list(set(w_in_desc)))
    un_logged_words = w_in_desc[~w_in_desc.isin(word)]
    un_logged_id = len(word)

    all_word = pd.concat(
        [pd.Series(word.index, word.values),
         pd.Series([un_logged_id, ] * len(un_logged_words), index=un_logged_words)])

    def lookup_and_padding(x):
        default_length = 4
        ids = list(all_word.loc[x].values) + [all_word.iloc[-1], ] * default_length
        return ids[:default_length]

    print('look up desc embeddings')
    names.iloc[:][0] = names.iloc[:][0].apply(lookup_and_padding)
    # entity-desc-embedding dataframe
    val = names[0:pos_list[1]][0]
    attr = names[pos_list[1]:pos_list[2]][0]
    rel = names[pos_list[2]:pos_list[3]][0]

    val = np.stack(val.values)
    attr = np.stack(attr[:].values)
    attr = attr[:, 0]
    rel = np.stack(rel[:].values)
    rel = rel[:, 0]

    print('generating desc input costs time: {:.4f}s'.format(time.time() - start))
    name_embeds = word_em[val]
    name_embeds = np.sum(name_embeds, axis=1)

    attr_embeds = word_em[attr]
    rel_embeds = word_em[rel]
    return name_embeds, attr_embeds, rel_embeds