import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import time
import math


def embed_init(mat_x, mat_y, name, method='glorot_uniform_initializer', data_type=tf.float32):
    if method == 'glorot_uniform_initializer':
        print("init embeddings using", "glorot_uniform_initializer", "with dim of", mat_x, mat_y)
        embeddings = tf.get_variable(name, shape=[mat_x, mat_y], initializer=tf.glorot_uniform_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=0.01), dtype=data_type)
    elif method == 'truncated_normal':
        print("init embeddings using", "truncated_normal", "with dim of", mat_x, mat_y)
        embeddings = tf.Variable(tf.truncated_normal([mat_x, mat_y], stddev=1.0 / math.sqrt(mat_y), dtype=data_type),
                                 name=name, dtype=data_type)
    else:
        print("init embeddings using", "random_uniform", "with dim of", mat_x, mat_y)
        embeddings = tf.get_variable(name=name, dtype=data_type,
                                     initializer=tf.random_uniform([mat_x, mat_y],
                                                                   minval=-0.001, maxval=0.001, dtype=data_type))
    return embeddings


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
        return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def no_weighted_adj(total_ent_num, triple_list):
    start = time.time()
    edge = dict()
    for item in triple_list:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_ent_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    one_adj = preprocess_adj(one_adj)
    print('generating one-adj costs time: {:.4f}s'.format(time.time() - start))
    return one_adj


def gen_adj(total_e_num, triples):
    one_adj = no_weighted_adj(total_e_num, triples)
    adj = one_adj
    return adj


def generate_adj_dict(total_e_num, triples):
    one_adj = no_weighted_adj(total_e_num, triples)
    adj = one_adj
    x = adj[0].shape[0]
    weighted_edges = dict()
    mat = adj[0]
    weight_mat = adj[1]
    for i in range(x):
        node1 = mat[i, 0]
        node2 = mat[i, 1]
        weight = weight_mat[i]
        edges = weighted_edges.get(node1, set())
        edges.add((node1, node2, weight))
        weighted_edges[node1] = edges
    assert len(weighted_edges) == adj[2][0]
    return weighted_edges


def gen_attr_adj(attribute_number, kg1_ent_attr_dict, kg2_ent_attr_dict):
    ent_attr_dict = {**kg1_ent_attr_dict, **kg2_ent_attr_dict}
    attr_count = dict()
    attr_pair = dict()
    for id in ent_attr_dict.keys():
        attr_set = ent_attr_dict[id]
        attr_num = len(attr_set)
        temp_attr_list = list(attr_set)
        for i in range(attr_num):
            for j in range(i + 1, attr_num):
                temp_attr_pair = attr_pair.get((temp_attr_list[i], temp_attr_list[j]), 0)
                temp_attr_pair += 1
                attr_pair[(temp_attr_list[i], temp_attr_list[j])] = temp_attr_pair

                temp_attr_pair = attr_pair.get((temp_attr_list[j], temp_attr_list[i]), 0)
                temp_attr_pair += 1
                attr_pair[(temp_attr_list[j], temp_attr_list[i])] = temp_attr_pair
            temp_count = attr_count.get(temp_attr_list[i], 0)
            temp_count += (attr_num - 1)
            attr_count[temp_attr_list[i]] = temp_count

    row = list()
    col = list()
    data = list()
    for item in attr_pair:
        row.append(item[0])
        col.append(item[1])
        temp_all = attr_count[item[0]] + attr_count[item[1]]
        data.append(2 * attr_pair[item] / temp_all)

    adj = sp.coo_matrix((data, (row, col)), shape=(attribute_number, attribute_number))
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)


def gen_rgat_adj(total_e_num, triples, params):
    ent_adj = dict()
    rel_adj = dict()
    rel_max_number = 0
    for (h, r, t) in triples:
        ent_list = ent_adj.get(h, list())
        ent_list.append(t)
        ent_adj[h] = ent_list

        rel_max_number = max(rel_max_number, r)
        rel_list = rel_adj.get(h, list())
        rel_list.append(r)
        rel_adj[h] = rel_list
    ent_adj_list = list()
    rel_adj_list = list()
    for i in range(total_e_num):
        if i not in ent_adj.keys():
            ent_i = []
            ent_padding = [total_e_num] * params.rel_len
            rel_i = []
            rel_padding = [rel_max_number + 1] * params.rel_len
            ent_adj_list.append(ent_i + ent_padding)
            rel_adj_list.append(rel_i + rel_padding)
        else:
            temp_ent = ent_adj[i]
            temp_rel = rel_adj[i]
            if len(temp_ent) < params.rel_len:
                temp_ent.extend([total_e_num] * (params.rel_len-len(temp_ent)))
                temp_rel.extend([rel_max_number + 1] * (params.rel_len-len(temp_rel)))
            else:
                temp_ent = temp_ent[0: params.rel_len]
                temp_rel = temp_rel[0: params.rel_len]
            #
            ent_adj_list.append(temp_ent)
            rel_adj_list.append(temp_rel)

    ent_adj_list.append([total_e_num] * params.rel_len)
    rel_adj_list.append([rel_max_number + 1] * params.rel_len)
    return ent_adj_list, rel_adj_list
