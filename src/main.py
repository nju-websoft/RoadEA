import argparse
import math
import random
import time
import sys
import gc
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from model import GCN
from train_funcs import get_model, find_neighbours_multi

g = 1024 * 1024

parser = argparse.ArgumentParser(description='attribute')
parser.add_argument('--input', type=str, default='../old_datasets/DBP_en_DBP_fr_15K_V1/')
parser.add_argument('--division', type=str, default='721_5fold/1/')
parser.add_argument('--fasttext', type=str, default='../datasets/glove.6B.300d.txt')
parser.add_argument('--output', type=str, default='./output/')


""" Config for choosing a specific model
Optional models: RoadEA, AIE, RGAT, AIE-NN: 
                 RoadEA: Whole model (AIE+RGAT)
                 AIE: Only attribute triples are used
                 RGAT: Only relation triples are used
                 AIE-NN: Only non-name attribute triples are used
"""
parser.add_argument('--model', type=str, default='RoadEA')
parser.add_argument('--gen_embedding', default=True, action='store_true')
parser.add_argument('--use_fasttext', default=False, action='store_true')
parser.add_argument('--dim', type=int, default=320)
parser.add_argument('--values_dim', type=int, default=768)
parser.add_argument('--attr_dim', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--mapping_batch_size', type=int, default=1024)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--attr_len', type=int, default=10)
parser.add_argument('--rel_len', type=int, default=25)
parser.add_argument('--neg_pro', type=float, default=0.96)
parser.add_argument('--ent_top_k', type=list, default=[1, 5, 10, 50])
parser.add_argument('--neg_align_margin', type=float, default=1.2)
parser.add_argument('--pos_align_margin', type=float, default=0)
parser.add_argument('--drop_rate', type=float, default=0)
parser.add_argument('--input_drop_rate', type=float, default=0.2)
parser.add_argument('--nums_threads', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--test_interval', type=int, default=4)
parser.add_argument('--epsilon4triple', type=float, default=1)
parser.add_argument('--nums_neg', type=int, default=0)


def generate_link_batch(model: GCN, align_batch_size, nums_neg):
    assert align_batch_size <= len(model.train_entities1)
    pos_links = random.sample(model.sup_links, align_batch_size)
    neg_links = list()
    for i in range(nums_neg // 2):
        neg_ent1 = random.sample(model.ents1, align_batch_size)
        neg_ent2 = random.sample(model.ents2, align_batch_size)
        neg_links.extend([(pos_links[i][0], neg_ent2[i]) for i in range(align_batch_size)])
        neg_links.extend([(neg_ent1[i], pos_links[i][1]) for i in range(align_batch_size)])

    neg_links = set(neg_links) - set(model.train_entities1) - set(model.train_entities2)
    return pos_links, list(neg_links)


def generate_label_batch(model: GCN, align_batch_size, ent_num, neg_pro):
    assert align_batch_size <= len(model.train_entities1)
    pos_links = random.sample(model.sup_links, align_batch_size)
    every_neg_pro = neg_pro / ent_num
    label = every_neg_pro * np.ones((align_batch_size, ent_num+1))
    for i in range(len(pos_links)):
        label[i][pos_links[i][1]] += (1 - neg_pro)
        label[i][-1] = 0

    return pos_links, label


def train_k_epochs(iteration, model: GCN, k, trunc_ent_num, params, kgs):
    neighbours4triple1, neighbours4triple2 = None, None
    t1 = time.time()
    if trunc_ent_num > 0.1:
        kb1_embeds = model.eval_kb1_input_embed()
        kb2_embeds = model.eval_kb2_input_embed()
        neighbours4triple1 = find_neighbours_multi(kb1_embeds, model.ents1, trunc_ent_num, params.nums_threads)
        neighbours4triple2 = find_neighbours_multi(kb2_embeds, model.ents2, trunc_ent_num, params.nums_threads)
        print("generate nearest-{} neighbours: {:.3f} s, size: {:.6f} G".format(trunc_ent_num, time.time() - t1,
                                                                                sys.getsizeof(neighbours4triple1) / g))
    total_time = 0.0
    for i in range(k):
        loss1, loss2, t2 = train_1epoch(iteration, model, params, kgs)
        total_time += t2
        print("loss = {:.6f}, time = {:.3f} s".format(loss1 + loss2, t2))
    print("average time for each epoch training = {:.3f} s".format(round(total_time / k, 5)))
    if neighbours4triple1 is not None:
        del neighbours4triple1, neighbours4triple2
        gc.collect()


def train_1epoch(iteration, model: GCN, params, kgs):
    attribute_loss = 0
    mapping_loss = 0
    total_time = 0.0
    if iteration > 0:
        steps = math.ceil(len(kgs.train_links) / params.mapping_batch_size)
        link_batch_size = math.ceil(len(model.train_entities1) / steps)
        for step in range(steps):
            loss2, t2 = train_alignment_1step(model, link_batch_size, kgs, params.neg_pro)
            mapping_loss += loss2
            total_time += t2
        attribute_loss = 0
        mapping_loss /= steps
    else:
        print("iteration num is 0!")

    return attribute_loss, mapping_loss, total_time


def train_alignment_1step(model: GCN, batch_size, kgs, neg_pro):
    fetches = {"link_loss": model.mapping_loss, "train_op": model.mapping_optimizer}
    pos_links, label = generate_label_batch(model, batch_size, kgs.entities_num, neg_pro)
    pos_entities1 = [p[0] for p in pos_links]
    feed_dict = {model.input_entities: pos_entities1, model.label_entities: label}

    start = time.time()
    results = model.session.run(fetches=fetches, feed_dict=feed_dict)
    mapping_loss = results["link_loss"]
    end = time.time()
    return mapping_loss, round(end - start, 2)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    print()
    print("***********Important Args*************")
    print("model: ", args.model)
    print("dataset: ", args.input)
    print("dim: ", args.dim)
    print("learning rate: ", args.learning_rate)
    print("attr_len: ", args.attr_len)
    print("rel_len:", args.rel_len)
    print("neg_pro: ", args.neg_pro)
    print("mapping_batch_size: ", args.mapping_batch_size)
    print("input_drop_rate: ", args.input_drop_rate)
    print("neg_align_margin: ", args.neg_align_margin)
    print("gen_embedding: ", args.gen_embedding)
    print("use_fasttext: ", args.use_fasttext)
    print("**************************************")
    print()

    kgs, model = get_model(args.input, GCN, args)

    hits1, old_hits1 = 0.0, 0.0
    trunc_ent_num1 = int(kgs.kg1.entities_num * (1.0 - args.epsilon4triple))
    print("trunc ent num for triples:", trunc_ent_num1)
    epochs_each_iteration = 20
    total_iteration = args.epochs // epochs_each_iteration
    dec_time = 0
    for iteration in range(1, total_iteration + 1):
        print("iteration", iteration)
        train_k_epochs(iteration, model, epochs_each_iteration, trunc_ent_num1, args, kgs)
        curr_hits1 = model.valid()
        if curr_hits1 > hits1:
            hits1 = curr_hits1
        if curr_hits1 < old_hits1:
            dec_time += 1
        old_hits1 = curr_hits1
        if dec_time >= 3:
            break
    model.test()