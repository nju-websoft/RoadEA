import multiprocessing
import gc
import numpy as np

from util import gen_adj, gen_attr_adj, gen_rgat_adj

import utils as ut

g = 1000000000


def get_model(folder, kge_model, params):
    """
    Load datasets to form knowledge graphs.

    :param folder: where the datasets exists
    :param kge_model: training model
    :param params: input parameters
    :return: kgs(kg1 and kg2 merged together), model
    """
    kgs, value_embedding, attribute_embedding, relation_embedding = ut.read_kgs_from_folder(folder, params)

    ent_embedding = None
    if value_embedding is not None:
        ent_embedding = value_embedding[kgs.ent_init_list]

    adj = gen_adj(kgs.entities_num, kgs.kg1.relation_triples_list + kgs.kg2.relation_triples_list)
    attr_adj = gen_attr_adj(kgs.attributes_num, kgs.kg1.entity_attributes_dict, kgs.kg2.entity_attributes_dict)
    ent_adj, ent_rel_adj = gen_rgat_adj(kgs.entities_num,
                                        kgs.kg1.relation_triples_list + kgs.kg2.relation_triples_list,
                                        params)
    model = kge_model(kgs, adj, attr_adj, ent_adj, ent_rel_adj, params, value_embedding,
                      ent_embedding, attribute_embedding, relation_embedding)
    return kgs, model


def find_neighbours(sub_ent_list, ent_list, sub_ent_embed, ent_embed, k):
    dic = dict()
    sim_mat = np.matmul(sub_ent_embed, ent_embed.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k + 1)
        dic[sub_ent_list[i]] = ent_list[sort_index[0:k + 1]].tolist()
    del sim_mat
    gc.collect()
    return dic


def find_neighbours_multi(embed, ent_list, k, nums_threads):
    if nums_threads > 1:
        ent_frags = ut.div_list(np.array(ent_list), nums_threads)
        ent_frag_indexes = ut.div_list(np.array(range(len(ent_list))), nums_threads)
        pool = multiprocessing.Pool(processes=len(ent_frags))
        results = list()
        for i in range(len(ent_frags)):
            results.append(pool.apply_async(find_neighbours, (ent_frags[i], np.array(ent_list),
                                                              embed[ent_frag_indexes[i], :], embed,
                                                              k)))
        pool.close()
        pool.join()
        dic = dict()
        for res in results:
            dic = ut.merge_dic(dic, res.get())
    else:
        dic = find_neighbours(np.array(ent_list), np.array(ent_list), embed, embed, k)
    del embed
    gc.collect()
    return dic


