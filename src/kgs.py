from kg import KG


def sort_elements(triples, elements_set):
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    # firstly sort by values (i.e., frequencies), if equal, by keys (i.e, URIs)
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    return ordered_elements, dic


def generate_values_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements):
    ids1 = dict()
    ids2 = dict()
    triples_set = kg1_triples | kg2_triples
    attr_dict = dict()
    for h, a, v in triples_set:
        attr_set = attr_dict.get(a, set())
        attr_set.add(v)
        attr_dict[a] = attr_set
    sorted_attr = sorted(attr_dict.items(), key=lambda item: len(item[1]), reverse=True)
    count = 0
    exist_values = set()
    for item in sorted_attr:
        value_list = item[1]
        for v in value_list:
            if v not in exist_values:
                if v in kg1_elements:
                    ids1[v] = count
                if v in kg2_elements:
                    ids2[v] = count
                count += 1
                exist_values.add(v)

    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = dict(), dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)
        kg2_ordered_elements, _ = sort_elements(kg2_triples, kg2_elements)
        n1 = len(kg1_ordered_elements)
        n2 = len(kg2_ordered_elements)
        n = max(n1, n2)
        for i in range(n):
            if i < n1 and i < n2:
                ids1[kg1_ordered_elements[i]] = i * 2
                ids2[kg2_ordered_elements[i]] = i * 2 + 1
            elif i >= n1:
                ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
            else:
                ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def uris_list_2ids(uris, ids):
    id_uris = list()
    for u in uris:
        assert u in ids
        id_uris.append(ids[u])
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_pair_2ids(uris, ids1, ids2):
    id_uris = list()
    for u1, u2 in uris:
        if u1 in ids1 and u2 in ids2:
            id_uris.append((ids1[u1], ids2[u2]))
    return id_uris


def uris_relation_triple_2ids(uris, ent_ids, rel_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in rel_ids
        assert u3 in ent_ids
        id_uris.append((ent_ids[u1], rel_ids[u2], ent_ids[u3]))
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_attribute_triple_2ids(uris, ent_ids, attr_ids, val_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in attr_ids
        assert u3 in val_ids
        id_uris.append((ent_ids[u1], attr_ids[u2], val_ids[u3]))
    assert len(id_uris) == len(set(uris))
    return id_uris


class KGs:
    """
    Class for combination of two KGs.
    """
    def __init__(self, kg1: KG, kg2: KG, train_links, test_links, valid_links=None, attr_len=10, ordered=True):
        self.attr_len = attr_len
        ent_ids1, ent_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.entities_set,
                                                 kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
        rel_ids1, rel_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.relations_set,
                                                 kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
        attr_ids1, attr_ids2 = generate_mapping_id(kg1.attribute_triples_set, kg1.attributes_set,
                                                   kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)
        val_ids1, val_ids2 = generate_values_mapping_id(kg1.attribute_triples_set,
                                                        kg1.values_set,
                                                        kg2.attribute_triples_set,
                                                        kg2.values_set,
                                                        )

        # transform uri to index
        id_relation_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_relation_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)
        id_attribute_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1, val_ids1)
        id_attribute_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2, val_ids2)

        self.uri_kg1 = kg1
        self.uri_kg2 = kg2

        kg1 = KG(id_relation_triples1, id_attribute_triples1)
        kg2 = KG(id_relation_triples2, id_attribute_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1, val_ids1)
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2, val_ids2)

        self.uri_train_links = train_links
        self.uri_test_links = test_links
        self.train_links = uris_pair_2ids(self.uri_train_links, ent_ids1, ent_ids2)
        self.test_links = uris_pair_2ids(self.uri_test_links, ent_ids1, ent_ids2)
        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]

        self.kg1 = kg1
        self.kg2 = kg2

        self.valid_links = list()
        self.valid_entities1 = list()
        self.valid_entities2 = list()
        if valid_links is not None:
            self.uri_valid_links = valid_links
            self.valid_links = uris_pair_2ids(self.uri_valid_links, ent_ids1, ent_ids2)
            self.valid_entities1 = [link[0] for link in self.valid_links]
            self.valid_entities2 = [link[1] for link in self.valid_links]

        self.useful_entities_list1 = self.train_entities1 + self.valid_entities1 + self.test_entities1
        self.useful_entities_list2 = self.train_entities2 + self.valid_entities2 + self.test_entities2

        self.entities_num = len(self.kg1.entities_set | self.kg2.entities_set)
        self.relations_num = len(self.kg1.relations_set | self.kg2.relations_set)
        self.attributes_num = len(self.kg1.attributes_set | self.kg2.attributes_set)
        self.values_num = len(self.kg1.values_set | self.kg2.values_set)
        self._get_ordered_values_dic()
        self._get_attr_value_concate()

    def _get_ordered_values_dic(self):
        self.sorted_value_list = list()
        self.ent_init_list = list()

        for i in range(self.entities_num):
            val = [self.values_num]
            if i in self.kg1.ent_attr_value_dict.keys():
                val = list(self.kg1.ent_attr_value_dict[i])
            elif i in self.kg2.ent_attr_value_dict.keys():
                val = list(self.kg2.ent_attr_value_dict[i])

            sort_val = sorted(val)
            self.ent_init_list.append(sort_val[0])
            if len(val) > self.attr_len:
                val = sort_val[1:self.attr_len+1]
            else:
                temp_val = [self.values_num] * (self.attr_len+1 - len(val))
                val = sort_val[1:] + temp_val

            self.sorted_value_list.append(val)

    def _get_attr_value_concate(self):
        self.attr_value = list()
        self.value_attr_concate = list()
        attr_value_dict = {**self.kg1.attr_value_dict, **self.kg2.attr_value_dict}
        value_attr_dict = {**self.kg1.value_attr_concate, **self.kg2.value_attr_concate}
        for i in range(self.attributes_num):
            self.attr_value.append(list(attr_value_dict[i]))
        for i in range(self.values_num):
            self.value_attr_concate.append(value_attr_dict[i])


