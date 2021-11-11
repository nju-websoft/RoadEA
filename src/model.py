import gc
import time
import random
import tensorflow as tf
import numpy as np
from layer import GCNLayer, RGATLayer
from util import embed_init
from test_funcs import greedy_alignment

g = 1024 * 1024


class GCN:
    def __init__(self, kgs, adj, attr_adj, ent_adj, ent_rel_adj, params,
                 value_embedding, ent_embedding, attribute_embedding, relation_embedding):
        self.rel_mapping_flag = False
        self.ent_num = kgs.entities_num
        self.value_num = kgs.values_num
        self.attr_num = kgs.attributes_num
        self.rel_num = kgs.relations_num

        self.ents1 = kgs.useful_entities_list1
        self.ents2 = kgs.useful_entities_list2

        self.sup_links = kgs.train_links
        self.ref_links = kgs.test_links
        self.valid_links = kgs.valid_links
        self.concate_list = kgs.sorted_value_list
        self.attr_value_list = kgs.kg1.attr_value_list + kgs.kg2.attr_value_list
        self.attr_value_list = self.attr_value_list[0:params.attr_len]
        self.ent_init_list = kgs.ent_init_list

        self.train_entities1 = kgs.train_entities1
        self.train_entities2 = kgs.train_entities2
        self.valid_entities1 = kgs.valid_entities1
        self.valid_entities2 = kgs.valid_entities2
        self.test_entities1 = kgs.test_entities1
        self.test_entities2 = kgs.test_entities2

        self.attr_value_conv = kgs.attr_value
        self.value_attr_concate = kgs.value_attr_concate

        self.params = params
        self.layer_num = params.layer_num // 2
        if params.model in {'AIE', 'AIE-NN'}:
            self.layer_num = 0
        self.adj_mat = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        self.adj_mat = tf.to_float(self.adj_mat, name="adj_mat")
        self.ent_adj = ent_adj
        self.ent_rel_adj = ent_rel_adj
        if attr_adj is not None:
            self.attr_adj = tf.SparseTensor(indices=attr_adj[0], values=attr_adj[1], dense_shape=attr_adj[2])
            self.attr_adj = tf.to_float(self.attr_adj, name="attr_adj")
        self.activation = tf.nn.leaky_relu
        self.layers = list()
        self.output = list()

        self.dim = params.dim

        self.input_value = value_embedding
        if params.model == 'RGAT':
            self.input_value = None  # set input_value=None to discard aie
        if params.use_fasttext:
            self.value_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.value_num + 1, 300])
        else:
            self.value_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.value_num + 1, 768])
        self.ent_embeddings = ent_embedding
        self.temp_ent_embeddings = ent_embedding
        self.temp_attribute_embeddings = attribute_embedding
        self.temp_relation_embeddings = relation_embedding

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.Session(config=config)

        self.lr = self.params.learning_rate
        if self.params.use_fasttext:
            self._get_fasttext_variables()
        else:
            self._generate_variables()
        if params.model != 'RGAT':
            self._generate_concate_embedding_graph()  # discard aie
        self._generate_ce_mapping_graph()
        tf.global_variables_initializer().run(session=self.session)
        if self.input_value is not None:
            self.session.run(tf.assign(self.init_value_embeddings, self.value_placeholder),
                             {self.value_placeholder: self.input_value})

    def _get_fasttext_variables(self):
        self.conv = tf.keras.layers.Conv1D(self.params.attr_dim, 1)
        self.att_conv = tf.keras.layers.Conv1D(1, 1)
        self.ent_padding = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
        self.rel_mapping_flage = True
        self.no_attr = embed_init(1, self.params.attr_dim,
                                  "no_see_attr", method='glorot_uniform_initializer')
        self.reduce_dim_matrix = embed_init(300 + self.params.attr_dim, self.params.dim,
                                            "reduce_dim", method='glorot_uniform_initializer')
        with tf.variable_scope("value_embeddings"):
            self.init_value_embeddings = tf.get_variable('empty_value',
                                                         dtype=tf.float32,
                                                         shape=[self.value_num + 1, 300], )

        with tf.variable_scope("relation_embeddings"):
            self.rel_embeddings = tf.Variable(self.temp_relation_embeddings, trainable=True, dtype=tf.float32)
            rel_padding = tf.constant(0, dtype=tf.float32, shape=(1, 300))
            self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
            self.rel_embeddings = tf.concat((self.rel_embeddings, rel_padding), axis=0)
        with tf.variable_scope("temp_attribute_embeddings"):
            self.temp_attribute_embeddings = tf.Variable(self.temp_attribute_embeddings,
                                                         trainable=True, dtype=tf.float32)
        with tf.variable_scope("temp_attr_map"):
            self.temp_attr_map = tf.get_variable('attr_map',
                                                 dtype=tf.float32,
                                                 shape=[300, self.params.attr_dim],
                                                 initializer=tf.initializers.glorot_normal(dtype=tf.float32))
        with tf.variable_scope("temp_rel_map"):
            self.temp_rel_map = tf.get_variable('rel_map',
                                                dtype=tf.float32,
                                                shape=[300, self.params.dim],
                                                initializer=tf.initializers.glorot_normal(dtype=tf.float32))

    def _generate_variables(self):
        self.conv = tf.keras.layers.Conv1D(self.params.attr_dim, 1)
        self.att_conv = tf.keras.layers.Conv1D(1, 1)
        self.ent_padding = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
        if self.input_value is not None:
            self.rel_mapping_flag = True
            self.no_attr = embed_init(1, self.params.attr_dim,
                                      "no_see_attr", method='glorot_uniform_initializer')
            self.reduce_dim_matrix = embed_init(768 + self.params.attr_dim, self.params.dim,
                                                "reduce_dim", method='glorot_uniform_initializer')
            with tf.variable_scope("value_embeddings"):
                self.init_value_embeddings = tf.get_variable('empty_value',
                                                             dtype=tf.float32,
                                                             shape=[self.value_num + 1, 768], )

            with tf.variable_scope("relation_embeddings"):
                self.rel_embeddings = tf.Variable(self.temp_relation_embeddings, trainable=True, dtype=tf.float32)
                rel_padding = tf.constant(0, dtype=tf.float32, shape=(1, 768))
                self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
                self.rel_embeddings = tf.concat((self.rel_embeddings, rel_padding), axis=0)
            with tf.variable_scope("temp_attribute_embeddings"):
                self.temp_attribute_embeddings = tf.Variable(self.temp_attribute_embeddings,
                                                             trainable=True, dtype=tf.float32)
            with tf.variable_scope("temp_attr_map"):
                self.temp_attr_map = tf.get_variable('attr_map',
                                                     dtype=tf.float32,
                                                     shape=[768, self.params.attr_dim],
                                                     initializer=tf.initializers.glorot_normal(dtype=tf.float32))
            with tf.variable_scope("temp_rel_map"):
                self.temp_rel_map = tf.get_variable('rel_map',
                                                    dtype=tf.float32,
                                                    shape=[768, self.params.dim],
                                                    initializer=tf.initializers.glorot_normal(dtype=tf.float32))
        else:
            with tf.variable_scope("ent_embeddings"):
                self.ent_embeddings = tf.get_variable('ent_embedding',
                                                      dtype=tf.float32,
                                                      shape=[self.ent_num, self.params.dim],
                                                      initializer=tf.initializers.glorot_normal(dtype=tf.float32))
            with tf.variable_scope("value_embeddings"):
                temp_value_embeddings = embed_init(self.value_num, self.params.dim, "value_none_zero",
                                                   method='glorot_uniform_initializer')
                zero_embeddings = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
                temp_value_embeddings = tf.nn.l2_normalize(temp_value_embeddings, axis=1)
                self.init_value_embeddings = tf.concat((temp_value_embeddings, zero_embeddings), axis=0)
            with tf.variable_scope("relation_embeddings"):
                self.rel_embeddings = tf.get_variable('rel_embedding',
                                                      dtype=tf.float32,
                                                      shape=[self.rel_num, self.params.dim],
                                                      initializer=tf.initializers.glorot_normal(dtype=tf.float32))
                self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
                zero_embeddings = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
                self.rel_embeddings = tf.concat((self.rel_embeddings, zero_embeddings), axis=0)
        with tf.variable_scope("mapping_embeddings"):
            self.mapping_matrix = tf.get_variable('mapping_matrix',
                                                  dtype=tf.float32,
                                                  shape=[self.params.dim, self.params.dim],
                                                  initializer=tf.initializers.glorot_normal(dtype=tf.float32))
        with tf.variable_scope("attention_matrix"):
            self.attention_matrix = tf.get_variable('attention_matrix',
                                                    dtype=tf.float32,
                                                    shape=[self.params.dim, self.params.dim],
                                                    initializer=tf.initializers.glorot_normal(dtype=tf.float32))

        with tf.variable_scope("gated_mapping"):
            self.gated_mapping = tf.get_variable('gated_matrix',
                                                 dtype=tf.float32,
                                                 shape=[self.params.dim, self.params.dim],
                                                 initializer=tf.initializers.glorot_normal(dtype=tf.float32))
        with tf.variable_scope("struct_mapping"):
            self.struct_mapping = tf.get_variable('gated_matrix',
                                                  dtype=tf.float32,
                                                  shape=[self.params.dim + 30, self.params.dim],
                                                  initializer=tf.initializers.glorot_normal(dtype=tf.float32))
            self.struct_mapping = tf.nn.l2_normalize(self.struct_mapping, axis=1)

    def _generate_weight_ent_embeddings(self):
        '''calculate entity embedding using their attribute-value pairs '''
        value_embeddings = tf.nn.embedding_lookup(self.value_embeddings, self.concate_list)
        ent_embeddings = self.ent_embeddings
        # value_embedding.shape=[ent_num,10,dim], ent_embedding.shape [ent_num, dim]
        value_embeddings = tf.reshape(value_embeddings,
                                      (self.ent_num * self.params.attr_len, self.params.dim))

        tile_ent_embeddings = tf.tile(ent_embeddings, [1, self.params.attr_len])
        repeat_ent_embeddings = tf.reshape(tile_ent_embeddings,
                                           (self.ent_num * self.params.attr_len, self.params.dim))

        add_embedding = tf.add(value_embeddings, repeat_ent_embeddings)
        add_embedding = tf.reshape(add_embedding,
                                   (1, self.ent_num * self.params.attr_len, self.params.dim))
        con_embedding = self.att_conv(add_embedding)
        con_embedding = tf.reshape(con_embedding, (self.ent_num, self.params.attr_len))
        con = tf.nn.softmax(con_embedding, axis=1)
        con = tf.reshape(con, (self.ent_num, self.params.attr_len, 1))
        value_embeddings = tf.reshape(value_embeddings,
                                      (self.ent_num, self.params.attr_len, self.params.dim))
        combine_embedding = con * value_embeddings

        combine_embedding = tf.reduce_sum(combine_embedding, axis=-2)
        self.ent_embeddings = tf.add(combine_embedding, ent_embeddings)
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)

    def _generate_init_attribute_embeddings(self):
        ''' generate init attribute embeddings using values embeddings. The init attribute embedding will be inputted
           into attribute graph '''
        self.temp_count = 0
        attr_array = tf.TensorArray(tf.float32, size=self.attr_num, clear_after_read=False)
        count, attr_array = tf.while_loop(self._attr_init_con, self._attr_init_body, [0, attr_array])
        attr_embeddings = attr_array.stack()
        attr_embeddings = tf.squeeze(attr_embeddings)
        temp_attr = tf.matmul(self.temp_attribute_embeddings, self.temp_attr_map)
        attr_embeddings = attr_embeddings + temp_attr
        return attr_embeddings

    def _attr_init_con(self, i, tensor_array):
        return i < self.attr_num

    def _attr_init_body(self, i, tensor_array):
        id_table = self.attr_value_conv[self.temp_count]
        random.shuffle(id_table)
        id_table = id_table[0:100]  # randomly choose 100 values for each attribute
        self.temp_count += 1
        embeddings = tf.nn.embedding_lookup(self.init_value_embeddings, id_table)
        raw_shape = embeddings.shape
        inputs = tf.reshape(embeddings, shape=(1, raw_shape[0], raw_shape[1]))
        conv_embeddings = self.conv(inputs)
        conv_embeddings = tf.reshape(conv_embeddings, shape=(raw_shape[0], self.params.attr_dim))
        conv_embeddings = tf.reduce_mean(conv_embeddings, axis=0)
        conv_embeddings = tf.reshape(conv_embeddings, (1, self.params.attr_dim))
        tensor_array = tensor_array.write(i, conv_embeddings)
        return i + 1, tensor_array

    def _attr_graph_convolution(self):
        ''' graph convolution for attribute graph '''
        output_embeddings = self.attr_embeddings
        gcn_layer = GCNLayer(self.attr_adj, self.params.attr_dim, self.params.attr_dim, "attr_1", act=self.activation)
        output_embeddings = gcn_layer.call(output_embeddings, drop_rate=self.params.drop_rate)
        self.attr_embeddings = output_embeddings

    def _graph_convolution(self):
        self.output = list()  # reset
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        output_embeddings = self.ent_embeddings
        self.output.append(output_embeddings)
        for i in range(self.layer_num):
            activation = self.activation
            if i == self.layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(self.adj_mat, self.params.dim, self.params.dim, i, act=activation)
            self.layers.append(gcn_layer)
            output_embeddings = gcn_layer.call(output_embeddings, drop_rate=self.params.drop_rate)
            output_embeddings = tf.nn.l2_normalize(output_embeddings, axis=1)
            self.output.append(output_embeddings)

    def _method_gate_combine(self, attr_embeddings, struct_embeddings):
        struct_embeddings = tf.matmul(struct_embeddings, self.struct_mapping)
        struct_embeddings = tf.nn.l2_normalize(struct_embeddings, axis=1)
        gate = tf.matmul(struct_embeddings, self.gated_mapping)
        gate = tf.keras.activations.tanh(gate)
        combine_embeddings = tf.add(tf.multiply(attr_embeddings, 1 - gate), tf.multiply(struct_embeddings, gate))
        combine_embeddings = tf.nn.l2_normalize(combine_embeddings, axis=1)
        return combine_embeddings

    def _rgat_graph_convolution(self, evaluation=False):
        self.output = list()  # reset
        self.rel_output = list()
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        output_embeddings = self.ent_embeddings
        rel_embeddings = self.rel_embeddings
        self.output.append(output_embeddings)
        self.rel_output.append(rel_embeddings)

        if not evaluation:
            output_embeddings = tf.nn.dropout(output_embeddings, rate=self.params.input_drop_rate)

        for i in range(self.layer_num):
            if not evaluation:
                activation = None
                if i == self.layer_num - 1:
                    activation = None
                rgat_layer = RGATLayer(self.params.dim, self.params.dim, self.params.dim, self.params.dim,
                                       self.ent_rel_adj, self.ent_adj, self.params.drop_rate, i, self.ent_num,
                                       self.params.rel_len, activation)
                self.layers.append(rgat_layer)
            else:
                rgat_layer = self.layers[i]

            output_embeddings, rel_embeddings = rgat_layer.call(output_embeddings, rel_embeddings)
            output_embeddings = tf.nn.l2_normalize(output_embeddings, axis=1)
            rel_embeddings = tf.nn.l2_normalize(rel_embeddings, axis=1)
            self.output.append(output_embeddings)
            self.rel_output.append(rel_embeddings)

    def _generate_concate_embedding_graph(self):
        ''' calculate attribute-value pair embedding '''
        self.attr_embeddings = self._generate_init_attribute_embeddings()
        self._attr_graph_convolution()
        concate_embeddings = tf.nn.embedding_lookup(self.attr_embeddings, self.value_attr_concate)
        concate_embeddings = tf.concat((concate_embeddings, self.no_attr), axis=0)
        concate_embeddings = tf.concat((self.init_value_embeddings, concate_embeddings), axis=1)
        self.value_embeddings = tf.matmul(concate_embeddings, self.reduce_dim_matrix)
        self.value_embeddings = tf.nn.l2_normalize(self.value_embeddings, axis=1)
        self.ent_embeddings = tf.nn.embedding_lookup(self.value_embeddings, self.ent_init_list)

    def _generate_mapping_loss(self, pos_embeds1, pos_embeds2, neg_embeds1, neg_embeds2):
        pos_embeds1 = tf.nn.l2_normalize(pos_embeds1, axis=1)
        pos_embeds2 = tf.nn.l2_normalize(pos_embeds2, axis=1)
        pos_distance = tf.norm(tf.subtract(pos_embeds1, pos_embeds2), axis=-1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_distance - tf.constant(self.params.pos_align_margin, dtype=tf.float32)))

        neg_embeds1 = tf.nn.l2_normalize(neg_embeds1, axis=1)
        neg_embeds2 = tf.nn.l2_normalize(neg_embeds2, axis=1)
        neg_distance = tf.norm(tf.subtract(neg_embeds1, neg_embeds2), axis=-1)
        neg_distance = tf.clip_by_value(neg_distance, 1e-10, 10)

        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.params.neg_align_margin, dtype=tf.float32) - neg_distance))
        return pos_loss + neg_loss

    def _generate_ce_mapping_graph(self):
        self.input_entities = tf.placeholder(tf.int32, shape=[None])
        self.label_entities = tf.placeholder(tf.float32, shape=[None, None])
        if self.params.model != 'RGAT':
            self._generate_weight_ent_embeddings()  # cancel this if discard aie
        self.ent_embeddings = tf.concat((self.ent_embeddings, self.ent_padding), axis=0)
        if self.rel_mapping_flag:
            self.rel_embeddings = tf.matmul(self.rel_embeddings, self.temp_rel_map)
            self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        self._rgat_graph_convolution()
        embed_list = list()
        self.mapping_loss = 0.0
        for output_embeds in self.output:
            output_embeds = tf.nn.l2_normalize(output_embeds, axis=1)
            embed_list.append(output_embeds)
            input_embeds = tf.nn.embedding_lookup(output_embeds, self.input_entities)
            self.mapping_loss += self._bce_loss(input_embeds, self.label_entities, output_embeds)

        ent_embeddings = tf.concat(embed_list, axis=1)
        ent_embeddings = tf.nn.l2_normalize(ent_embeddings, axis=1)
        input_embeds = tf.nn.embedding_lookup(ent_embeddings, self.input_entities)
        self.mapping_loss += self._bce_loss(input_embeds, self.label_entities, ent_embeddings)
        opt = tf.train.AdamOptimizer(self.params.learning_rate)
        self.mapping_optimizer = opt.minimize(self.mapping_loss)

    @staticmethod
    def _bce_loss(input_embeds, label, embeds):
        sim = tf.matmul(input_embeds, tf.transpose(embeds))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=sim, dim=-1)
        loss = tf.reduce_mean(loss)
        return loss

    def test(self):
        ti = time.time()
        self._rgat_graph_convolution(evaluation=True)
        embeds_list1, embeds_list2 = list(), list()
        for output_embeds in self.output:
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.test_entities1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.test_entities2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)

        test_embeds1 = tf.concat(embeds_list1, axis=1)
        test_embeds2 = tf.concat(embeds_list2, axis=1)
        test_embeds1 = tf.nn.l2_normalize(test_embeds1, axis=1)
        test_embeds2 = tf.nn.l2_normalize(test_embeds2, axis=1)
        test_embeds1 = test_embeds1.eval(session=self.session)
        test_embeds2 = test_embeds2.eval(session=self.session)
        alignment_rest, hits1, mr_12, mrr_12 = greedy_alignment(test_embeds1,
                                                                test_embeds2,
                                                                self.params.ent_top_k,
                                                                self.params.nums_threads,
                                                                'inner', False, 0, True)

        print("test totally costs {:.1f} s ".format(time.time() - ti))
        del test_embeds1, test_embeds2
        gc.collect()
        return hits1

    def valid(self):
        ti = time.time()
        embeds_list1, embeds_list2 = list(), list()
        self._rgat_graph_convolution(evaluation=True)

        for output_embeds in self.output:
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.valid_entities1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.valid_entities2 + self.test_entities2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        valid_embeds1 = tf.concat(embeds_list1, axis=1)
        valid_embeds2 = tf.concat(embeds_list2, axis=1)
        valid_embeds1 = tf.nn.l2_normalize(valid_embeds1, axis=1)
        valid_embeds2 = tf.nn.l2_normalize(valid_embeds2, axis=1)
        valid_embeds1 = valid_embeds1.eval(session=self.session)
        valid_embeds2 = valid_embeds2.eval(session=self.session)
        alignment_rest, hits1, mr_12, mrr_12 = greedy_alignment(valid_embeds1,
                                                                valid_embeds2,
                                                                self.params.ent_top_k,
                                                                self.params.nums_threads,
                                                                'inner', False, 0, True)

        print("validation costs {:.1f} s ".format(time.time() - ti))
        del valid_embeds1, valid_embeds2
        gc.collect()
        return hits1

    def save(self):
        self._rgat_graph_convolution(evaluation=True)
        output = list()
        for output_embeds in self.output:
            embeds = output_embeds
            embeds = tf.nn.l2_normalize(embeds, axis=1)
            output.append(embeds)
        embeds = tf.concat(output, axis=1)
        embeds = tf.nn.l2_normalize(embeds, axis=1)
        first_embeds = output[0]
        first_embeds = tf.nn.l2_normalize(first_embeds, axis=1)
        first_embeds = first_embeds.eval(session=self.session)
        embeds = embeds.eval(session=self.session)
        dataset_name = self.params.input.split("/")[-2]
        name = "../"+dataset_name+"_"+str(self.layer_num)+".npy"

        np.save(name, embeds)

    def eval_kb1_input_embed(self):
        ent_embeddings = self.output[-1]
        embeds = tf.nn.embedding_lookup(ent_embeddings, self.ents1)
        return embeds.eval(session=self.session)

    def eval_kb2_input_embed(self):
        ent_embeddings = self.output[-1]
        embeds = tf.nn.embedding_lookup(ent_embeddings, self.ents2)
        return embeds.eval(session=self.session)

    def eval_attribute_embed(self):
        ent_embeddings = self.ent_embeddings
        ent_embeddings = tf.nn.l2_normalize(ent_embeddings, axis=1)
        return ent_embeddings.eval(session=self.session)

    def eval_relation_embed(self):
        ent_embeddings = self.output[-1]
        ent_embeddings = tf.nn.l2_normalize(ent_embeddings, axis=1)
        return ent_embeddings.eval(session=self.session)
