import gc
import time
import tensorflow as tf
from util import embed_init
from test_funcs import greedy_alignment

g = 1024 * 1024


class GCNLayer:
    def __init__(self,
                 adj,
                 input_dim,
                 output_dim,
                 layer_id,
                 bias=True,
                 act=None):
        self.bias = bias
        self.act = act
        self.adj = adj
        with tf.variable_scope("gcn_layer_" + str(layer_id)):
            self.weight_mat = tf.get_variable("gcn_weights" + str(layer_id),
                                              shape=[input_dim, output_dim],
                                              initializer=tf.glorot_uniform_initializer(),
                                              dtype=tf.float64,
                                              trainable=True)
            if bias:
                self.bias_vec = tf.get_variable("gcn_bias" + str(layer_id),
                                                shape=[1, output_dim],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float64,
                                                trainable=True)

    def call(self, inputs, drop_rate=0.0):
        input_tensor = inputs
        if drop_rate > 0.0:
            input_tensor = tf.nn.dropout(input_tensor, rate=drop_rate) * (1 - drop_rate)  # not scaled up
        output = tf.matmul(input_tensor, self.weight_mat)
        output = tf.sparse.sparse_dense_matmul(self.adj, output)
        if self.bias:
            bias_vec = self.bias_vec
            output = tf.add(output, bias_vec)
        if self.act is not None:
            output = self.act(output)
        return output


class GCN:
    def __init__(self, kgs, adj, attr_adj, params, value_embedding, ent_embedding, attribute_embedding):
        self.ent_num = kgs.entities_num
        self.value_num = kgs.values_num
        self.attr_num = kgs.attributes_num

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
        self.layer_num = params.layer_num
        self.adj_mat = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        if attr_adj is not None:
            self.attr_adj = tf.SparseTensor(indices=attr_adj[0], values=attr_adj[1], dense_shape=attr_adj[2])
        self.activation = tf.nn.tanh
        self.layers = list()
        self.output = list()

        self.dim = params.dim
        self.value_embeddings = value_embedding
        self.ent_embeddings = ent_embedding
        self.temp_ent_embeddings = ent_embedding
        self.temp_attribute_embeddings = attribute_embedding

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.compat.v1.Session(config=config)

        self.lr = self.params.learning_rate
        self._generate_variables()
        self._generate_mapping_graph()

        tf.global_variables_initializer().run(session=self.session)

    def _generate_variables(self):

        with tf.variable_scope("ent_embeddings"):
            self.ent_embeddings = embed_init(self.ent_num, self.params.dim, "init_ent_embedding",
                                             method='glorot_uniform_initializer')
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        with tf.variable_scope("mapping_embeddings"):
            self.mapping_matrix = tf.get_variable('mapping_matrix',
                                                  dtype=tf.float64,
                                                  shape=[self.params.dim, self.params.dim],
                                                  initializer=tf.initializers.orthogonal(dtype=tf.float64))

    def _graph_convolution(self):
        self.output = list()  # reset
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        output_embeddings = self.ent_embeddings
        self.output.append(output_embeddings)
        for i in range(self.params.layer_num):
            activation = self.activation
            if i == self.layer_num - 1:
                activation = None
            gcn_layer = GCNLayer(self.adj_mat, self.params.dim, self.params.dim, i, act=activation)
            self.layers.append(gcn_layer)
            output_embeddings = gcn_layer.call(output_embeddings, drop_rate=self.params.drop_rate)
            output_embeddings = tf.nn.l2_normalize(output_embeddings, axis=1)
            self.output.append(output_embeddings)

    def _generate_mapping_graph(self):
        self.pos_entities1 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.pos_entities2 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_entities1 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_entities2 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self._graph_convolution()
        ent_embeddings = self.output[-1]
        pos_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.pos_entities1)
        pos_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.pos_entities2)
        neg_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.neg_entities1)
        neg_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.neg_entities2)

        self.mapping_loss = self._generate_mapping_loss(pos_embeds1, pos_embeds2, neg_embeds1, neg_embeds2)
        opt = tf.train.AdamOptimizer(self.params.learning_rate)
        self.mapping_optimizer = opt.minimize(self.mapping_loss)

    def _generate_mapping_loss(self, pos_embeds1, pos_embeds2, neg_embeds1, neg_embeds2):
        pos_embeds2 = tf.nn.l2_normalize(pos_embeds2, axis=1)

        pos_embeds1 = tf.matmul(pos_embeds1, self.mapping_matrix)
        pos_embeds1 = tf.nn.l2_normalize(pos_embeds1, axis=1)

        pos_distance = tf.norm(tf.subtract(pos_embeds1, pos_embeds2), axis=-1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_distance - tf.constant(self.params.pos_align_margin, dtype=tf.float64)))

        neg_embeds1 = tf.nn.l2_normalize(neg_embeds1, axis=1)
        neg_embeds2 = tf.nn.l2_normalize(neg_embeds2, axis=1)

        neg_embeds1 = tf.matmul(neg_embeds1, self.mapping_matrix)
        neg_embeds1 = tf.nn.l2_normalize(neg_embeds1, axis=1)

        neg_distance = tf.norm(tf.subtract(neg_embeds1, neg_embeds2), axis=-1)

        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.params.neg_align_margin, dtype=tf.float64) - neg_distance))
        return pos_loss + neg_loss

    def test(self):
        ti = time.time()
        ent_embeddings = self.output[-1]
        test_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.test_entities1)
        test_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.test_entities2)

        test_embeds2 = tf.nn.l2_normalize(test_embeds2, axis=1)

        test_embeds1 = tf.matmul(test_embeds1, self.mapping_matrix)
        test_embeds1 = tf.nn.l2_normalize(test_embeds1, axis=1)

        test_embeds1 = test_embeds1.eval(session=self.session)
        test_embeds2 = test_embeds2.eval(session=self.session)
        alignment_rest, hits1, mr_12, mrr_12 = greedy_alignment(test_embeds1,
                                                                test_embeds2,
                                                                self.params.ent_top_k,
                                                                self.params.nums_threads,
                                                                'inner', False, 0, False)
        print('[mr, mrr]', mr_12, mrr_12)
        print("test totally costs {:.3f} s ".format(time.time() - ti))
        del test_embeds1, test_embeds2
        gc.collect()
        return hits1

    def valid(self):
        ti = time.time()
        ent_embeddings = self.output[-1]
        valid_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.valid_entities1)
        valid_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.valid_entities2 + self.test_entities2)

        valid_embeds2 = tf.nn.l2_normalize(valid_embeds2, axis=1)

        valid_embeds1 = tf.matmul(valid_embeds1, self.mapping_matrix)
        valid_embeds1 = tf.nn.l2_normalize(valid_embeds1, axis=1)

        valid_embeds1 = valid_embeds1.eval(session=self.session)
        valid_embeds2 = valid_embeds2.eval(session=self.session)
        alignment_rest, hits1, mr_12, mrr_12 = greedy_alignment(valid_embeds1,
                                                                valid_embeds2,
                                                                self.params.ent_top_k,
                                                                self.params.nums_threads,
                                                                'inner', False, 0, False)

        print('[mr, mrr]', mr_12, mrr_12)

        print("test totally costs {:.3f} s ".format(time.time() - ti))
        del valid_embeds1, valid_embeds2
        gc.collect()
        return hits1

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
