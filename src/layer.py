import tensorflow as tf
import numpy as np


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
                                              dtype=tf.float32,
                                              trainable=True)
            if bias:
                self.bias_vec = tf.get_variable("gcn_bias" + str(layer_id),
                                                shape=[1, output_dim],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
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


class RGATLayer:
    def __init__(self, ent_in_dim, rel_in_dim, ent_out_dim, rel_out_dim, ent_rel_adj, ent_adj,
                 drop_rate, layer_id, ent_num, rel_len, act=None):
        self.ent_rel_adj = ent_rel_adj
        self.ent_adj = ent_adj
        self.drop_rate = drop_rate
        self.ent_num = ent_num
        self.rel_len = rel_len
        self.temp_count = 0
        self.smooth_k = np.sqrt(ent_in_dim)
        self.ent_embeddings = None
        self.rel_embeddings = None
        self.ent_dim = ent_in_dim
        self.rel_dim = rel_in_dim
        self.EPS = tf.constant(1e-5, dtype=tf.float32)
        self.zero = tf.constant(-1e15, dtype=tf.float32)
        self.activation = tf.tanh
        self.act = act
        self.layer_id = layer_id
        with tf.variable_scope("rgat_weight" + str(layer_id)):
            self.ent_weight = tf.get_variable("ent_weight" + str(layer_id),
                                              shape=[ent_in_dim, ent_out_dim],
                                              initializer=tf.glorot_uniform_initializer(),
                                              dtype=tf.float32,
                                              trainable=True)
            self.ent_rel_weight = tf.get_variable("ent_rel_weight" + str(layer_id),
                                                  shape=[ent_in_dim + 30, ent_out_dim],
                                                  initializer=tf.glorot_uniform_initializer(),
                                                  dtype=tf.float32,
                                                  trainable=True)
            self.rel_weight = tf.get_variable("rel_weight" + str(layer_id),
                                              shape=[rel_out_dim, 30],
                                              initializer=tf.glorot_uniform_initializer(),
                                              dtype=tf.float32,
                                              trainable=True)
            self.weight = tf.get_variable("weight" + str(layer_id),
                                          shape=[ent_out_dim, ent_out_dim],
                                          initializer=tf.glorot_uniform_initializer(),
                                          dtype=tf.float32,
                                          trainable=True)
            self.rel_mapping = tf.get_variable("rel_mapping" + str(layer_id),
                                               shape=[rel_in_dim, rel_out_dim],
                                               initializer=tf.glorot_uniform_initializer(),
                                               dtype=tf.float32,
                                               trainable=True)

    def call(self, ent_inputs, rel_inputs):
        self.temp_count = 0
        ent_embeddings = ent_inputs
        rel_embeddings = rel_inputs
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        padding_rel = tf.nn.embedding_lookup(self.rel_embeddings, self.ent_rel_adj)
        raw_padding_rel = padding_rel
        padding_rel = tf.transpose(padding_rel, perm=[0, 2, 1])
        ent_embeddings = tf.expand_dims(self.ent_embeddings, axis=1)
        coef_table = tf.matmul(ent_embeddings, padding_rel)
        coef_table = tf.where(tf.abs(coef_table) < self.EPS, self.zero * tf.ones_like(coef_table), coef_table)
        coef_table = coef_table / self.smooth_k
        coef_table = tf.nn.softmax(coef_table, axis=-1)
        coef_table = tf.transpose(coef_table, perm=[0, 2, 1])

        neighbor_ent = tf.nn.embedding_lookup(self.ent_embeddings, self.ent_adj)
        neighbor_ent = coef_table * neighbor_ent
        neighbor_ent = tf.reduce_sum(neighbor_ent, axis=1)
        gate = tf.matmul(self.ent_embeddings, self.weight)
        gate = tf.keras.activations.tanh(gate)
        self.ent_embeddings = tf.add(tf.multiply(neighbor_ent, 1 - gate), tf.multiply(self.ent_embeddings, gate))
        # ******************************************************
        rel_combine = coef_table * raw_padding_rel
        rel_combine = tf.reduce_sum(rel_combine, axis=1)
        rel_combine = tf.matmul(rel_combine, self.rel_weight)
        self.ent_embeddings = tf.concat((self.ent_embeddings, rel_combine), axis=1)
        if self.act is not None:
            return self.activation(self.ent_embeddings), self.activation(self.rel_embeddings)
        return self.ent_embeddings, self.rel_embeddings

    def _init_con(self, i, tensor_array):
        return i < self.ent_num

    def _init_body(self, i, tensor_array):
        query_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, i)
        query_embeddings = tf.expand_dims(query_embeddings, 0)
        key_table = self.ent_rel_adj[self.temp_count]
        key_embeddings = tf.nn.embedding_lookup(self.rel_embeddings, key_table)
        value_table = self.ent_adj[self.temp_count]
        value_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, value_table)
        self.temp_count += 1

        exp_table = tf.matmul(query_embeddings, tf.transpose(key_embeddings)) / self.smooth_k
        exp_table = tf.nn.softmax(exp_table, axis=-1)
        output_embeddings = tf.transpose(exp_table) * value_embeddings
        output_embeddings = tf.reduce_sum(output_embeddings, axis=0)
        tensor_array = tensor_array.write(i, output_embeddings)
        return i + 1, tensor_array
