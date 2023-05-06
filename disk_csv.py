import sys
from pathlib import Path

CURRENT_DIR = Path('.')
UTILS_DIR = CURRENT_DIR / '../'
sys.path.append(UTILS_DIR.absolute().as_posix())
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def batched_gather1(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseduocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
      tensor: Tensor of arbitrary size.
      indices: Vector of indices.
    Returns:
      output: A tensor of gathered values.
    """
    shape = (tensor.get_shape().as_list())
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output 

class DISK_CSV(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [config["batch_size"], self.max_len])
        self.label = tf.placeholder(tf.float32, [config["batch_size"], self.n_class], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)
    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        self.W_concept = tf.Variable(tf.random_uniform([self.n_class, self.embedding_size], -1.0, 1.0), name="W_concept",trainable=True)

        self.batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x) 

        
        # AVERAGE EMBEDDING
        self.em = tf.nn.tanh(tf.reduce_mean(self.batch_embedded,1, name="emb_sent"))
        
        
        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=(self.batch_embedded), dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(tf.tanh(fw_outputs + bw_outputs  )  , [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        self.r = tf.matmul(tf.transpose(self.H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        self.tr = tf.transpose(self.H, [0, 2, 1])
        r = tf.squeeze(self.r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        self.h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        


        y_hat = tf.nn.xw_plus_b(self.h_drop, FC_W, FC_b)
        self.probs = tf.nn.softmax(y_hat)
        
        
        
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label)
        self.loss = tf.reduce_mean(self.loss )
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
        correct_predictions = tf.equal(self.prediction, tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        # REWARD LOSS 
        self.reward_class = tf.cast((batched_gather1(self.label,tf.cast(self.prediction, tf.int32))), tf.float32)*0.9
        # PREDICT CONCEPT VECTOR
        self.concept_vect = tf.nn.tanh(tf.nn.embedding_lookup(self.W_concept,self.prediction ))
	self.reward_class = tf.cast((batched_gather1(self.label,tf.cast(self.prediction, tf.int32))), tf.float32)*0.9
	self.latent_loss = tf.reduce_mean(tf.losses.cosine_distance(tf.nn.l2_normalize(self.concept_vect, 1), tf.nn.l2_normalize(self.h_drop, 1), 1)*tf.expand_dims(self.reward_class,1))
        
        
        
        x_norm = tf.nn.l2_normalize((self.em), 1)
        y_norm = tf.nn.l2_normalize(self.concept_vect, 1)
        cos = 1-tf.reduce_sum(x_norm * y_norm, axis=1)
        self.emb_loss = tf.reduce_mean(((cos))*tf.expand_dims(self.reward_class,1))
 
        

        q = tf.nn.tanh(self.W_concept)
        # PAIRWISE DISTANCE SIMILAIRTY/DISTANCE
        self.pair=-tf.reduce_sum(tf.reduce_sum((tf.expand_dims(q, 1)-tf.expand_dims(q, 0))**2,2))
        

        # optimization
        # loss_to_minimize = self.loss
	#Sometimes you might need to mutliply by a large number depending on the problem here i'm providing an example
        loss_to_minimize = self.loss + 285*self.emb_loss + 12*self.pair  +  0.4*self.latent_loss
        self.target_loss = loss_to_minimize


        
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")




vocab_size = 50002
maxlen = 32
n_classes = 15
config = {
    "max_len": 50,
    "hidden_size": 128,
    "vocab_size": 1995,
    "embedding_size": 128,
    "n_class": 2,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "train_epoch": 20
}
classifier = DISK_CSV(config)
classifier.build_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
BATCH_SIZE = 512




total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)
