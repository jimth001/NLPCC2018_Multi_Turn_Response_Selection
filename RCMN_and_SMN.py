import tensorflow as tf
import pickle
import utils
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import rnn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import Evaluate
import time
from datetime import timedelta
import Layers
import math

# In this project,the path of a file include the name of this file,but dir not
#data_path='./data'

embedding_file = r"./data/embedding.pkl"
evaluate_file = r"./data/Evaluate.pkl"
response_file = r"./data/responses.pkl"
history_file = r"./data/utterances.pkl"


class SCN():
    def __init__(self, embedding):
        self.max_num_utterance = 10
        self.negative_samples = 2
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.rnn_units = 200
        self.total_words = embedding.shape[0]  # 136365
        self.batch_size = 40
        self.print_batch = 6000
        self.embedding=embedding

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def LoadModel(self, model_path):
        # init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        # with tf.Session() as sess:
        # sess.run(init)
        # saver.restore(sess,"neg5model\\model.5")
        return sess
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        # with tf.Session() as sess:
        #     # Restore variables from disk.
        #     saver.restore(sess, "/model/model.5")
        #     print("Model restored.")

    def build_model(self):
        # define placeholder:
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        # embedding layer:
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                      word_embedding_size), dtype=tf.float32,
                                          trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        # sentence_GRU = rnn.GRUCell(self.rnn_units,kernel_initializer=tf.orthogonal_initializer())
        sentence_GRU = rnn.GRUCell(self.rnn_units)
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units),
                                   initializer=xavier_initializer(), dtype=tf.float32)
        # final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        final_GRU = rnn.GRUCell(self.rnn_units)
        reuse = None
        #
        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings,
                                                       sequence_length=self.response_len, dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])  # 转置，1和2对换
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])  # 转置，1和2对换
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings,
                                                            sequence_length=utterance_len, dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse,
                                              name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        '''Time_major决定了inputs Tensor前两个dim表示的含义
        time_major = False时[batch_size, sequence_length, embedding_size]
        time_major = True时[sequence_length, batch_size, embedding_size]'''
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'),
                                           dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.class_label_pred=tf.argmax(self.y_pred, 1)# 预测类别
        self.total_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)

    def build_new_model(self):
        # define placeholder:
        with tf.variable_scope('placeholders') as scope:
            self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
            self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
            self.y_true = tf.placeholder(tf.int32, shape=(None,))
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.response_len = tf.placeholder(tf.int32, shape=(None,))
            self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        with tf.variable_scope('embedding') as scope:
            # embedding layer:
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                            word_embedding_size), dtype=tf.float32,trainable=False)
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
            all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
            all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
            response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        with tf.variable_scope('rnn_representation') as scope:
            sentence_GRU = rnn.GRUCell(self.rnn_units,kernel_initializer=tf.orthogonal_initializer())
            A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units,2, self.rnn_units),
                                   initializer=xavier_initializer(), dtype=tf.float32)
            matching_vectors = []
            response_GRU_embeddings, response_state = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings,
                                                       sequence_length=self.response_len, dtype=tf.float32)
            matching_vectors.append(response_state)
            for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
                utterance_GRU_embeddings, last_state = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings,
                                                            sequence_length=utterance_len, dtype=tf.float32)
                matching_vectors.append(last_state)
            matching_vectors=tf.stack(matching_vectors, axis=0, name='matching_stack')#batchsize*rnnunit
        with tf.variable_scope('matching_image_cnn'):
            matching_vectors=tf.transpose(matching_vectors,perm=[1,0,2])
            tmp=tf.tensordot(matching_vectors,A_matrix,axes=[[2],[0]])
            mv_t=tf.transpose(matching_vectors,perm=[0,2,1])
            mv_t=tf.stack([mv_t]*2,axis=1)
            matching_image=tf.matmul(tf.transpose(tmp,perm=[0,2,1,3]),mv_t)
            matching_image=tf.transpose(matching_image,perm=[0,2,3,1])
            #conv_layer = tf.layers.conv2d(matching_image, filters=32, kernel_size=(3, 3), padding='VALID',
            conv_layer = tf.layers.conv2d(matching_image, filters=8, kernel_size=(3, 3), padding='VALID',
                                    kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                    activation=tf.nn.relu, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                    padding='VALID', name='max_pooling')  # TODO: check other params
            final_matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.tanh,name='matching_v')  # TODO: check wthether this is correct
        with tf.variable_scope('output'):
            '''Time_major决定了inputs Tensor前两个dim表示的含义
            time_major = False时[batch_size, sequence_length, embedding_size]
            time_major = True时[sequence_length, batch_size, embedding_size]'''
            logits = tf.layers.dense(final_matching_vector, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='final_v')
            self.y_pred = tf.nn.softmax(logits)
            self.class_label_pred = tf.argmax(self.y_pred, 1)  # 预测类别
        with tf.variable_scope('optimize'):
            self.total_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
            tf.summary.scalar('loss', self.total_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(self.total_loss)

    def copy_list(self,list):
        new_list=[]
        for l in list:
            if type(l)==type([0]) or type(l)==np.array([0]):
                new_list.append(self.copy_list(l))
            else:
                new_list.append(l)
        return new_list

    def predict(self,model_path,history,response):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 只分配40%的显存
        first = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            all_candidate_scores = []
            all_pred_labels = []
            low = 0
            batch_size_for_val = 3000
            while True:
                batch_history = self.copy_list(history[low:low + batch_size_for_val])
                batch_history, batch_history_len = utils.multi_sequences_padding(batch_history, self.max_sentence_len)
                batch_history, batch_history_len = np.array(batch_history), np.array(batch_history_len)
                batch_response = self.copy_list(response[low:low + batch_size_for_val])
                batch_response_len = np.array(utils.get_sequences_length(batch_response, maxlen=self.max_sentence_len))
                batch_response = np.array(pad_sequences(batch_response, padding='post', maxlen=self.max_sentence_len))
                feed_dict = {self.utterance_ph: batch_history,
                             self.all_utterance_len_ph: batch_history_len,
                             self.response_ph: batch_response,
                             self.response_len: batch_response_len,
                             }
                candidate_scores, pred_labels = sess.run([self.y_pred, self.class_label_pred], feed_dict=feed_dict)
                if first:
                    print(pred_labels)
                    first = False
                all_candidate_scores.append(candidate_scores[:, 1])
                all_pred_labels.append(pred_labels)
                low = low + batch_size_for_val
                if low >= len(response):
                    break
            all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
            all_pred_labels = np.concatenate(all_pred_labels, axis=0)
        return all_candidate_scores,all_pred_labels

    def Evaluate(self, test_path,model_path):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 只分配40%的显存
        with open(test_path, 'rb') as f:
            val_history, val_response, val_labels = pickle.load(f)
        first = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            all_candidate_scores = []
            all_pred_labels=[]
            low = 0
            batch_size_for_val = 2000
            while True:
                batch_history=self.copy_list(val_history[low:low + batch_size_for_val])
                batch_history,batch_history_len=utils.multi_sequences_padding(batch_history, self.max_sentence_len)
                batch_history, batch_history_len=np.array(batch_history),np.array(batch_history_len)
                batch_response=self.copy_list(val_response[low:low + batch_size_for_val])
                batch_response_len=np.array(utils.get_sequences_length(batch_response, maxlen=self.max_sentence_len))
                batch_response=np.array(pad_sequences(batch_response, padding='post', maxlen=self.max_sentence_len))
                feed_dict = {self.utterance_ph: batch_history,
                            self.all_utterance_len_ph: batch_history_len,
                            self.response_ph: batch_response,
                            self.response_len: batch_response_len,
                            }
                candidate_scores,pred_labels = sess.run([self.y_pred,self.class_label_pred], feed_dict=feed_dict)
                if first:
                    print(pred_labels)
                    first=False
                all_candidate_scores.append(candidate_scores[:, 1])
                all_pred_labels.append(pred_labels)
                low = low + batch_size_for_val
                if low >= len(val_labels):
                    break
            all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
            all_pred_labels=np.concatenate(all_pred_labels,axis=0)
        return Evaluate.precision_of_classification(all_pred_labels,val_labels),Evaluate.mrr_and_rnk(all_candidate_scores,val_labels,response_num_per_query=11)

    def evaluate_val_for_train(self, sess, data):
        val_history, val_response, val_labels = data
        all_candidate_scores = []
        low = 0
        batch_size_for_val=4000
        while True:
            batch_history = self.copy_list(val_history[low:low + batch_size_for_val])
            batch_history, batch_history_len = utils.multi_sequences_padding(batch_history, self.max_sentence_len)
            batch_history, batch_history_len = np.array(batch_history), np.array(batch_history_len)
            batch_response = self.copy_list(val_response[low:low + batch_size_for_val])
            batch_response_len = np.array(utils.get_sequences_length(batch_response, maxlen=self.max_sentence_len))
            batch_response = np.array(pad_sequences(batch_response, padding='post', maxlen=self.max_sentence_len))
            feed_dict = {self.utterance_ph: batch_history,
                         self.all_utterance_len_ph: batch_history_len,
                         self.response_ph: batch_response,
                         self.response_len: batch_response_len,
                         self.y_true: np.concatenate([val_labels[low:low + batch_size_for_val]], axis=0),
                         }
            candidate_scores,loss = sess.run([self.y_pred,self.total_loss], feed_dict=feed_dict)
            all_candidate_scores.append(candidate_scores[:, 1])
            low = low + batch_size_for_val
            if low >= len(val_labels):
                break
        all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
        return Evaluate.precision_of_matching_1(all_candidate_scores, val_labels,response_num_per_query=11),loss

    def train_model_with_random_sample(self, continue_train=False, previous_model_path="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 只分配40%的显存
        # prepare data for val:
        with open(evaluate_file, 'rb') as f:
            val_history, val_response, val_labels = pickle.load(f)
        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter('output2', sess.graph)
            # prepare data for train:
            with open(response_file, 'rb') as f:
                actions = pickle.load(f)  # action is a list of response-candidates
            with open(history_file, 'rb') as f:
                # history is a 3d-list.1d:samples;2d:one utterance if a sample;3d:one word of a utterance
                # true_utt is a 2d-list.1d:sample;2d the true response of the sample
                history, true_utt = pickle.load(f)
            actions=self.copy_list(actions)
            actions_len = np.array(utils.get_sequences_length(actions, maxlen=self.max_sentence_len))
            actions = np.array(pad_sequences(actions, padding='post', maxlen=self.max_sentence_len))
            history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
            true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
            true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
            history, history_len = np.array(history), np.array(history_len)
            if continue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
            else:
                saver.restore(sess, previous_model_path)
            low = 0
            epoch = 1
            start_time = time.time()
            sess.graph.finalize()
            best_score=100
            while epoch < 10:
                # low means the start location of the array of data should be feed in next
                # n_samples means how many group-samples will be feed in next time
                # one group-samples means one context and its true response and some neg responses
                n_sample = min(low + self.batch_size, history.shape[0]) - low
                # negative_samples means the num of neg for one context
                # negative_indices is a 2d-list(negative_samples*n_sample)
                negative_indices = [np.random.randint(0, actions.shape[0], n_sample) for _ in
                                    range(self.negative_samples)]  #
                # negs's shape is negative_samples*n_sample*sentence_max_len
                negs = [actions[negative_indices[i], :] for i in range(self.negative_samples)]
                negs_len = [actions_len[negative_indices[i]] for i in range(self.negative_samples)]
                feed_dict = {
                    self.utterance_ph: np.concatenate([history[low:low + n_sample]] * (self.negative_samples + 1),
                                                      axis=0),
                    self.all_utterance_len_ph: np.concatenate(
                        [history_len[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                    self.response_ph: np.concatenate([true_utt[low:low + n_sample]] + negs, axis=0),
                    self.response_len: np.concatenate([true_utt_len[low:low + n_sample]] + negs_len, axis=0),
                    self.y_true: np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * self.negative_samples,
                                                axis=0)
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % (self.batch_size * self.print_batch) == 0:
                    time_dif = self.__get_time_dif(start_time)
                    r10_1,loss=self.evaluate_val_for_train(sess, [val_history, val_response, val_labels])
                    if best_score>loss:
                        best_score=loss
                        saver.save(sess, "model/model_best.{0}".format(low))
                    print("train loss:", sess.run(self.total_loss, feed_dict=feed_dict), "; val evaluation:",r10_1,
                           "time:", time_dif)
                    print('loss',loss)
                if low >= history.shape[0]:  # 即low>=total conversations number
                    low = 0
                    saver.save(sess, "model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i}'.format(i=epoch), 'ended')
                    epoch += 1

    def train_model_with_fixed_data(self, file_src_dict, response_num=3, continue_train=False,
                                    previous_model_path="model"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1.0  # 只分配40%的显存
        # prepare data for val:
        with open(evaluate_file, 'rb') as f:
            val_history, val_response, val_labels = pickle.load(f)
        val_history, val_history_len = utils.multi_sequences_padding(val_history, self.max_sentence_len)
        val_history, val_history_len = np.array(val_history), np.array(val_history_len)
        val_response_len = np.array(utils.get_sequences_length(val_response, maxlen=self.max_sentence_len))
        val_response = np.array(pad_sequences(val_response, padding='post', maxlen=self.max_sentence_len))
        val_data = [val_history, val_history_len, val_response, val_response_len, val_labels]
        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter('output2', sess.graph)
            # prepare data for train:
            with open(file_src_dict['train_file'], 'rb') as f:
                history, responses, labels = pickle.load(f)
            history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
            responses_len = np.array(utils.get_sequences_length(responses, maxlen=self.max_sentence_len))
            responses = np.array(pad_sequences(responses, padding='post', maxlen=self.max_sentence_len))
            history, history_len = np.array(history), np.array(history_len)
            if continue_train is False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
            else:
                saver.restore(sess, previous_model_path)
            low = 0
            epoch = 1
            start_time = time.time()
            sess.graph.finalize()
            best_score=100
            while epoch < 10:
                # low means the start location of the array of data should be feed in next
                # n_samples means how many group-samples will be feed in next time
                # one group-samples means one context and its true response and some neg responses
                n_sample = min(low + self.batch_size * response_num, history.shape[0]) - low
                feed_dict = {
                    self.utterance_ph: np.array(history[low:low + n_sample]),
                    self.all_utterance_len_ph: np.array(history_len[low:low + n_sample]),
                    self.response_ph: np.array(responses[low:low + n_sample]),
                    self.response_len: np.array(responses_len[low:low + n_sample]),
                    self.y_true: np.array(labels[low:low + n_sample])
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % (self.batch_size * self.print_batch) == 0:
                    time_dif = self.__get_time_dif(start_time)
                    r10_1,loss=self.evaluate_val_for_train(sess, val_data)
                    if best_score>loss:
                        best_score=loss
                        saver.save(sess, "model/model_best.{0}".format(low))
                    print("train loss:", sess.run(self.total_loss, feed_dict=feed_dict), "; val evaluation:",r10_1
                          ,loss, "time:", time_dif)
                if low >= history.shape[0]:  # 即low>=total conversations number
                    low = 0
                    saver.save(sess, "model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i}'.format(i=epoch), 'ended')
                    epoch += 1


def train_onehotkey():
    print('start')
    file_src_dict = {'embedding_file': './data/embedding.pkl', 'train_file': './data/train.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f, encoding="bytes")
    scn = SCN(embedding=embeddings)
    print('build graph')
    scn.build_model()
    #scn.build_new_model()
    #scn.build_ifan_model()
    #scn.train_model_with_fixed_data(file_src_dict=file_src_dict)
    print('start train')
    scn.train_model_with_random_sample()

def test_onehotkey():
    file_src_dict = {'embedding_file': './data/embedding.pkl', 'train_file': './data/train.pkl','test_file':'./data/Evaluate.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f, encoding="bytes")
    scn = SCN(embedding=embeddings)
    scn.build_model()
    class_report,precision_of_matching=scn.Evaluate(file_src_dict['test_file'],'./model/model_best.1200000')
    print(class_report)
    print(precision_of_matching)


if __name__ == "__main__":
    train_onehotkey()
    print('all work has finish')
