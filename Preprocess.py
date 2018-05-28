import random
from Tokenizer import Tokenizer
import numpy as np
import pickle
import word2vec


class Conversation:
    def __init__(self, context, true_response, false_response):
        # context,true_response,false_response should be list
        self.context = context  # list of str
        self.true_response = true_response  # str
        self.false_response = false_response  # list of str

    def word_segmentation(self, tokenizer):  # word segmentation for conversation
        all_segmented_sen = []
        for i in range(0, len(self.context)):
            self.context[i] = tokenizer.parser(self.context[i])
            all_segmented_sen.append(self.context[i])
        for i in range(0, len(self.false_response)):
            self.false_response[i] = tokenizer.parser(self.false_response[i])
            all_segmented_sen.append(self.false_response[i])
        self.true_response = tokenizer.parser(self.true_response)
        all_segmented_sen.append(self.true_response)
        return all_segmented_sen

    def word2index_with_dict(self, word_dict, with_unk=True):
        # if with_unk is true,the unknown word is will be mapped into a int index with number len(word_dict)
        if with_unk:
            for i in range(0, len(self.context)):
                index = []
                for word in self.context[i].split(' '):
                    if word in word_dict:
                        index.append(word_dict[word])
                    else:
                        index.append(len(word_dict))
                self.context[i] = index
            for i in range(0, len(self.false_response)):
                index = []
                for word in self.false_response[i].split(' '):
                    if word in word_dict:
                        index.append(word_dict[word])
                    else:
                        index.append(len(word_dict))
                self.false_response[i] = index
            index = []
            for word in self.true_response.split(' '):
                if word in word_dict:
                    index.append(word_dict[word])
                else:
                    index.append(len(word_dict))
            self.true_response = index
        else:
            for i in range(0, len(self.context)):
                self.context[i] = [word_dict[word] for word in self.context[i].split(' ') if word in word_dict]
            for i in range(0, len(self.false_response)):
                self.false_response[i] = [word_dict[word] for word in self.false_response[i].split(' ') if
                                          word in word_dict]
            self.true_response = [word_dict[word] for word in self.true_response.split(' ') if word in word_dict]

    # def


def data_loader(data_path="./data/train.txt", stat=True):
    # 加载原始数据并输出统计信息
    file = open(data_path, encoding="utf-8", mode="r")
    conversations = []
    utterances = []
    total_utterances = 0
    utterances_len_dict = {}
    total_len = 0
    all_utterances = []
    for line in file:
        line = line.strip("\n")
        if len(line) == 0:  #
            if len(utterances) == 0:  # 遇到空行，当前utterance为空，说明上轮对话读完后还未读到新的，因此跳过
                continue
            else:  # 遇到空行，当前utterance不空，读完了一轮对话，
                context = utterances[0:len(utterances) - 1]
                true_response = utterances[len(utterances) - 1]  # str
                conversations.append(Conversation(context=context, true_response=true_response, false_response=[]))
                total_utterances += len(utterances)
                utterances = []
        else:
            all_utterances.append(line)
            utterances.append(line)
            length = len(line)
            total_len += length
            if length in utterances_len_dict:
                utterances_len_dict[length] += 1
            else:
                utterances_len_dict[length] = 1
    utterances_per_conversation = total_utterances / len(conversations)
    length_per_utterance = total_len / total_utterances
    print(len(conversations))
    print(utterances_per_conversation)
    print(length_per_utterance)
    print(utterances_len_dict)
    return conversations, all_utterances


def generate_neg_data_randomly(input_data, neg_number=1):  # 为原始数据生成负例
    conversations, all_utterance = input_data
    for con in conversations:
        false_response = []
        while len(false_response) < neg_number:
            tmp = all_utterance[random.randint(0, len(all_utterance) - 1)]
            if tmp != con.true_response:
                false_response.append(tmp)
        con.false_response = false_response


def geneate_neg_data_base_tfidf():
    pass


def partition_dataset(data, n_fold=4, with_test=False):  # base on Conversation
    # 根据x，y划分n-fold交叉验证的数据集。如果with_test=True,则会划分测试集
    # 若输入x有n个m维特征。则x.shape is [n,data_num,m]
    assert n_fold >= 3, "n_flod must be bigger than 3"
    val_len = int(len(data) / n_fold)
    indices = np.random.permutation(np.arange(len(data)))
    data_shuffle = [data[i] for i in indices]
    val = data_shuffle[0:val_len]
    if with_test:
        test = data_shuffle[val_len:2 * val_len]
        train = data_shuffle[2 * val_len:]
        return train, val, test
    else:
        val = data_shuffle[0:val_len]
        train = data_shuffle[val_len:]
        return train, val


def generate_raw_dataset():
    conversations, all_utterances = data_loader(data_path='./data/train.txt', stat=True)
    train, val, test = partition_dataset(conversations, n_fold=8, with_test=True)
    generate_neg_data_randomly([train, all_utterances], neg_number=2)
    generate_neg_data_randomly([val, all_utterances], neg_number=10)
    generate_neg_data_randomly([test, all_utterances], neg_number=10)  # todo 原来是train。重大错误。要重跑
    pickle.dump(train, open('./data/train.raw.pkl', 'wb+'), protocol=True)
    pickle.dump(val, open('./data/val.raw.pkl', 'wb+'), protocol=True)
    pickle.dump(test, open('./data/test.raw.pkl', 'wb+'), protocol=True)
    pickle.dump(all_utterances, open('./data/all_utt.raw.pkl', 'wb+'), protocol=True)


def seg_sen_for_word2vec():
    tokenizer = Tokenizer()
    # train
    train = pickle.load(open('./data/train.raw.pkl', 'rb'))
    with open('./data/corpus_for_word2vec.txt', 'w+', encoding='utf-8') as f:
        for con in train:
            all_seged_sens = con.word_segmentation(tokenizer)
            all_seged_sens = [line + '\n' for line in all_seged_sens]
            f.writelines(all_seged_sens)
    pickle.dump(train, open('./data/train.seg.pkl', 'wb+'), protocol=True)
    del train
    # val
    val = pickle.load(open('./data/val.raw.pkl', 'rb+'))
    for con in val:
        con.word_segmentation(tokenizer)
    pickle.dump(val, open('./data/val.seg.pkl', 'wb+'), protocol=True)
    del val
    # test
    test = pickle.load(open('./data/test.raw.pkl', 'rb'))
    for con in test:
        con.word_segmentation(tokenizer)
    pickle.dump(test, open('./data/test.seg.pkl', 'wb+'), protocol=True)


def pre_train_word_embedding():
    word2vec.word2vec('./data/corpus_for_word2vec.txt', './data/word_embedding.bin', size=200, window=8, sample='1e-5',
                      cbow=0, save_vocab='./data/worddict', min_count=6)


def load_word_embedding():
    # word_embedding:[clusters=None,vectors,vocab,vocab_hash]
    word_embedding = word2vec.load('./data/word_embedding.bin')
    return word_embedding


def transfer_words_to_index(word_dict, with_unk=True):
    train = pickle.load(open('./data/train.seg.pkl', 'rb'))
    for con in train:
        con.word2index_with_dict(word_dict, with_unk=with_unk)
    pickle.dump(train, open('./data/train.index.pkl', 'wb+'), protocol=True)
    del train
    val = pickle.load(open('./data/val.seg.pkl', 'rb'))
    for con in val:
        con.word2index_with_dict(word_dict, with_unk=with_unk)
    pickle.dump(val, open('./data/val.index.pkl', 'wb+'), protocol=True)
    del val
    test = pickle.load(open('./data/test.seg.pkl', 'rb'))
    for con in test:
        con.word2index_with_dict(word_dict, with_unk=with_unk)
    pickle.dump(test, open('./data/test.index.pkl', 'wb+'), protocol=True)


def transfer_data_format_for_scn(with_unk=True, random_sampling_during_train=True):
    word_embedding = load_word_embedding()
    if with_unk is True:
        word_embedding.vectors = np.insert(word_embedding.vectors, -1, [0.0] * word_embedding.vectors.shape[1])
    pickle.dump(word_embedding.vectors, open('./data/embedding.pkl', 'wb+'), protocol=True)
    train = pickle.load(open('./data/train.index.pkl', 'rb'))
    val = pickle.load(open('./data/val.index.pkl', 'rb'))
    test = pickle.load(open('./data/test.index.pkl', 'rb'))
    actions = []
    true_utt = []
    contexts = []
    if random_sampling_during_train is True:
        for con in train:
            actions += con.false_response
            true_utt.append(con.true_response)
            contexts.append(con.context)
        pickle.dump(actions, open('./data/responses.pkl', 'wb+'), protocol=True)
        pickle.dump([contexts, true_utt], open('./data/utterances.pkl', 'wb+'), protocol=True)
    else:
        contexts = []
        labels = []
        responses = []
        for con in train:
            contexts += [con.context] * (len(con.false_response) + 1)
            labels += ([1] + [0] * len(con.false_response))
            responses.append(con.true_response)
            responses += con.false_response
        pickle.dump([contexts, responses, labels], open('./data/train.pkl', 'wb+'), protocol=True)
    # for val and test:
    contexts = []
    labels = []
    responses = []
    for con in val:
        contexts += [con.context] * (len(con.false_response) + 1)
        labels += ([1] + [0] * len(con.false_response))
        responses.append(con.true_response)
        responses += con.false_response
    pickle.dump([contexts, responses, labels], open('./data/Evaluate.pkl', 'wb+'), protocol=True)
    contexts.clear()
    labels.clear()
    responses.clear()
    for con in test:
        contexts += [con.context] * (len(con.false_response) + 1)
        labels += ([1] + [0] * len(con.false_response))
        responses.append(con.true_response)
        responses += con.false_response
    pickle.dump([contexts, responses, labels], open('./data/Evaluate_test.pkl', 'wb+'), protocol=True)


def divide_evaluate_file(file_path, neg_num=10, part_num=5):
    # 这是专门为了在pc上能够测试写的。pc内存和显存都很小。
    data = pickle.load(open(file_path, 'rb'))
    data_num = len(data)
    assert data_num % (neg_num + 1) == 0, 'neg_num and data_num are not match'
    batch_num = int(data_num / part_num)
    p = 0
    for i in range(0, part_num - 1):
        new_data = data[p:p + batch_num]
        pickle.dump(new_data, open(file_path + str(i), 'wb+'), protocol=True)
        p += batch_num
    new_data = data[p:]
    pickle.dump(new_data, open(file_path + str(part_num - 1), 'wb+'), protocol=True)


def add_noise_to_testdataset(original_neg_num=10,original_pos_num=1):
    #todo 为测试集添加噪声。随机选择utterances中的一个句子作为负例，看看model表现是否大幅下降
    evaluate_test = pickle.load(open('./data/Evaluate_test.pkl', 'rb'))
    samples_per_query=original_neg_num+original_pos_num
    data_len=len(evaluate_test[2])
    query_num=data_len/samples_per_query
    assert data_len%samples_per_query==0,'data length wrong'
    new_utterances=[]
    new_response=[]
    new_label=[]
    p=0
    while p<data_len:
        utt=evaluate_test[0][p]
        index=random.randint(0,len(utt)-1)
        new_response.append(utt[index])
        new_utterances.append(utt)
        new_label.append(0)
        p+=samples_per_query
    p=samples_per_query
    for i in range(0,query_num):
        evaluate_test[0].insert(p,new_utterances[i])
        evaluate_test[1].insert(p,new_response[i])
        evaluate_test[2].insert(p,new_label[i])
        p+=(samples_per_query+1)
    pickle.dump(evaluate_test,open("./data/Evaluate_test_noise.pkl",'wb+'),protocol=True)
    #todo 加入和正确response的tfidf很像的数据作为负例（这个不好选，又要tfidf像又不能是utterances的合理回复）

if __name__ == "__main__":
    generate_raw_dataset()
    seg_sen_for_word2vec()
    word_embedding = load_word_embedding()
    transfer_words_to_index(word_embedding.vocab_hash, with_unk=False)
    transfer_data_format_for_scn(random_sampling_during_train=True, with_unk=False)
    add_noise_to_testdataset()
    print("all work has finished")

