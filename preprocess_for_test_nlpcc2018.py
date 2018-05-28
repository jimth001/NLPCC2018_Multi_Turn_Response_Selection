import random
from Tokenizer import Tokenizer
import numpy as np
import pickle
import word2vec
import Evaluate

def load_context_data(path='./nlpcc2018test/seq_context.txt'):
    contexts=[]
    with open(path,'r',encoding='utf-8') as f:
        a_context=[]
        for line in f:
            string=line.strip('\r\n').strip('\n')
            if len(string)==0:
                if len(a_context)==0:
                    continue
                else:
                    contexts.append(a_context)
                    a_context=[]
            else:
                a_context.append(string)
    return contexts

def load_responses_data(path='./nlpcc2018test/seq_replies.txt'):
    all_context_responses=[]
    with open(path,'r',encoding='utf-8') as f:
        one_context_responses=[]
        for line in f:
            string=line.strip('\r\n').strip('\n')
            if len(string)==0:
                if len(one_context_responses)==0:
                    continue
                else:
                    all_context_responses.append(one_context_responses)
                    one_context_responses=[]
            else:
                one_context_responses.append(string)
    return all_context_responses

def gen_raw_test_data(store_path='./nlpcc2018test/test.raw.pkl',add_sen=False):
    contexts=load_context_data()
    all_context_response=load_responses_data()
    assert len(contexts)==len(all_context_response),'test data num error'
    tmp=np.array([len(r) for r in all_context_response])
    wrong_index=[i for i in range(0,len(tmp)) if tmp[i]!=10]
    wrong_contents=[all_context_response[i] for i in range(0,len(tmp)) if tmp[i]!=10]
    if add_sen:
        addition_string = 'nlpcc2018错误，异常。补齐候选句子。aaaaaa'
        for i in range(0,len(tmp)):
            if tmp[i]!=10:#统计过了nlpcc2018的数据不是10就是9。有几个9的。9是错误的。
                all_context_response[i].append(addition_string)
    new_context=[]
    new_response=[]
    for i in range(0,len(contexts)):
        for j in range(0,tmp[i]):
            new_context.append(contexts[i])
    for context_responses in all_context_response:
        for r in context_responses:
            new_response.append(r)
    assert len(new_context)==len(new_response),'new_context or new_response length error'
    pickle.dump([new_context,new_response],open(store_path,'wb+'),protocol=True)
    pickle.dump(tmp,open('./nlpcc2018test/reponses_num_of_context.pkl','wb+'),protocol=True)
    pickle.dump([wrong_contents,wrong_index], open('./nlpcc2018test/wrong_contents_and_index.pkl', 'wb+'), protocol=True)
    return new_context,new_response

def get_seged_test_data(raw_data,store_path='./nlpcc2018test/test.seg.pkl'):
    raw_context, raw_response=raw_data
    tokenizer=Tokenizer()
    seged_context=[]
    seged_response=[]
    for con in raw_context:
        one_seged_con=[]
        for utt in con:
            one_seged_con.append(tokenizer.parser(utt).split())
        seged_context.append(one_seged_con)
    for r in raw_response:
        seged_response.append(tokenizer.parser(r).split())
    pickle.dump([seged_context,seged_response],open(store_path,'wb+'),protocol=True)
    return seged_context,seged_response

def get_final_test_data_without_unk(seg_context,seg_response,word_dict,store_path='./nlpcc2018test/test.pkl'):
    index_context=[]
    index_response=[]
    for con in seg_context:
        index_one_con=[]
        for utt in con:
            index_one_utt=[word_dict[word] for word in utt if word in word_dict]
            index_one_con.append(index_one_utt)
        index_context.append(index_one_con)
    for r in seg_response:
        index_one_r=[word_dict[word] for word in r if word in word_dict]
        index_response.append(index_one_r)
    pickle.dump([index_context,index_response],open(store_path,'wb+'),protocol=True)

def load_word_embedding():
    # word_embedding:[clusters=None,vectors,vocab,vocab_hash]
    word_embedding = word2vec.load('./data/word_embedding.bin')
    return word_embedding

def gen_result_readable():
    reponses_num_of_context=pickle.load(open('./nlpcc2018test/reponses_num_of_context.pkl','rb'))
    final_result=pickle.load(open('./nlpcc2018result/final.pkl','rb'))
    contexts = load_context_data()
    all_context_response = load_responses_data()
    p=0
    with open('./nlpcc2018result/final.readable.txt','w+') as f:
        for i in range(0,len(contexts)):
            num=reponses_num_of_context[i]
            tmp=final_result[p:p+num]
            tmp=[[tmp[i],i] for i in range(0,len(tmp))]
            tmp.sort(reverse=True)
            f.write('id='+str(i)+'\n')
            for utt in contexts[i]:
                f.write(utt+'\n')
            f.write('responses ranking:\n')
            for score,index in tmp:
                f.write(str(index)+'    '+str(score)+'    '+all_context_response[i][index]+'\n')
            f.write('\n')
            p+=num
    assert p==len(final_result),'gen_result_readable error'

def load_true_label():
    label=[]
    with open('./nlpcc2018result/sub2-index.txt') as f:
        for line in f:
            label.append(int(line.strip('\r\n').strip('\n')))
    return label

if __name__=='__main__':
    #gen_raw_test_data()
    #seg_context,seg_response=get_seged_test_data(gen_raw_test_data(add_sen=False))
    #word_embedding=load_word_embedding()
    #get_final_test_data_without_unk(seg_context=seg_context,seg_response=seg_response,word_dict=word_embedding.vocab_hash)
    #gen_result_readable()
    label = load_true_label()
    all_result = pickle.load(open('./nlpcc2018result/all.pkl', 'rb'))
    wrong_contents_and_index=pickle.load(open('./nlpcc2018test/wrong_contents_and_index.pkl','rb'))
    test_raw=pickle.load(open('./nlpcc2018test/test.raw.pkl','rb'))
    final_result = pickle.load(open('./nlpcc2018result/final.pkl', 'rb'))
    new_all_rst = []
    for rst in all_result[0]:
        tmp = list(rst)
        new_all_rst.append(tmp)
    new_final_result=list(final_result)
    for index in wrong_contents_and_index[1]:
        for rst in new_all_rst:
            rst.insert(index*10+9,0.0)
        new_final_result.insert(index*10+9,0.00001)
    new_label=[]
    for l in label:
        new_label+=[int(l==i) for i in range(0,10)]
    for i in range(0,len(all_result[0])):
        print(Evaluate.mrr_and_rnk(new_all_rst[i],new_label,response_num_per_query=10))
    print(Evaluate.mrr_and_rnk(new_final_result,new_label,response_num_per_query=10))
    print('all work has finished')




