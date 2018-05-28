from sklearn.metrics import classification_report

def ComputeR10_1(scores,labels,count = 11):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    return str(float(correct)/ total)

def ComputeR2_1(scores,labels,count = 2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    return str(float(correct)/ total)

def precision_of_classification(pred_label,true_label,class_num=2):
    assert class_num==2,'You should design this function for your own task.'
    return classification_report(true_label, pred_label, target_names=['not_match','can_match'])

def mrr_and_rnk(pred_score,true_label,response_num_per_query=11,k=[1,2,3,4,5]):
    #一个query只有一个正确的response的时候用这个
    assert len(pred_score) == len(true_label), 'length not same'
    result_of_rnk=[0.0]*len(k)
    p=0
    sample_num=len(true_label)
    total_q_num=0.0
    mrr_result=0.0
    while p<sample_num:
        one_q_p = pred_score[p:p + response_num_per_query]
        one_q_t = true_label[p:p + response_num_per_query]
        right_index = [i for i in range(0, len(one_q_t)) if one_q_t[i] == 1]
        assert len(right_index) == 1, 'true label is not right'
        pred_of_right_sample = one_q_p[right_index[0]]
        index_of_value_of_bigger_than_valueof_oneqp_of_location_rightindex = [i for i in range(0, len(one_q_p)) if
                                                                              pred_of_right_sample <= one_q_p[i]]  #
        mrr_result+=1.0/len(index_of_value_of_bigger_than_valueof_oneqp_of_location_rightindex)
        rank=len(index_of_value_of_bigger_than_valueof_oneqp_of_location_rightindex)
        for i in range(0,len(k)):
            if rank<=k[i]:
                result_of_rnk[i]+=1
        total_q_num += 1
        p += response_num_per_query
    mrr_result=mrr_result/total_q_num
    result_of_rnk=[r/total_q_num for r in result_of_rnk]
    return mrr_result,result_of_rnk


def precision_of_matching_1(pred_label,true_label,response_num_per_query=11):
    assert len(pred_label)==len(true_label),'length not same'
    p=0
    sample_num=len(true_label)
    right_q_num=0.0
    total_q_num=0.0
    while p<sample_num:
        one_q_p=pred_label[p:p+response_num_per_query]
        one_q_t=true_label[p:p+response_num_per_query]
        right_index=[i for i in range(0,len(one_q_t)) if one_q_t[i]==1]
        assert len(right_index)==1,'true label is not right'
        tmp=one_q_p[right_index[0]]
        index_of_value_of_bigger_than_valueof_oneqp_of_location_rightindex=[i for i in range(0,len(one_q_p)) if tmp<=one_q_p[i]]#
        if len(index_of_value_of_bigger_than_valueof_oneqp_of_location_rightindex)==1:
            right_q_num+=1
        total_q_num+=1
        p+=response_num_per_query
    return  right_q_num,total_q_num

def map_and_rnk(pred_score,true_label,response_num_per_query=11,k=[1,2,3,4,5]):
    #todo 一个query有多个正确的response时用这个
    pass
