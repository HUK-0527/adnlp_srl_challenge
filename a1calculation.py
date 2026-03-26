######error rate calculation######
def target_gold_prediction(gold, prediction):
    '''
    extract the target gold and prediction per sentence
    parameter:
    -gold: list[list[str]]
    -prediction: list[list[str]]

    return:
    -ta_go: target gold list
    -ta_pr: target prediction list
    '''
    ta_go= []
    ta_pr= []


    for g_sent, p_sent in zip(gold, prediction):
        sent_go = []
        sent_pr = []
        for g_label, p_label in zip(g_sent, p_sent):
            if g_label != '_':
                sent_go.append(g_label.strip('"').strip())
                sent_pr.append(str(p_label).strip('"').strip())
        ta_go.append(sent_go)
        ta_pr.append(sent_pr)
                
    return ta_go, ta_pr

def mft_error (gold, prediction):
    '''
    Calculate the error rates for MFT tests (nr of errors/ nr of all instances)
    parameter:
    -gold: list[list[str]]
    -prediction: list[list[str]]
    prints:
    error rates
    nr referring to the error cases
    '''
    ta_go, ta_pr= target_gold_prediction(gold, prediction)
    
    instances= 0
    errors= 0
    error_case_nr= []
    for n, (g, p) in enumerate (zip(ta_go, ta_pr)):
        instances +=1
        if g != p:
            errors +=1
            error_case_nr.append(n+1)

    rate= round((errors/instances),2)
    print(f'The MFT/general error rate is {rate}. Case(s) {error_case_nr} is/are wrong')


def inv_two_datasets (gold1, prediction1, gold2, prediction2):
    '''
    Calculate the error rates for INV tests across two datasets.
    Two types of error rates are calculated:
    -general rates: (number of flipped predictions)/total pairs
    -conditional rates: 
    Cases where [(gold1== gold2 == prediction1) AND (prediction1 !=prediction2)]/ total cases where (gold1=gold2 = prediction1)
    
    parameter:
    -gold1: list[list[str]]
    -gold2: list[list[str]]
    -prediction1: list[list[str]]
    -prediction2: list[list[str]]
    prints:
    error rates and nr referring to the error cases
    '''
    ta_go1, ta_pr1= target_gold_prediction(gold1, prediction1)
    ta_go2, ta_pr2= target_gold_prediction(gold2, prediction2)
    instances= 0
    p1_correct= 0
    inconsistency1= 0
    inconsistency2= 0
    raw_error_rate= 0
    p1correct_error_rate= 0
    incon_error_case_nr= []
    robust_error_case_nr= []
    for n, (g1, g2, p1, p2) in enumerate (zip(ta_go1, ta_go2, ta_pr1, ta_pr2)):
        
        ##1. Calculate the raw inconstitency rate
        instances +=1
        if p1!=p2:
            inconsistency1 +=1
            incon_error_case_nr.append(n+1)

        ##2. Calculate the conditional rate only when pred1 is correct
        if p1 == g1 ==g2:
            p1_correct +=1
            if p1!=p2:
                inconsistency2 +=1
                robust_error_case_nr.append(n+1)

    raw_error_rate= round((inconsistency1/instances),2)
    print(f'The raw error rate is {raw_error_rate}, with {incon_error_case_nr} wrong')
    
    if  p1_correct !=0:
        p1correct_error_rate= round((inconsistency2/p1_correct),2)
        print(f'The conditional error rate (i.e. cases where starting sentences are correctly predicted) is {p1correct_error_rate}, with {robust_error_case_nr} wrong')
    else:
        print(f'1.0: all first sentences are wrongly predicted')

def inv_onedataset (gold, prediction):
    '''
    Calculate the error rates for INV tests in one dataset.
    Two types of error rates are calculated:
    -general rates: (number of flipped predictions)/total pairs
    -conditional rates: 
    Cases where [(gold1 == prediction1) AND (prediction1 !=prediction2)]/ total cases where (gold1 = prediction1)
    
    parameter:
    -gold: list[list[str]]
    -prediction: list[list[str]]
    prints:
    error rates and nr referring to the error cases
    '''
    ta_go, ta_pr= target_gold_prediction(gold, prediction)
    
    p1_correct= 0
    inconsistency1= 0
    inconsistency2= 0
    raw_error_rate= 0
    p1correct_error_rate= 0
    incon_error_case_nr= []
    robust_error_case_nr= []

    for target_index in range(0, len(ta_go) - 1, 2):
        n = target_index // 2 #find out which case
        ##finding out the sentences in question
        p1= ta_pr[target_index]
        case_gold= ta_go [target_index]
        p2= ta_pr[target_index+1]
        
        ##1. Calculate the raw inconstitency rate
        if p1!=p2:
            inconsistency1 +=1
            incon_error_case_nr.append(n+1)

        ##2. Calculate the inconstitency rate only when pred1 is correct
        if p1 == case_gold:
            p1_correct +=1
            if p1== case_gold!=p2:
                inconsistency2 +=1
                robust_error_case_nr.append(n+1)
        
    total_pairs = len(ta_go) // 2
    raw_error_rate= round((inconsistency1/total_pairs),2)
    print(f'The raw error rate is {raw_error_rate}, with pair {incon_error_case_nr} wrong')
    print()
    if  p1_correct !=0:
        p1correct_error_rate= round((inconsistency2/p1_correct),2)
        print(f'The conditional error rate (i.e. cases where starting sentences are correctly predicted) is {p1correct_error_rate}, with pair {robust_error_case_nr} wrong')
    else:
        print(f'all first sentences are wrongly predicted')

def dir_onedataset (gold, prediction):
    '''
    Calculate the error rates for DIR tests in one dataset.
    Here, the expectation is that the label MUST change after input changes.
    - raw error rates: cases where the label failed to change (p1 == p2) / total pairs
    - robustness-error rates: cases where [gold1 == p1 AND p1 == p2] / total cases where gold1 == p1
    
    parameter:
    -gold: list[list[str]]
    -prediction: list[list[str]]
    prints:
    error rates and nr referring to the error cases
    '''
    ta_go, ta_pr= target_gold_prediction(gold, prediction)
    
    p1_correct= 0
    error1= 0
    error2= 0
    raw_error_rate= 0
    p1correct_error_rate= 0
    error_case_nr= []
    robust_error_case_nr= []

    for target_index in range(0, len(ta_go) - 1, 2):
        n = target_index // 2 #find out which case
        ##finding out the sentences in question
        p1= ta_pr[target_index]
        g1= ta_go [target_index]
        p2= ta_pr[target_index+1]
        g2= ta_go [target_index+1]
        
        ##1. Calculate the raw inconstitency rate
        if p1==p2:
            error1 +=1
            error_case_nr.append(n+1)

        ##2. Calculate the inconstitency rate only when pred1 is correct
        if p1 == g1:
            p1_correct +=1
            if p1==p2:
                error2 +=1
                robust_error_case_nr.append(n+1)
        
    total_pairs = len(ta_go) // 2
    raw_error_rate= round((error1/total_pairs),2)
    print(f'The raw error rate is {raw_error_rate}, with pair {error_case_nr} wrong')
    print()
    if  p1_correct !=0:
        p1correct_error_rate= round((error2/p1_correct),2)
        print(f'The conditional error rate (i.e. cases where starting sentences are correctly predicted) ia {p1correct_error_rate}, with pair {robust_error_case_nr} wrong')
    else:
        print(f'all first sentences are wrongly predicted')
