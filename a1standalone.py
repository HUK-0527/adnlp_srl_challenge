import spacy
from spacy.tokens import Doc
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pickle

def logreg_srl (token_list, predicate_list, modelfile, vecfile):
    """
    Given a sentence in the form of a list of tokens, as well as a list of its predicate info (WITH ONE PREDICATE ONLY), the function:
    A. Extracts features for SRL in the sentence:
    1) token
    2) token_head+head_pos
    3) predicate's lemma + directed ud path
    B. Use the model and vectorizer trained to predict the per token in the given list 

    Parameters
    ----------
    token_list: list[str]
        a list of tokens
    predicate_list: list[str]
        a list of predicate labels ('_'/'x'); MUST CONTAIN ONLY ONE 'x'.
     modelfile: str
        path to the logreg model
    vecfile: str
        path to the logreg vectorizer trained
   
    Returns
    -------
    feature, prediction
        a tuple of lists (a list of feature dicts and a list of prediction)
    """
    #------------ A: Feature extraction----------------
    # load the parser and parse the sentence: it is the same no matter nr of predicates
    nlp = spacy.load("en_core_web_lg")
    spacy_doc = Doc(nlp.vocab, words= token_list)
    processed_doc = nlp(spacy_doc)

    #extract predicate indices
    pre_index= 0 
    for i, prel in enumerate (predicate_list):
        if prel == 'x':
            pre_index = i
            
    #create a list to collect feature dict
    feature= []

    path_prelemma = "NONE"
    #focus on the target predicate
    predicate= processed_doc[pre_index]
    predicate_ancestor= [predicate]+ list(predicate.ancestors)

    #now loop over every token in the sentence
    for token in processed_doc:
    
        # extracting NER type
        ###attributes inspried by https://spacy.io/api/token, last access 22 March, 2026
        NER= token.ent_type_
        if NER== '':
            NER= 'O'
        ###

        #extracting predicate's lemma + directed UD path through ancestors using token.ancestors
    
        #finding the ancestors of the token
        token_ancestor= [token]+ list(token.ancestors)

        # a token and a predicate can either be direct ancestors of each other or be in a different branch but share a common ancestor
    
        # 1: if predicate is the direct ancestor of the token, the directed path goes up.
        if predicate in token_ancestor:
            pre_index= token_ancestor.index(predicate) 
            path_token= token_ancestor[:pre_index] #we slice where the predicate is
            path= [('↑'+ t.dep_) for t in path_token]
            path_string= ''.join(path)
            path_prelemma= f'{predicate.lemma_}_{path_string}'
        
        
        # 2: if token is a direct ancestor of the predicate, the directed path goes down.
        elif token in predicate_ancestor:
            to_index= predicate_ancestor.index(token)
            path_token= predicate_ancestor[:to_index]
            path= [('↓'+t.dep_) for t in path_token[::-1]] #here list is reversed as we are mapping from tokens to predicate rather than the other way around.
            path_string= ''.join(path)
            path_prelemma= f'{predicate.lemma_}_{path_string}'
                
        #3. Token has an indirect relation w.r.t predicate: find the first shared node between the two ancestor lists
    
        else: 
            common_ancestor = None
            for a in token_ancestor:
                if a in predicate_ancestor:
                    common_ancestor = a
                    break
            
            if common_ancestor:
                 #fix 21 March: down path needs to be reversed
                if common_ancestor:
                    # Up from token to common_ancestor
                    up_path = [('↑'+t.dep_)for t in token_ancestor[:token_ancestor.index(common_ancestor)]]
                     # Down from common_ancestor to predicate
                    predicate_ca_path= predicate_ancestor[:predicate_ancestor.index(common_ancestor)]
                    down_path = [('↓'+t.dep_) for t in predicate_ca_path[::-1]]
                
                    path_string = ''.join(up_path + down_path)
                    path_prelemma= f'{predicate.lemma_}_{path_string}'
                else:
                    # safety: If  no connection at all
                    path_prelemma= f'{predicate.lemma_}_NO_PATH'
            
        #create a feature dict per token and collect     
        feature_dict= {'token': token.text, 'NER': NER, 'pre_lemma_ud_path': path_prelemma}
        feature.append(feature_dict)

    #------------ B: Prediction----------------
     #load the model and vectorizer
    loaded_classifier = pickle.load(open(modelfile, 'rb'))
    loaded_vectorizer = pickle.load(open(vecfile, 'rb'))

    #vectorise the test_feature
    vec_feature= loaded_vectorizer.transform(feature)

    #now predict
    predictions = loaded_classifier.predict(vec_feature)

    return feature, predictions
