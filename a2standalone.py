
import numpy as np
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, AutoConfig
from collections import Counter

def align_predictions_with_tokens(predictions, original_tokens_list, loaded_tokenizer):
    """
    Convert subword predictions back to token-level predictions.
    - Majority voting among subword predictions for each token.
    - In case of a tie, use the prediction from the first subword.
    
    Returns:
    --------
    token_predictions : list of lists
    """
    token_predictions = []
    # Override tokenizer if given
    if loaded_tokenizer:
        tokenizer = loaded_tokenizer
    
    for pred_seq, orig_tokens in zip(predictions, original_tokens_list):
        # Tokenize to get word_ids mapping
        tokenized = tokenizer(orig_tokens, is_split_into_words=True, truncation=True)
        word_ids = tokenized.word_ids()
        
        # Store votes and first label for each original token
        votes = {}
        first_labels = {}
        
        for i, word_idx in enumerate(word_ids):
            # Skip special tokens
            if word_idx is None:
                continue
            if orig_tokens[word_idx] == '[PRE]': #also skip the label for the special token we added 
                continue
            
            # Get label prediction for current subword
            label_id = int(pred_seq[i])
            
            # Group predictions for subwords with same word_idx
            if word_idx not in votes:
                votes[word_idx] = []
                first_labels[word_idx] = label_id
            votes[word_idx].append(label_id)
        
        #majority vote for each token
        sentence_preds = []
        for word_idx in sorted(votes.keys()):
            word_votes = votes[word_idx]
            counts = Counter(word_votes) #count label frequency per wordid
            
            #max frequency
            max_freq = max(counts.values())
            
            #Collect ALL candidates with max frequency
            candidates = []
            for label, count in counts.items():
                if count == max_freq:
                    candidates.append(label) 
            
            #use the first subword's prediction if there's a tie
            if len(candidates) > 1:
                final_label = first_labels[word_idx]
            else:
                final_label = candidates[0]
            
            sentence_preds.append(final_label)
        
        token_predictions.append(sentence_preds)
    
    return token_predictions

def predict(sentence, predicate_label, model_path):
    """
     Given a sentence in the form of a list of tokens, as well as a list of its predicate labels, the function:
    A. Duplicate sentences x many times the nr of predicates in the sentence, and mark the target predicate per sentence
    B. Use the fine-tuned BERT to predict the arguments

    Parameters
    ----------
    token_list: list[str]
        a list of tokens
    predicate_list: list[str]
        a list of predicate labels ('_'/'x');.
    model_path:
        path to the saved BERT file
    Returns
    -------
    feature, prediction
        a tuple of lists (a list of feature dicts and a list of predictions)
    """
    processed_sentence= []
    for i, pre in enumerate(predicate_label):
        if pre != '_': #find where the predicate is and augment it with the special token:
            target_sentence = list(sentence) 
            new_predicate = f'[PRE] {target_sentence[i]}'
            target_sentence[i] = new_predicate
            processed_sentence.append(target_sentence)
    
    # Load the model from the model path
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    #tokenizing sentences
    tokenized_sentence= tokenizer(processed_sentence, truncation=True, is_split_into_words=True)
    
    #initialise the trainer, create the dataset and predict
    trainer = Trainer(model=model)
    standalone_dataset = Dataset.from_dict(tokenized_sentence)
    predictions, labels, _ = trainer.predict(standalone_dataset)
    predictions = np.argmax(predictions, axis=2)
    
    #aligning the labels
    int_predictions= align_predictions_with_tokens(predictions, processed_sentence, tokenizer)

    #converting label id back to label string
    final= []
    ###code learned from: https://huggingface.co/docs/transformers/main_classes/configuration, last accessed 24 feb
    config = AutoConfig.from_pretrained(model_path)
    id2label = config.id2label
    ###
    for sent_pred_id in int_predictions:
        sent_pred_str= []
        for token_pre_id in sent_pred_id:
            token_pre_str = id2label[token_pre_id]
            sent_pred_str.append(token_pre_str)
        final.append(sent_pred_str)
        
    return processed_sentence, final