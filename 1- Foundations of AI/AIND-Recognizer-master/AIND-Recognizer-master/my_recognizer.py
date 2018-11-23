import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for i in test_set.get_all_sequences():
        X, length = test_set.get_item_Xlengths(i)
        max_probability = float("-inf") 
        best_scored_word = None
        probabilities_of_i = {}
        for word, model in models.items():
            try:
                probabilities_of_i[word] = model.score(X, length)
            except:
                continue
            
            if probabilities_of_i[word] > max_probability:
                max_probability = probabilities_of_i[word]
                best_scored_word = word
                
        probabilities.append(probabilities_of_i)
        guesses.append(best_scored_word)
        
    return probabilities, guesses















