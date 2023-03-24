import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

stemmer = PorterStemmer()
stops = set(stopwords.words('english'))

ignore_words = ['?', '.', '!','"',',',':']

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha()]
    return words

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word)
    #return word

def bag_of_words(sentence, text):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """

    #first prepare corpus (text) content
    #print(stops)
    words = [stem(w) for w in tokenize(text) if w not in stops]
    #next remove duplicates and sort, which will alphabatize
    words = sorted(set(words))
    print(words)

    ##second prep the sentence(s)
    #not sure if we want to sort, need to experiment
    sentence_words = [stem(word) for word in tokenize(sentence) if word not in stops]

    print('size of sorted text: ',len(words))
    ## initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
