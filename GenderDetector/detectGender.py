import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from nltk.tag import pos_tag
import nltk
global gender_model, graph

nltk.download('averaged_perceptron_tagger')

graph = tf.get_default_graph()
longest_name_length = 15

try:
    token_model = pickle.load(open('static/name_tokenizer.tkn', 'rb+'))
    gender_model = pickle.load(open('static/gender.model', 'rb+'))

except Exception as e:
    print(e)
    pass


def predict_gender(name):

    name, tag = pos_tag([str(name)])[0]

    if tag == 'NNP' or  tag == 'NN':
        name_token = [[x for x in name]]
        name_vect = token_model.texts_to_sequences(name_token)
        name_padded = pad_sequences(name_vect, maxlen=longest_name_length, padding='post', value=0)

        with graph.as_default():
            gender_pred = gender_model.predict(name_padded)
            print(gender_pred[0][0])

        int_conv = [int(x) for x in np.round(gender_pred[0])]
        gender = ''
        if int_conv[0] == 1:
            gender = 'Male'
            pass
        if int_conv[0] == 0:
            gender = 'Female'
            pass

        return {
            "gender":gender,
            "name":name,
            "confidence":gender_pred[0][0]
        }
    else:
        return 'Unknown'
