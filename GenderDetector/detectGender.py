import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
global gender_model, graph


graph = tf.get_default_graph()
longest_name_length = 15

try:
    token_model = pickle.load(open('static/name_tokenizer.tkn', 'rb+'))
    gender_model = pickle.load(open('static/gender.model', 'rb+'))

except Exception as e:
    print(e)
    pass


def predict_gender(name):

    name_token = [[x for x in name]]
    name_vect = token_model.texts_to_sequences(name_token)
    name_padded = pad_sequences(name_vect, maxlen=longest_name_length, padding='post', value=0)

    with graph.as_default():
        gender_pred = gender_model.predict(name_padded)
        print(gender_pred[0])

    int_conv = [int(x) for x in np.round(gender_pred[0])]
    gender = ''
    confidence = gender_pred[0][0]
    if int_conv[0] == 1:
        gender = 'Male'
        pass
    if int_conv[0] == 0:
        gender = 'Female'
        confidence = 1 - confidence
        pass

    return {
        "gender":gender,
        "name":name,
        "confidence":str(confidence)
    }
   
