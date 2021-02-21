# import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pickle

# tf.config.run_functions_eagerly(True)


class Model:
    def __init__(self):

        self.VOCAB_SIZE = 53
        self.MAX_LEN = 20
        self.EMBEDDING_SIZE = 100
        self.pretrained_model =  keras.models.load_model('/static/gender_model.h5')
        self.MALE = [0.0,1.0]
        self.FEMALE = [1.0, 0.0]
        pass


    def load_existing_models(self):
        try:
            with open('/static/char_to_number.enc', 'rb') as encoder:
                char_encoder = pickle.load(encoder)
                pass

            pretrained_model =  keras.models.load_model('/static/gender_model.h5')

            return pretrained_model, char_encoder            
            

        except Exception as e:
            print(e)
            return None, None
        pass


    def predict(self, name_tokens = None):
        if name_tokens is None:
            return None
        
        pred = self.pretrained_model.predict(name_tokens)[0]
        print(pred[0])
        confidence_score = {
            "FEMALE": "{:.2f}%".format(pred[0] * 100),
            "MALE":"{:.2f}%".format(pred[1] * 100)
        }
        round_number = np.round(pred)
        if round_number[0] == 1:
            gender = "FEMALE"
            pass
        else:
            gender = "MALE"
            pass
        return {
            "gender":gender,
            "confidence":confidence_score
        }
        



    def get_model(self):
        _input = keras.Input(shape=(self.MAX_LEN,))
        embedding = keras.layers.Embedding(self.VOCAB_SIZE, self.EMBEDDING_SIZE, input_length=self.MAX_LEN)(_input)
        bidirectional_lstm = self.BI_LSTM_BLOCK(64)
        bidirectional_lstm_2 = self.BI_LSTM_BLOCK(128, return_sequence=True)



        blk = bidirectional_lstm(embedding)
        blk_2 = bidirectional_lstm_2(embedding)
        bl_2 = keras.layers.Flatten()(blk_2)

        concat = keras.layers.Concatenate()([blk, bl_2])

        dense = keras.layers.Dense(192, activation="relu")(concat)
        dense = keras.layers.Dense(64, activation="relu")(dense)
        output = keras.layers.Dense(2, activation="sigmoid")(dense)

        model = keras.Model(_input, output)


        opt = keras.optimizers.Adam(lr=1e-3)
        loss = keras.losses.BinaryCrossentropy()
        model.compile(metrics=["accuracy"], loss=loss,optimizer=opt)


        return model


    def BI_LSTM_BLOCK(self, units,return_sequence = False, input_shape=None):
        fwd = keras.layers.LSTM(units, return_sequences=return_sequence, recurrent_dropout=0.2, input_shape=(None, self.MAX_LEN, self.EMBEDDING_SIZE))
        bkw = keras.layers.LSTM(units, return_sequences=return_sequence, go_backwards=True, recurrent_dropout=0.2, input_shape=(None, self.MAX_LEN, self.EMBEDDING_SIZE))
        if input_shape != None:
            lstm = keras.layers.Bidirectional(fwd, bkw, input_shape=input_shape)
            pass
        else:
            lstm = keras.layers.Bidirectional(layer=fwd, backward_layer=bkw)
            pass

        return lstm    