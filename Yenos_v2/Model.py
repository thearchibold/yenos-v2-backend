# import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K


# tf.config.run_functions_eagerly(True)


class Model:
    def __init__(self):
        self.VOCAB_SIZE = 54
        self.MAX_LEN = 20
        self.EMBEDDING_SIZE = 100
        self.pretrained_model = self.get_model()
        self.pretrained_model.load_weights("static/weight_v2")
        # self.pretrained_model = keras.models.load_model('static/gender_model.h5')
        self.MALE = [0.0,1.0]
        self.FEMALE = [1.0, 0.0]
        print("Done Loading and Initialising params")
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
        # building the model

        _input = keras.Input(shape=(20,))
        embedding = keras.layers.Embedding(self.VOCAB_SIZE, 100, input_length=self.MAX_LEN)(_input)
        bidirectional_lstm = self.BI_LSTM_BLOCK(128)
        bidirectional_lstm_2 = self.BI_LSTM_BLOCK(256, return_sequence=True)



        blk = bidirectional_lstm(embedding)
        blk_2 = bidirectional_lstm_2(embedding)

        att_1 = self.Attention(blk,128)
        att_2 = self.Attention(blk_2,256)
        # bl_2 = keras.layers.Flatten()(blk_2)

        concat = keras.layers.Concatenate()([att_1, att_2])

        dense = keras.layers.Dense(192, activation="relu")(concat)
        dense = keras.layers.Dense(64, activation="relu")(dense)
        output = keras.layers.Dense(2, activation="sigmoid")(dense)


        model = keras.Model(inputs=_input, outputs=output)


        opt = keras.optimizers.Adam(lr=1e-3)
        loss = keras.losses.CategoricalCrossentropy()
        metrics = keras.metrics.CategoricalAccuracy("acc")
        model.compile(metrics=metrics, loss=loss,optimizer=opt)

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

    def Attention(self, activations, last_unit = 64):
        attention = keras.layers.Dense(1, activation="tanh")(activations)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation("softmax")(attention)
        attention = keras.layers.RepeatVector(last_unit * 2)(attention)
        attention = keras.layers.Permute([2,1])(attention)
        
        sent_representation = keras.layers.Multiply()([activations, attention])
        sent_representation = keras.layers.Lambda(lambda xin: K.sum(xin, axis= -2), output_shape=last_unit*2)(sent_representation)
        
        return sent_representation