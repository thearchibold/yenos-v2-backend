from Model import Model
from tensorflow import keras
import nltk



class Preprocessor:
    def __init__(self):
        model = Model()
        self._, self.encoder = model.load_existing_models()
        self.VOCAB_SIZE = 53
        self.MAX_LEN = 20
        self.EMBEDDING_SIZE = 100
        pass


    def process_name(self, name = ""):
        if name == "":
            return None
        # convert the characters into numbers
        name_token = [self.encoder.get(ch) for ch in name]
        
        # pad to the max length
        tokens = keras.preprocessing.sequence.pad_sequences([name_token], maxlen=self.MAX_LEN, truncating="post", padding="post", value=0)
        
        return tokens