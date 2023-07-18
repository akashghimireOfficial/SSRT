import tensorflow as tf

class lstm(tf.keras.layers.Layer):

    def __init__(self):
        super(lstm,self).__init__()

        self.lstm1=tf.keras.layers.LSTM(units=512,return_sequences=True)
        self.layernorm1=tf.keras.layers.LayerNormalization()
        self.lstm2=tf.keras.layers.LSTM(units=512,return_sequences=True)
        self.layernorm2=tf.keras.layers.LayerNormalization()
        self.lstm3=tf.keras.layers.LSTM(units=128,return_sequences=False)
        self.layernorm3=tf.keras.layers.LayerNormalization()

    def call(self,x):
        x=self.lstm1(x)
        x=self.layernorm1(x)
        x=self.lstm2(x)
        x=self.layernorm2(x)
        x=self.lstm3(x)
        x=self.layernorm3(x)
        return x

class lstm_fusion(tf.keras.layers.Layer):
    def __init__(self):

        self.lstm=lstm()

        def call(self,inputs):
            inp,tar=inputs
            merged_features=tf.keras.layers.concatenate([inp,tar])
            output=self.lstm(merged_features)
            return output