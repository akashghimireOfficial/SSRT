import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.resnet183d import resnet3d


tf.random.set_seed(1234)
np.random.seed(1234)

### building tensorflow-encoder from scratch

### defining positional Encoding


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                        np.arange(d_model)[np.newaxis, :],
                        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class pos_encoding(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(**kwargs).__init__()
        
        
    

    
    def build(self,input_shape):
        _,seq_len,self.d_model=input_shape
        #self.positional_embedding=tf.keras.layers.Embedding(input_dim=seq_len,output_dim=self.d_model)
        self.positional_encoding=positional_encoding(seq_len,self.d_model)
        #self.position=tf.range(start=0,limit=seq_len,delta=1)
    

    
    def call(self,inputs):
        #pos_emb=self.positional_embedding(self.position)
        #pos_emb=pos_emb+self.positional_encoding
        return self.positional_encoding+ inputs
        #return inputs + pos_emb





### defining multi head attention layer


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    ## shape is determined by simple matrix multiplication rule 

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    #dk is the square root dimension of keys
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights




class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads): ##(self,*,d_model=dimension of k,num_heads)
        
        ##  d_model refer to the dimension of the q,v,k
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model) ## Must Define dense layer separately for all because Dense() always expect same shape but q,v,k may have different shape
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
    def split_heads(self, x, batch_size):
    #     Split the last dimension into (num_heads, depth).
    #     Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

         #because of this split the computational time is same 
         #even when the number of heads is increased
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # q.shape= (batch_size, seq_len, d_model) ;Change the last dimesnion to d_model
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model) 
        ## this must assume the d_model and feature length is same 

        return output, attention_weights


## Point wise feed forward network

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=tf.nn.gelu),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])





## Defining the embedding Class



## defining the encoder layer 

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # att_ouptput.shape=(batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        ##d_model must be equal to feature_dimension otherwise error in addition x + attn_output
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.dropout2(ffn_output, training=training)
        
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2




## Decoder Layer

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
               mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x,mask) ## v,k,q order  
        # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1,mask)  ## v and k are from encoder 
        # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2





class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, d_model, num_heads, dff,
               rate=0.1,pos=True,lstm_enc=True,pos_enc=False):
        super(TransformerEncoder,self).__init__()
        self.pos=pos
        self.lstm_enc=lstm_enc
        self.pos_enc=pos_enc
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding=pos_encoding()
        self.projection=tf.keras.layers.Dense(units=self.d_model,activation='relu')
        self.lstm=tf.keras.layers.LSTM(units=self.d_model,return_sequences=True)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)


   
    

    def call(self, x, training, mask):
        if self.pos:
            if self.lstm_enc:
                x=self.lstm(x)
                x=self.layernorm(x)
            if self.pos_enc:
                x=self.pos_encoding(x)
                x=self.projection(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


## Dec

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, d_model, num_heads, dff,
                   rate=0.1,pos=True,lstm_enc=True,pos_enc=False):
        super(TransformerDecoder, self).__init__()
        self.pos=pos
        self.lstm_enc=lstm_enc
        self.pos_enc=pos_enc
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding=pos_encoding()
        self.projection=tf.keras.layers.Dense(units=self.d_model,activation='relu')
        self.lstm=tf.keras.layers.LSTM(units=self.d_model,return_sequences=True)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
               mask):
        if self.pos:
            if self.lstm_enc:
                x=self.lstm(x)
                x=self.layernorm(x)
            if self.pos_enc:
                x=self.pos_encoding(x)
                x=self.projection(x)
        attention_weights = {}

        

        

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights




class Transformerfusion(tf.keras.Model):
    def __init__(self,*, num_layers, d_model, num_heads, dff,
                    rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                                rate=rate,pos=True,lstm_enc=True,pos_enc=False)

        self.decoder = TransformerDecoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               rate=rate,pos=True,lstm_enc=True,pos_enc=False)

        self.resnet=resnet3d()

        

    def call(self, inputs, training,mask):
        # Keras models prefer if you pass all your inputs in the first argument
        rgb, pose = inputs

        
        rgb=self.resnet(rgb)
        enc_output = self.encoder(rgb, training, mask)  # (batch_size, inp_seq_len, d_model)

        # tar=self.resnet(tar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            pose, enc_output, training, mask)

        
        return dec_output,enc_output


class Transformerfusion2(tf.keras.Model):
    def __init__(self,*, num_layers, d_model, num_heads, dff,
                    rate=0.1):
        super(Transformerfusion2,self).__init__()
        self.encoder_pose = TransformerEncoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                                rate=rate+0.1,pos=True,lstm_enc=True,pos_enc=False)


        self.encoder_rgb = TransformerEncoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                                rate=rate,pos_enc=False,pos=True,lstm_enc=True)
        

        self.decoder_pose = TransformerDecoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               rate=rate+0.1,pos=False,lstm_enc=False,pos_enc=False)

        
       
        

        self.decoder_rgb = TransformerDecoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               rate=rate+0.1,pos=False,lstm_enc=False,pos_enc=False)



        self.encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               rate=rate,pos=True,lstm_enc=False,pos_enc=True)

        self.resnet=resnet3d()
        

        

    def call(self, inputs, training,mask):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        
        inp=self.resnet(inp)
        enc_output_pose = self.encoder_pose(tar, training, mask)  # (batch_size, inp_seq_len, d_model)

       

        enc_output_rgb = self.encoder_rgb(inp, training, mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output_rgb, attention_weights_rgb = self.decoder_rgb(
            enc_output_pose, enc_output_rgb, training, mask)

        dec_output_pose, attention_weights_pose = self.decoder_pose(
            enc_output_rgb, enc_output_pose, training, mask)


        merged_output=tf.keras.layers.concatenate([dec_output_rgb,dec_output_pose])

        final_output=self.encoder(merged_output,training=training,mask=mask)

        

        return final_output



         

class TransformerEarlyConcat(tf.keras.Model):
    def __init__(self,*, num_layers, d_model, num_heads, dff,
                    rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                                rate=rate,pos=True,lstm_enc=False,pos_enc=True)

        

        

    def call(self, inputs, training,mask):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        

        merged_features=tf.keras.layers.Concatenate([inp,tar])

        final_output=self.encoder(merged_features,training,mask)

        
        return final_output
