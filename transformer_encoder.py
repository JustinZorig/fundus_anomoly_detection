from tensorflow import keras
from tensorflow.keras import layers


################## Stupid method

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout = 0):
    # normalization and attention

    x = layers.LayerNormalization(epsilon = 1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim = head_size, num_heads = num_heads, dropout = dropout)(x,x)
    x = layers.Dropout(droput)(x)
    res = x+inputs
    
    # feedforward part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters= ff_dim, kernel_size =1, activation = "relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=input.shape[-1], kernel_size=1)(x)

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_droput=0
):
    inputs = keras.Input(shape = input_shape)    
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encocder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation = 'relu')
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation = 'softmax')(x)
    return keras.Model(inputs, outputs)


################## Smarter method

def positional_encoding():
    return True

#class PositionalEmbedding(tf.keras.layers.Layer):
    #def __init__(self, vocab_size)    


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size = 1):
        super(Time2Vec, self).__init__(trainable = True, name = 'Time2VecLayer')
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        # Trend
        self.wb = self.add_weight(name = 'wb', shape =(input_shape[1], ),
                                 initializer = 'uniform', trainable = True)
        
        self.bb = self.add_weight(name = 'bb', shape =(input_shape[1], ),
                                 initializer = 'uniform', trainable = True)                                 
        # Periodic
        self.wa = self.add_weight(name = 'wa', shape = (1, input_shape[1], self.kernel_size),
                                initializer = 'uniform', trainable = True)
        self.ba = self.add_weight(name = 'ba', shape = (1, input_shape[1], self.kernel_size),
                                initializer = 'uniform', trainable = True)     

        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb                          
        dp = keras.dot(inputs, self.wa) +self.ba
        wgts = keras.sin(dp) # cos works too

        ret = keras.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = keras.reshape(ret, (-1, inputs.shape[1]* (self.kernel_size+1)))

        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.kernel_size +1))


from tensorflow_addons.layers import MultiHeadAttention
class AttentionBlock(keras.Model):
    def __init__(self, name = 'Attention block', num_heads=2, head_size = 128,
                 ff_dim = None, dropout=0, **kwargs):
        super().__init__(name =name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size
        self.attention = MultiHeadAttention(num_heads = num_heads, head_size = head_size, dropout = dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters = ff_dim, kernel_size=1, activation ='relu')

        self.ff_dropout= keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 =keras.layers.Conv1D(filters = input_shape[-1], kernel_size =1)
    
    def call(self, inputs):
        x = self.attention([inputsm, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs+x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs+x)
        return x

class ModelTrunk(keras.Model):
    def __init__(self, name = 'ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128,
                ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers  =[AttentionBlock(num_heads = num_heads, 
                                head_size=head_size, ff_dim=ff_dim, dropout=dropout)
                                for _ in range(num_layers)]

        def call(self, inputs):
            time_embedding= keras.layers.TimeDistributed(self.time2vec)(inputs)
            x = keras.concatenate([inputs, time_embedding], -1)
            for attention_layer in self.attention_layers:
                x = attention_layer(x)

            return keras.reshape(x, (-1, x.shape[1]* x.shape[2])) # flat vector of features out



