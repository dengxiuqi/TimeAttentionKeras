# coding: utf-8
from keras.models import Model
import keras.backend as K
from keras.layers import *
from keras.activations import tanh

learning_rate = 1e-3
target_length = 2
embedding_size = 64
attention_units = 64
encoder_hidden_size = 64
decoder_hidden_size1 = 128
decoder_hidden_size2 = 64


def network(seq_length, vocab_size, target_length=target_length,
            embedding_size=embedding_size, attention_units=attention_units, encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size1=decoder_hidden_size1, decoder_hidden_size2=decoder_hidden_size2):
    # Input
    encoder_input = Input(shape=(seq_length, ), name="encoder_input")     # input of encoder
    decoder_input = Input(shape=(target_length, ), name="decoder_input")  # input of decoder
    fcn_input = Input(shape=(target_length, ), name="fcn_input")          # the input of bypass network
    mask = Input(shape=(seq_length, ), name="mask")                       # mask

    # Embedding, embedding 3 inputs by the same embedding layer
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True, name="embeddings")
    encoder_embed = embedding(encoder_input)
    decoder_embed = embedding(decoder_input)
    fcn_embed = embedding(fcn_input)

    # Encoder, 1 BiDirectional LSTM layer
    BiLSTM = Bidirectional(LSTM(encoder_hidden_size,
                                return_sequences=True,
                                name="encoder"),
                           merge_mode='concat')
    # encoder_output is the output of the Encoder at each time stamp, as h1, h2, h3 = Encoder(x1, x2, x3)
    encoder_output = BiLSTM(encoder_embed)
    # add mask to the encoder_output as the context information of the attention machine
    context = multiply([encoder_output, Reshape((-1, 1))(mask)], name="context")


    # Attention
    Wa_s = Dense(attention_units, name="Wa_s")(decoder_embed)     # Wa*s(i-1)
    Ua_h = Dense(attention_units, name="Ua_h")(context)           # Ua*h(j)
    Wa_s_tile = Lambda(
        lambda x: K.tile(K.expand_dims(x, 2), [1, 1, seq_length, 1]))(Wa_s)
    Ua_h_tile = Lambda(
        lambda x: K.tile(K.expand_dims(x, 1), [1, target_length, 1, 1]))(Ua_h)
    Wa_s_add_Ua_h = Lambda(
        lambda x: tanh(x))(add([Wa_s_tile, Ua_h_tile]))           # tanh(Wa*s(i-1) +  Ua*h(j))
    eij = Dense(1, name="eij")(Wa_s_add_Ua_h)                     # v'*tanh(Wa*s(i-1) +  Ua*h(j))

    _aij = Softmax(axis=2)(eij)                                   # aij = softmax(eij)
    attention_mask = Lambda(
        lambda x: K.tile(K.expand_dims(K.expand_dims(x, axis=1)), [1, target_length, 1, 1]))(mask)
    # add mask to ensure the attention(aij) on the invalid position is 0
    _aij_with_mask = multiply([_aij, attention_mask])
    # normalize the aij to ensure the ∑aj = 1
    aij = Lambda(
        lambda x: x / K.expand_dims(K.sum(x, axis=2)), name="attention")(_aij_with_mask)

    # ci = ∑aij*hj
    encoder_context_tile = Lambda(
        lambda x: K.tile(K.expand_dims(x, 1), [1, target_length, 1, 1]))(context)
    ci = Lambda(
        lambda x: K.sum(x, axis=2), name="attention_context")(multiply([encoder_context_tile, aij]))

    # Decoder, 2 LSTM layers
    decoder_lstm1 = LSTM(
        decoder_hidden_size1,
        return_sequences=True,
        return_state=True,
        name="decoder1")
    decoder_lstm2 = LSTM(
        decoder_hidden_size2,
        return_sequences=True,
        return_state=True,
        name="decoder2")
    decoder_output1, decoder_state_h, decoder_state_c = decoder_lstm1(ci)
    decoder_output2, decoder_state_h, decoder_state_c = decoder_lstm2(decoder_output1)

    # FCN
    fcn = Dense(vocab_size, activation="softmax", name="logits")
    decoder_logits = fcn(decoder_output2)   # the real output of the network
    fcn_logits = fcn(fcn_embed)             # the output of bypass that improve the effect

    # Model
    model = Model(inputs=[encoder_input, decoder_input, fcn_input, mask], output=decoder_logits)
    return model

