# coding: utf-8
import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from data import get_data, words2id, id2words, seq_length, vocab_size
from network import network

learning_rate = 1e-3        # learning rate
epochs = 100                # total epochs
batch_size = 512            # batch size
target_length = 2           # the length of output sequence
embedding_size = 64         # the size of embedding layer's units
attention_units = 64        # the size of attention layer's units
encoder_hidden_size = 64    # the size of encoder LSTM's units
decoder_hidden_size1 = 128  # the size of 1st decoder LSTM's units
decoder_hidden_size2 = 64   # the size of 2nd decoder LSTM's units

input_data, target_data = get_data()                        # get the data that has been preprocessed
X_data, mask_data = words2id(input_data, seq_length)        # X_data is the input of encoder, mask_data is the mask
y_data, _ = words2id(target_data, 2)                        # y_data is the target of network
y_target = to_categorical(y_data, num_classes=vocab_size)   # transform target into one-hot
y_input = ["<hour> <min>"] * len(y_data)                    # y_input is the input of decoder
y_input, _ = words2id(y_input, 2)

# network
model = network(seq_length, vocab_size, target_length, embedding_size, attention_units,
                encoder_hidden_size, decoder_hidden_size1, decoder_hidden_size2)


# Loss function
def loss(y_true, y_pred):
    loss1 = categorical_crossentropy(y_true, model.get_layer("logits").get_output_at(0))
    loss2 = categorical_crossentropy(y_true, model.get_layer("logits").get_output_at(1))
    return loss1 + loss2


# Training
adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss=loss)
history = model.fit([X_data, y_input, y_data, mask_data], y_target,
                    batch_size=batch_size, epochs=epochs, validation_split=0.01)
model.save_weights("model/model.h5")
