import torch
from torch import nn

'''TODO: Add LSTM and Dense layers to define the RNN model using the Sequential API.'''
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = nn.Sequential(
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #   of a fixed embedding size
    nn.Embedding(vocab_size, embedding_dim),
    #tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size,  ]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    # TODO: Call the LSTM function defined above to add this layer.
    nn.LSTM(embedding_dim,rnn_units, batch_first=True),
    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size. 
    # TODO: Add the Dense layer.
    nn.Linear(rnn_units, vocab_size),
  )

  return model

class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
    super().__init__()
    self.lstm = nn.Sequential(
      nn.Embedding(vocab_size, embedding_dim),
      nn.LSTM(embedding_dim,rnn_units, batch_first=True),
    )
    self.linear = nn.Sequential(nn.Linear(rnn_units, vocab_size)) 
  
  def forward(self, x):
    y, (h,c) = self.lstm(x)
    # print(y)
    # print(y.shape)
    return self.linear(y)

if __name__ == "__main__":
    # Build a simple model with default hyperparameters. You will get the 
    #   chance to change these later.
    model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)