import torch
import numpy as np
import os
import time
import functools
from tqdm import tqdm
import mitdeeplearning as mdl 
import utils
import torchmodel as md
from torchinfo import summary
from torch import nn
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"

songs = mdl.lab1.load_training_data()

# # Print one of the songs to inspect it in greater detail!
# example_song = songs[0]
# print("\nExample song: ")
# print(example_song)

# # Convert the ABC notation to audio file and listen to it
# utils.play_song(example_song)

# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs) 

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d", 
#   we can evaluate `char2idx["d"]`.  
char2idx = {u:i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

### Vectorize the songs string ###

'''TODO: Write a function to convert the all songs string to a vectorized
    (i.e., numeric) representation. Use the appropriate mapping
    above to convert from vocab characters to the corresponding indices.

NOTE: the output of the `vectorize_string` function 
should be a np.array with `N` elements, where `N` is
the number of characters in the input string
'''

def vectorize_string(string):
    vectorize_char = np.vectorize(lambda i: char2idx[string[i]])
    return vectorize_char(np.arange(len(string)))

# Optimization parameters:
num_training_iterations = 4000  # Increase this to train longer
batch_size = 64  # Experiment between 1 and 64
seq_length = 300  # Experiment between 50 and 500
learning_rate = 0.0003  # Experiment between 1e-5 and 1e-1

# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 2048  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = Path('training_checkpoints')
checkpoint_prefix = checkpoint_dir/"my_ckpt"

model = md.LSTM(vocab_size, embedding_dim, rnn_units, batch_size).to(device)
model.load_state_dict(torch.load(checkpoint_prefix))

### Prediction of a generated song ###

def generate_text(model, start_string, generation_length=1000):
  # Evaluation step (generating ABC text using the learned RNN model)

  '''TODO: convert the start string to numbers (vectorize)'''
  input_eval = vectorize_string(start_string)
  input_eval = torch.unsqueeze(torch.tensor(input_eval), 0).to(device)
  print(input_eval)
  # Empty string to store our results
  text_generated = []

  tqdm._instances.clear()
  for i in tqdm(range(generation_length)):
      '''TODO: evaluate the inputs and generate the next character predictions'''
      predictions = model(input_eval)
      print(predictions.shape)
      # Remove the batch dimension
      
      '''TODO: use a multinomial distribution to sample'''
      predicted_id = torch.distributions.Categorical(predictions.softmax(dim=-1)).sample()[-1][-1]
      print(predicted_id, predicted_id.shape)
      text_generated.append(idx2char[predicted_id.item()])
      predicted_id = torch.unsqueeze(predicted_id, 0)
      predicted_id = torch.unsqueeze(predicted_id, 0)
      # Pass the prediction along with the previous hidden state
      #   as the next inputs to the model
      input_eval = torch.cat((input_eval, predicted_id),dim=1)
      print(input_eval)
      
      '''TODO: add the predicted character to the generated text!'''
      # Hint: consider what format the prediction is in vs. the output
      
    
  return (start_string + ''.join(text_generated))


generated_text = generate_text(model, start_string="X", generation_length=600) 

### Play back generated songs ###

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs): 
  # Synthesize the waveform from a song
  waveform = utils.play_song(song)
  # If its a valid song (correct syntax), lets play it! 
#   if waveform:
#     print("Generated song", i)
#     ipythondisplay.display(waveform)