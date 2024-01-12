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

if __name__ == "__main__":
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

    vectorized_songs = vectorize_string(songs_joined)
    print(vectorized_songs)

    print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
    # check that vectorized_songs is a numpy array
    assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"

    ### Batch definition to create training examples ###

    def get_batch(vectorized_songs, seq_length, batch_size):
        # the length of the vectorized songs string
        n = vectorized_songs.shape[0] - 1
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(n-seq_length, batch_size)

        chunk_num = n//(seq_length+1)


        '''TODO: construct a list of input sequences for the training batch'''
        input_batch = np.array([vectorized_songs[i:i+seq_length] for i in idx])
        '''TODO: construct a list of output sequences for the training batch'''
        output_batch = np.array([vectorized_songs[i+1:i+seq_length+1] for i in idx])

        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch


    # Perform some simple tests to make sure your batch function is working properly! 
    test_args = (vectorized_songs, 10, 2)
    if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
        not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
        not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
        print("======\n[FAIL] could not pass tests")
    else: 
        print("======\n[PASS] passed all tests!")
    
    x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

    for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
        print("Step {:3d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    
    batch_size=32
    model = md.LSTM(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=batch_size).to(device)
    summary(model, input_size=(batch_size, 100), dtypes=[torch.long])
    x, y = get_batch(vectorized_songs, seq_length=100, batch_size=batch_size)
    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    pred = model(x)
    print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
    print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
    x = x.to("cpu")
    sampled_indices = torch.distributions.Categorical(pred[0].softmax(dim=-1)).sample().to("cpu")
    print(sampled_indices)
    print("Input: \n", repr("".join(idx2char[x[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    ### Defining the loss function ###

    '''TODO: define the loss function to compute and return the loss between
        the true labels and predictions (logits). Set the argument from_logits=True.'''
    
    loss_func = nn.CrossEntropyLoss(reduction="mean")

    def compute_loss(logits, labels):
        #print(logits.mT, labels)
        return loss_func(logits.mT, labels)

    '''TODO: compute the loss using the true next characters from the example batch 
        and the predictions from the untrained model several cells above'''
    example_batch_loss = compute_loss(pred, y) # TODO

    print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
    print("scalar_loss:      ", example_batch_loss)

    ### Hyperparameter setting and optimization ###

    # Optimization parameters:
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

    ### Define optimizer and training operation ###

    '''TODO: instantiate a new model for training using the `build_model`
    function and the hyperparameters created above.'''
    model = md.LSTM(vocab_size, embedding_dim, rnn_units, batch_size).to(device)

    '''TODO: instantiate an optimizer with its learning rate.
    Checkout the tensorflow website for a list of supported optimizers.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/
    Try using the Adam optimizer to start.'''
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    def train_step(x, y): 
    # Use tf.GradientTape()
        model.train()
    
        '''TODO: feed the current input into the model and generate predictions'''
        y_hat = model(x)
    
        '''TODO: compute the loss!'''
        loss = compute_loss(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    ##################
    # Begin training!#
    ##################

    # history = []
    # plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    for iter in tqdm(range(num_training_iterations)):

        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
        x = torch.tensor(x_batch, dtype=torch.long).to(device)
        y = torch.tensor(y_batch, dtype=torch.long).to(device)
        loss = train_step(x, y)

        # Update the progress bar
        # history.append(loss.item())
        # plotter.plot(history)

        if iter % 20 == 0:
            # torch.save(model.state_dict(),checkpoint_prefix)
            print(loss)
    
    torch.save(model.state_dict(),checkpoint_prefix)
    print(loss)