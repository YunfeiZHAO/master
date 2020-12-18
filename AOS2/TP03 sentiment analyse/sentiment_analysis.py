# %% [markdown]
# # Word embedding and RNN for sentiment analysis
#
# The goal of the following notebook is to predict whether a written
# critic about a movie is positive or negative. For that we will try
# three models. A simple linear model on the word embeddings, a
# recurrent neural network and a CNN.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data, datasets, vocab

# Used to cache pretrained embeddings
import appdirs

# %% [markdown]
# ## Download data
#
# First run the following block that will install spacy for tokenizing
# and download an IMDB dataset and a GloVe embedding.

# %%
#!pip install --user spacy
#!spacy download en --user

# Download IMDB data
torch_cache = appdirs.user_cache_dir("pytorch")
datasets.IMDB.download(torch_cache)

# Download GloVe word embedding
vocab.GloVe(name="6B", dim="100", cache=torch_cache)

# %% [markdown]
# ## Global variables
#
# First let's define a few variables. `vocab_size` is the size of the
# vocabulary (ie number of known words) we will use. `embedding_dim` is
# the dimension of the vector space used to embed all the words of the
# vocabulary. `seq_length` is the maximum length of a sequence,
# `batch_size` is the size of the batches used in stochastic
# optimization algorithms and `n_epochs` the number of times we are
# going thought the entire training set during the training phase.

# %%
# Define a few variables
vocab_size = ...
embedding_dim = ...
seq_length = ...
batch_size = ...
n_epochs = ...


# %% [markdown]
# ## The IMDB dataset
#
# We use SpaCy and torchtext to create training, validation and testing
# datasets.

# %%
import spacy

spacy_en = spacy.load("en")
tokenize = "spacy"

# Declare the fields
TEXT = data.Field(
    tokenize=tokenize, fix_length=seq_length, lower=True, batch_first=True
)
LABEL = data.LabelField(sequential=False, dtype=torch.float)

# IMDB dataset is already divided into train and test
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=torch_cache)

# Creating a validation dataset from the training one
train_data, valid_data = train_data.split(split_ratio=0.8)

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

TEXT.build_vocab(train_data, max_size=vocab_size)
LABEL.build_vocab(train_data)

# %%
# Define true vocabulary size because there are two more tokens
vocab_size_ = len(TEXT.vocab.stoi)

# %%
print(len(TEXT.vocab))
print(TEXT.vocab.itos[:10])
print(train_data.examples[0].text[:seq_length])


# %% [markdown]
# ## Training a linear classifier with an embedding
#
# We first test a simple linear classifier on the word embeddings.


# %%
class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Define an embedding of `vocab_size` words into a vector space
        # of dimension `embedding_dim`.

        self.embedding = ...


        # Define a linear layer from dimension `seq_length` *
        # `embedding_dim` to 1.

        self.l1 = ...


    def forward(self, x):
        # `x` is of size `batch_size` * `seq_length`

        # Compute the embedding `embedded` of the batch `x`. `embedded` is
        # of size `batch_size` * `seq_length` * `embedding_dim`

        embedded = ...


        # Flatten the embedded words and feed it to the linear layer.
        # `flatten` is of size `batch_size` * (`seq_length` * `embedding_dim`)

        flatten = ...


        # Apply the linear layer and return a squeezed version
        # `l1` is of size `batch_size`

        return ...



# %% [markdown]
# We need to implement an accuracy function to be used in the `Trainer`
# class (see below).


# %%
def accuracy(predictions, labels):
    # `predictions` and `labels` are both tensors of same length

    # Implement accuracy
    return ...



assert accuracy(torch.Tensor([1, -2, 3]), torch.Tensor([1, 0, 1])) == 1
assert accuracy(torch.Tensor([1, -2, -3]), torch.Tensor([1, 0, 1])) == 2 / 3


# %% [markdown]
# We implement now a `Trainer` class that takes care of the learning
# process

# %%
class Trainer:
    def __init__(
        self,
        n_epochs=10,
        model=None,
        optimizer=None,
        criterion=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        batch_size=None,
    ):
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iterator = data.BucketIterator(train_data, batch_size=batch_size)
        self.test_data = test_data
        self.valid_data = valid_data

    def train(self):
        # model might have changed so redefine `optimizer`
        optimizer = self.optimizer(self.model.parameters())

        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}/{self.n_epochs}")

            running_loss = 0.0
            self.model.train()  # turn on training mode
            for step, batch in enumerate(self.train_iterator):

                running_loss += loss.item()
                if step % 150 == 0:
                    acc = accuracy(predictions, batch.label)
                    print(f"Loss: {loss.item()/batch_size}, Accuracy {acc}")

            epoch_loss = running_loss / len(train_data)

            # Calculate the validation loss for this epoch
            self.model.eval()  # turn on evaluation mode
            full_batch = data.Batch(self.valid_data, self.valid_data)
            # Define the accuracy `valid_acc` and the loss `valid_loss` on `full_batch`
            predictions = ...
            valid_loss = ...
            valid_acc = ...


            print(
                f"Epoch: {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation accuracy: {valid_acc:.4f}"
            )

    def test(self):
        with torch.no_grad():
            self.model.eval()  # turn on evaluation mode
            full_batch = data.Batch(self.test_data, self.test_data)
            # Define the accuracy `test_acc` and the loss `test_loss` on `full_batch`
            predictions = ...
            test_loss = ...
            test_acc = ...


            print(f"Test Loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


embedding_net = EmbeddingNet(vocab_size_, embedding_dim, seq_length)
optimizer = optim.Adam
criterion = nn.BCEWithLogitsLoss()

tr = Trainer(
    n_epochs=n_epochs,
    model=embedding_net,
    optimizer=optimizer,
    criterion=criterion,
    train_data=train_data,
    test_data=test_data,
    valid_data=valid_data,
    batch_size=batch_size,
)

tr.train()
tr.test()


# %% [markdown]
# ## Training a linear classifier with a pretrained embedding
#
# Load a GloVe pretrained embedding instead

# %%
TEXT.build_vocab(
    train_data, max_size=vocab_size, vectors="glove.6B.100d", vectors_cache=torch_cache
)
LABEL.build_vocab(train_data)


# %%
class GloVeEmbeddingNet(nn.Module):
    def __init__(self, seq_length, freeze=True):
        super().__init__()
        self.seq_length = seq_length

        # Define `embedding_dim` from vocabulary and the pretrained `embedding`.
        self.embedding_dim = ...
        self.embedding = nn.Embedding.from_pretrained(...)


        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)

    def forward(self, x):
        # `x` is of size batch_size * seq_length

        # `embedded` is of size batch_size * seq_length * embedding_dim
        embedded = self.embedding(x)

        # `flatten` is of size batch_size * (seq_length * embedding_dim)
        flatten = embedded.view(-1, self.seq_length * self.embedding_dim)

        # L1 is of size batch_size
        return self.l1(flatten).squeeze()


# %% [markdown]
# ## Use pretrained embedding without fine-tuning

# Define model and freeze the embedding
glove_embedding_net1 = ...


tr.model = glove_embedding_net1
tr.train()
tr.test()

# %% [markdown]
# ## Fine-tuning the pretrained embedding

# %%
# Define model and don't freeze embedding weights
glove_embedding_net2 = ...


tr.model = glove_embedding_net2
tr.train()
tr.test()

# %% [markdown]
# ## Recurrent neural network with frozen pretrained embedding

# %%
class RNN(nn.Module):
    def __init__(self, hidden_size):
        super(RNN, self).__init__()

        # Define frozen pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=True)

        # Get size of input `x_t` from `embedding`
        self.embedding_size = self.embedding.embedding_dim
        self.input_size = self.embedding_size

        # Size of hidden state `h_t`
        self.hidden_size = hidden_size

        # Define a GRU, don't forget to set `batch_first` to True
        self.gru = nn.GRU(...)


        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        # `x` is of size batch_size * seq_length and `h0` is of size 1
        # * `batch_size` * `hidden_size`

        # Define first hidden state in not provided
        if h0 is None:
            # Get batch and define `h0` which is of size 1 *
            # `batch_size` * `hidden_size`
            batch_size = ...
            h0 = torch.zeros(...)


        # `embedded` is of size `batch_size` * `seq_length` *
        # `embedding_dim`
        embedded = self.embedding(x)

        # Define `output` and `hidden`

        # `output` is of size `batch_size` * `seq_length` * `hidden_size`
        return ...



rnn = RNN(hidden_size=100)

tr.model = rnn
tr.train()
tr.test()


# %% [markdown]
# ## CNN based text classification

# %%
class CNN(nn.Module):
    def __init__(self, vocab_size, freeze=False):
        super().__init__()

        self.embedding_dim = TEXT.vocab.vectors.shape[1]
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=freeze)
        self.conv_0 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(3, self.embedding_dim)
        )
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(4, self.embedding_dim)
        )
        self.conv_2 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(5, self.embedding_dim)
        )
        self.linear = nn.Linear(3 * 100, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # `x` is of size batch_size * seq_length
        embedded = self.embedding(x)

        # `embedded` is of size batch_size * seq_length * embedding_dim
        # and should be of size batch_size * (n_channels=1) *
        # seq_length * embedding_dim
        # Unsqueeze `embedded`

        # `embedded` is now of size batch_size * 1 * seq_length *
        # embedding_dim  before convolution and should be of size
        # batch_size * (out_channels = 100) * (seq_length - kernel_size[0]) * 1
        # after convolution.
        # Implement the convolution layer

        # Non-linearity step, we use Relu activation
        # Implement the relu non-linearity

        # Max-pooling layer: pooling along whole sequence
        # Implement max pooling

        # Dropout on concatenated pooled features
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # Linear layer
        return self.linear(cat).squeeze()


# %%
cnn = CNN(vocab_size_)
tr.model = cnn
tr.train()
tr.test()


# %% [markdown]
# ## Test function

# %%
def predict_sentiment(model, sentence):
    "Predict sentiment of given sentence according to model"

    tokens = TEXT.preprocess(sentence)
    padded = TEXT.pad([tokens])
    tensor = TEXT.numericalize(padded)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
