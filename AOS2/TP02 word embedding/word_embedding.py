# %% [markdown]
# # CBOW model trained on "20000 lieues sous les mers"
# ## Needed libraries

# %%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import spacy
from spacy.lang.fr import French

# %%
# python -m spacy download fr_core_news_sm
spacy_fr = spacy.load("fr_core_news_sm")


# %% [markdown]
# ## Tokenizing the corpus

# %%
# Create a tokenizer for the french language
tokenizer = French().Defaults.create_tokenizer()

with open("data/20_000_lieues_sous_les_mers.txt", "r", encoding="utf-8") as f:
    document = tokenizer(f.read())

# Define a filtered set of tokens by iterating on `document`
tokens = ...


# Make a list of unique tokens and dictionary that maps tokens to
# their index in that list.
idx2tok = []
tok2idx = {}
...


# %% [markdown]
# ## The continuous bag of words model

# %%
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Define an Embedding module (`nn.Embedding`) and a linear
        # transform (`nn.Linear`) without bias.
        self.embeddings = ...
        self.U_transpose = ...


    def forward(self, context):
        # Implements the forward pass
        # `context` is of size `batch_size` * NGRAMS

        # `e_i` is of size `batch_size` * NGRAMS * `embedding_size`
        e_i = ...


        # `e_bar` is of size `batch_size` * `embedding_size`
        e_bar = ...


        # `UT_e_bar` is of size `embedding_size` * `vocab_size`
        UT_e_bar = ...


        # Use `F.log_softmax` function
        return ...



# Set the size of vocabulary and size of embedding
VOCAB_SIZE = len(idx2tok)
EMBEDDING_SIZE = 64

# Create a Continuous bag of words model
cbow = CBOW(VOCAB_SIZE, EMBEDDING_SIZE)

# %% [markdown]
# ## Preparing the data

# %%
def ngrams_iterator(token_list, ngrams):
    """Generates sucessive N-grams from a list of tokens."""

    # Creates `ngrams` lists shifted to the left
    token_list_shifts = [token_list[i:] for i in range(ngrams)]
    for ngram in zip(*token_list_shifts):
        # Get indexes of tokens
        idxs = [tok2idx[tok] for tok in ngram]

        # Get center element in `idxs`
        center = idxs.pop(ngrams // 2)

        # Yield the index of center word and indexes of context words
        # as a Numpy array (for Pytorch to automatically convert it to
        # a Tensor).
        yield center, np.array(idxs)


# Create center, context data
NGRAMS = 5
ngrams = list(ngrams_iterator(tokens, NGRAMS))

BATCH_SIZE = 256
data = torch.utils.data.DataLoader(ngrams, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## Learn CBOW model

# %%
# Use the Adam algorithm on the parameters of `cbow` with a learning
# rate of 0.01
optimizer = ...


# Use a negative log-likelyhood loss from the `nn` submodule
nll_loss = ...


# %%
EPOCHS = 10
try:
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (center, context) in enumerate(data):
            # Reset the gradients of the computational graph
            ...

            # Forward pass
            nll_w_hat = ...


            # Compute negative log-likelyhood loss averaged over the
            # mini-batch
            loss = ...


            # Backward pass to compute gradients of each parameter
            ...

            # Gradient descent step according to the chosen optimizer
            ...

            total_loss += loss.data

            if i % 20 == 0:
                loss_avg = float(total_loss / (i + 1))
                print(
                    f"Epoch ({epoch}/{EPOCHS}), batch: ({i}/{len(data)}), loss: {loss_avg}"
                )

        # Print average loss after each epoch
        loss_avg = float(total_loss / len(data))
        print("{}/{} loss {:.2f}".format(epoch, EPOCHS, loss_avg))

        # Predict if `predict_center_word` is implemented
        try:
            left_words = ["le", "capitaine"]
            right_words = ["me", "dit"]
            word = predict_center_word(cbow, *left_words, *right_words)[0]
            print(" ".join(left_words + [word] + right_words))
        except:
            pass

except KeyboardInterrupt:
    print("Stopped!")


# %%
def predict_center_word_idx(cbow, *context_words_idx, k=10):
    """Return k-best center words given indexes of context words."""

    fake_minibatch = torch.LongTensor(context_words_idx).unsqueeze(0)
    dist_center = cbow(fake_minibatch).squeeze()
    _, best_idxs = torch.topk(dist_center, k=k)
    return [idx2tok[idx] for idx in best_idxs]


def predict_center_word(cbow, *context_words, k=10):
    """Return k-best center words given context words."""

    idxs = [tok2idx[tok] for tok in context_words]
    return cbow.predict_center_word_idx(*idxs, k=k)


# %%
cbow.predict_center_word("vingt", "mille", "sous", "les")
cbow.predict_center_word("mille", "lieues", "les", "mers")
cbow.predict_center_word("le", "commandant", "fut", "le")
