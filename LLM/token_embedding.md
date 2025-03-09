# Token embedding and word embeddings
There are differences between token embeddings and word embeddings, though they are closely related concepts in Natural Language Processing (NLP). These differences mainly arise from how tokens and words are defined and handled in modern NLP models.

## 1. Definition
Word Embedding

- Refers to a numerical representation of a word in a continuous vector space.
- Each unique word in the vocabulary is represented as a single vector.
- Commonly used in pre-transformer models like `Word2Vec`, `GloVe`, and `FastText`.

Token Embedding

- Refers to the representation of a token in a continuous vector space.
- A token may not correspond to a whole word. Tokens are subword units, such as characters, syllables, or word pieces (e.g., "play", "##ing" in the word "playing").
- Token embeddings are used in models like `BERT`, `GPT`, and other transformer-based architectures.

## 2. Granularity
Word Embedding

- Works at the word level.
- Each word, regardless of its internal structure, is treated as a single unit.
- Example: "playing" and "play" are treated as distinct entities.

Token Embedding

- Works at the subword level.
- Breaks words into smaller components for more flexibility.
- Example: "playing" might be tokenized as "play" and "##ing". Each token has its own embedding.

## 3. Handling Rare and Out-of-Vocabulary Words
Word Embedding

- Struggles with rare or unseen words (out-of-vocabulary issues).
- Requires retraining or additional mechanisms to handle new words.

Token Embedding
- Better at handling rare and unseen words due to subword tokenization.
- Even if a word is not in the vocabulary, its tokens (subparts) likely are.

## 4. Contextuality
Word Embedding

- Word embeddings are static.
- The same word always has the same embedding regardless of context.
- Example: The word "bank" has the same vector representation whether referring to a riverbank or a financial bank.

- Token Embedding
- Token embeddings in transformer models are contextualized.
- The embedding for a token depends on the surrounding words (context).
- Example: In BERT, "bank" in "riverbank" and "financial bank" will have different embeddings.

## 5. Use Cases
Word Embedding

- Used in simpler NLP tasks or traditional models.
- Example: Sentiment analysis, where context is less critical.

Token Embedding
- Used in transformer-based models for tasks where context is crucial.
- Example: Machine translation, question answering, and text generation.

## 6. Examples
Word Embedding Example (GloVe/Word2Vec)

```
from gensim.models import Word2Vec

# Training a Word2Vec model
sentences = [["hello", "world"], ["goodbye", "world"]]
model = Word2Vec(sentences, vector_size=10, min_count=1)

# Accessing a word embedding
embedding = model.wv["world"]  # Fixed vector for "world"
```

Token Embedding Example (BERT)

```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenizing input
inputs = tokenizer("Hello world!", return_tensors="pt")

# Getting token embeddings
outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state  # Contextualized embeddings
```

## In Summary

|Aspect|Word Embedding|Token Embedding|
|---|---|---|
|Granularity	|Whole word|	Subword (e.g., word pieces)
|Handling OOV Words	| Limited	|Robust with tokenization
|Contextuality	|Static	|Contextualized
|Common Algorithms	|Word2Vec, GloVe	|BERT, GPT, Transformer models
|Applications	|Simple NLP tasks	| Context-sensitive tasks

## Tokenization and Vectoring Text for an LLM

LLMs take text as input, but this text undergoes extensive processing before the LLM actually sees it. First, the text is tokenized (shown below)—or converted into a list of discrete tokens. These tokens are just words and sub-words. The LLM has a fixed set of tokens that it understands and is trained on, referred to as the model’s “vocabulary”. Vocabulary sizes change between models, but sizes of 64K to 256K total tokens are relatively common.

<img src="./images/b8aadf17-3bf6-4b79-9688-b6bfbc5840b1_1830x888.webp">

After the text has been converted to tokens, we can vectorize each token in the input. In addition to having a vocabulary, an LLM has a token embedding layer, which stores a (learned1) vector embedding for every token in its vocabulary. We can lookup the vector for each token in this layer, forming an input matrix. If each token embedding is d-dimensional and there are C total tokens in our input, then the total size of this input matrix is C by d; see below.

<img src="./images/e2f723f2-056a-4fc0-a3f7-7aa151fe297e_1194x1026.webp">
