# Hugging Face

## Running Hugging Face models on local machine

### Install Required Libraries

First, ensure you have Python installed on your machine (preferably version 3.8 or later). It is better to create one virtual environment.
```
conda create --name HuggingFace python=3.9
```

Then, install the Hugging Face transformers library and torch for running models:

```
pip install transformers torch
```
For GPU acceleration (if supported by your machine), install the GPU-compatible version of torch.

### Verify Installation
To verify everything works correctly, try running transformers-cli:

```
transformers-cli env
```
It will show your environment details and confirm if everything is set up correctly.

## transformers library

The Hugging Face [transformers](https://github.com/huggingface/transformers) library is a high-level Python framework built for natural language processing tasks. It provides a unified API for accessing pre-trained transformer models like BERT, GPT, and others.

The `transformers` library is designed to handle tasks like text generation, classification, and translation. Its key components include:

<b>Core Components</b>

- <b>Model Classes</b>: Each model architecture (e.g., BERT, GPT, T5) has a corresponding class, such as `GPT2LMHeadModel` for GPT-2.
- <b>Tokenizer Classes</b>: Tokenizers like `GPT2Tokenizer` handle tokenization and detokenization tasks.
- <b>Model Outputs</b>: Standardized model outputs (e.g., `BaseModelOutput`) simplify how results are accessed.

<b>Generation Utilities</b>

The `generate()` method is built as a high-level function that works across multiple models.
It includes utilities for sampling, beam search, and controlling sequence lengths.

## generate() method

The `generate()` method in Hugging Face's `transformers` library is used for text generation. It works by iteratively sampling tokens from a model's vocabulary based on the probabilities output by the model, starting from an initial input (or prompt). Here's a breakdown of what happens under the hood:

### 0. Input Preparation
  - Input tokens: Converts the input text into token IDs using the tokenizer. The generate function takes an input sequence (prompt), which is tokenized into a tensor of token IDs.
  - Prepares tensors fo the model (e.g., padding and attension masks).  

### 1. Initial Setup
  - Defines generation parameters, such as max_length, temperature, num_beams, etc.
  - Model preparation: The model processes the input tokens to generate an initial hidden state or context.

### 2. Decoding Loop

The decoding process begins, where the model generates one token at a time. The key steps are:

- Step 1: Predict Next Token

The model takes the input tokens and predicts the logits (raw scores) for the next token in the sequence.
Logits represent the likelihood of each token in the vocabulary being the next token.

- Step 2: Apply Decoding Strategies

Various strategies are applied to control how the next token is chosen:

  1. Greedy Decoding:

      - Select the token with the highest probability.
      - Example: If logits predict [0.1, 0.7, 0.2], the token corresponding to 0.7 is selected.
      - Advantage: Simple and deterministic.
      - Limitation: May lead to repetitive or suboptimal results.

  2. Beam Search:

      - Explores multiple potential sequences (beams) at each step and keeps the top k beams with the highest cumulative probabilities.
      - Produces more coherent results but is slower.

  3. Sampling:

      - Randomly selects the next token based on probabilities (adds randomness).
      - Example: If logits are [0.1, 0.7, 0.2], the token corresponding to 0.7 is most likely but not guaranteed.

  4. Top-k Sampling:

      - Limits selection to the top k tokens with the highest probabilities.
      - Adds diversity while avoiding unlikely tokens.

  5. Top-p (Nucleus) Sampling:

      - Chooses from the smallest subset of tokens whose cumulative probability exceeds a threshold p.
      - Balances diversity and coherence.

  6. Temperature Scaling:

      - Adjusts the "sharpness" of the probability distribution by scaling logits with a temperature factor.
      - Higher temperature → More randomness; Lower temperature → More deterministic.

- Step 3: Append Token

The selected token is appended to the sequence, and the new sequence is fed back into the model for the next iteration.

### 3. Stopping Criteria

The loop continues until:

  1. A predefined maximum length is reached (max_length).
  2. A special end-of-sequence token (<eos>) is generated.
  3. Custom stopping criteria (e.g., specific token conditions) are satisfied.

### 4. Post-Processing

  - The final sequence of token IDs is decoded back into text using the tokenizer.
  - Special tokens (e.g., padding or <eos>) can be removed if specified (skip_special_tokens=True).

### Example Code for generate()
Here’s a simplified view of how generate() is implemented internally:

```
import torch

class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, input_text, max_length=50, do_sample=False):
        # Tokenize input
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Prepare outputs
        outputs = []
        current_ids = input_ids

        for _ in range(max_length):
            # Forward pass
            logits = self.model(input_ids=current_ids).logits

            # Get next token (greedy by default)
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), 1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            # Append token to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop if <eos> is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode sequence
        return self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
```

### Extending model.generate()
If you want to customize generate():

- You can subclass a model and override the `generate()` method.
- You can implement custom decoding strategies by manipulating logits in the loop.

### Example Workflow

Here’s an example of how generate() works:

```
outputs = model.generate(
    inputs.input_ids,      # Input token IDs
    max_length=50,         # Maximum output length
    num_return_sequences=1, # Number of sequences to generate
    do_sample=True,         # Enable sampling
    top_k=50,               # Use top-k sampling
    temperature=0.7         # Adjust randomness
)
```

### In Summary
`model.generate` encapsulates the following processes:

1. Predicting token probabilities.
2. Applying decoding strategies (e.g., greedy, beam search, sampling).
3. Iteratively generating tokens until a stopping condition is met.
