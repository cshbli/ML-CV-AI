# key-value caching

During inference, transformers generate one token at a time. When we prompt the model to start generation by passing “She,” it will produce one word, such as “poured” (for the sake of avoiding distractions, let’s keep assuming one token is one word). Then, we can pass “She poured” to the model, and it produces “coffee.” Next, we pass “She poured coffee” and obtain the end-of-sequence token from the model, indicating that it considers generation to be complete.

This means we have run the forward pass three times, each time multiplying the queries by the keys to obtain the attention scores (the same applies to the later multiplication by the values).

In the first forward pass, there was just one input token (“She”), resulting in just one key vector and one query vector. We multiplied them to obtain the q1k1 attention score.

<img src="./images/Transformers-Key-Value-Caching-Explained-2.webp"/>

Next, we passed “She poured” to the model. It now sees two input tokens, so the computation inside our attention module looks as follows:

<img src="./images/Transformers-Key-Value-Caching-Explained-3.webp"/>

We did the multiplication to compute three terms, but q1k1 was computed needlessly—we had already calculated it before! This q1k1 element is the same as in the previous forward pass because:

- q1 is calculated as the embedding of the input (“She”) times the Wq matrix,
- k1 is calculated as the embedding of the input (“She”) times the Wk matrix,
- Both the embeddings and the weight matrices are constant at inference time.

Note the grayed-out entries in the attention scores matrix: these are masked with zero to achieve causal attention. For example, the top-right element where q1k3 would have been is not shown to the model as we don’t know the third word (and k3) at the moment of generating the second word.

Finally, here is the illustration of the query-times-keys calculation in our third forward pass.

<img src="./images/Transformers-Key-Value-Caching-Explained-4.webp"/>

We make the computational effort to calculate six values, half of which we already know and don’t need to recompute!

You may already have a hunch about what key-value caching is all about. At inference, as we compute the keys (K) and values (V) matrices, we store their elements in the cache. The cache is an auxiliary memory from which high-speed retrieval is possible. As subsequent tokens are generated, we only compute the keys and values for the new tokens.

For example, this is how the third forward pass would look with caching:

<img src="./images/Transformers-Key-Value-Caching-Explained-5.webp"/> 

When processing the third token, we don’t need to recompute the previous token’s attention scores. We can retrieve the keys and values for the first two tokens from the cache, thus saving computation time.

## Key-Value Cache Step by Step

### Detailed view of a transformer decoder layer

<img src="./images/1_AxPZ5-EZ-fT0Ma5utAx2sA.webp"/>

### a two-head (self)-attention layer (below) with a input sequence of length 3

<img src="./images/1_5dBHHM-jJCJjwg9xj-tmfA.webp">

### Redundant computations in the attention layer in the generation phase

<img src="./images/1_MwGRp2Z5DG4cUpvKfH90Qg.webp">

### Generation step with KV caching enabled

<img src="./images/1_4RwWnUm8zaUJmME0RkkUBQ.webp">

## References

* [Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching)
* [LLM Inference Series: 3. KV caching explained](https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8)