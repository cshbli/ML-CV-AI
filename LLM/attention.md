# Attention (Self-Attention)

## Create three vectors K, Q, V for each embedding

The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). These vectors `K`, `Q`, `V` are created by multiplying the embedding by three matrices W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> that we trained during the training process.

<img src="./images/transformer_self_attention_vectors.png" alt="self-attention-1"/>

<img src="./images/self-attention-matrix-calculation.png"/>

## Matrix Calculation of Self-Attention

The next step is to calculate the attention weights. The attention weights are calculated by dividing the dot product of the query vector and the key vector by the square root of the dimension of the key vector.

<img src="./images/self-attention-matrix-calculation-2.png"/>

- Another illustration of the matrix calculation of self-attention:

<img src="./images/ScaledDotProductAttention.png"/>

- Another illustration of the matrix calculation of self-attention:

<img src="./images/1_ScaledDotAttention.png"/>

## Multi-Head Attention

### Calculate Q, K, V for each head

<img src="./images/transformer_attention_heads_qkv.png"/>

### Calculate attention separately for each head

<img src="./images/transformer_attention_heads_z.png"/>

### Concatenate the results of each head

<img src="./images/transformer_attention_heads_weight_matrix_o.png"/>

### Put it all together

<img src="./images/transformer_multi-headed_self-attention-recap.png"/>

## References

* [The illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)