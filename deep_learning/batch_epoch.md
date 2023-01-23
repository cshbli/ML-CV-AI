# Batch Size, Training Steps and Epochs

Neural networks are trained on large datasets with thousands or millions of samples (observations/instances). When the dataset is large, it would be time-consuming and computationally expensive to use the entire dataset for each gradient update during the training process. Sometimes, very large datasets will not fit in the computer’s memory.

As a solution for this, we use batches — portions of the dataset instead of the entire dataset to perform gradient updates during training.

<b>Batch size refers to the number of training instances in the batch.</b>

You should not get confused with batch size and the number of batches! The number of batches is calculated as follows.

```
No. of batches = (Size of the entire dataset / batch size) + 1
```

Imagine that there are 60,000 instances in the training dataset, and the batch size = 128 To calculate the number of batches, we simply divide the size of the dataset by batch size.

```
No. of batches = int(60000/128) + 1 = 469
```

The training algorithm starts drawing batches from the dataset. It takes the first 128 instances (first batch) from the dataset, trains the model, calculates the average error and updates parameters one time (perform one gradient update). This completes one training step (also called iteration).

<b>A training step (iteration) is one gradient update.</b>

Then, the algorithm takes the second 128 instances (second batch) from the dataset, trains the model, calculates the average error and updates parameters one time (perform another gradient update). This completes another training step.

The algorithm keeps doing this procedure until all batches are drawn from the training dataset. That’s 469 times according to our example! That concludes one epoch in training. In an epoch, the entire dataset is shown to the model.

<b>Epochs refer to the number of times the model sees the entire dataset.</b>

```
No. of training steps = No. of batches = No. of gradient updates
```

```
No. of ALL gradient updates = No. of batches x No. of epochs
```

## Determining the right batch size

When the batch size increases,

- The algorithm performs stable gradient updates.
- The algorithm takes more time to complete each training step (iteration).
