# Information for an Event

The intuition behind quantifying information is the idea of measuring how much surprise there is in an event. Those events that are rare (low probability) are more surprising and therefore have more information than those events that are common (high probability).

- Low Probability Event: High Information (surprising).
- High Probability Event: Low Information (unsurprising).

Rare events are more uncertain or more surprising and require more information to represent them than common events.

We can calculate the amount of information there is in an event using the probability of the event. This is called “Shannon information,” “self-information,” or simply the “information,” and can be calculated for a discrete event x as follows:

$$ h(x)=-log(p(x))$$

Where log() is the base-2 logarithm and p(x) is the probability of the event x.

Other logarithms can be used instead of the base-2. For example, it is also common to use the natural logarithm that uses base-e (Euler’s number) in calculating the information, in which case the units are referred to as “nats.”

<img src="pic/Plot-of-Probability-vs-Information.png">

# Entropy

In information theory, the entropy of a random variable is the average level of "information", "surprise", or "uncertainty" inherent to the variable's possible outcomes. 

Entropy is the number of bits required to transmit a randomly selected event from a probability distribution. A skewed distribution has a low entropy, whereas a distribution where events have equal probability has a larger entropy.

A skewed probability distribution has less “surprise” and in turn a low entropy because likely events dominate. Balanced distribution are more surprising and turn have higher entropy because events are equally likely.

- Skewed Probability Distribution (unsurprising): Low entropy.
- Balanced Probability Distribution (surprising): High entropy.

$$ H(X) = - \sum_{x \in X}p(x)log_bp(x)$$

where b is the base of the logarithm used. Common values of b are 2, Euler's number e, and 10, and the corresponding units of entropy are the bits for b = 2, nats for b = e, and bans for b = 10.

<img src="pic/Plot-of-Probability-Distribution-vs-Entropy.png">

# Cross Entropy

Cross-entropy builds upon the idea of entropy from information theory and calculates the number of bits required to represent or transmit an average event from one distribution compared to another distribution.

The intuition for this definition comes if we consider a target or underlying probability distribution P and an approximation of the target distribution Q, then the cross-entropy of Q from P is the number of additional bits to represent an event using Q instead of P.

The cross-entropy between two probability distributions, such as Q from P, can be stated formally as:

$$H(P, Q) = – \sum_{x \in X} P(x) * log(Q(x))$$

For classification problems: 

<p align="center">
<img src="pic/gNip2.png">
</p>

Where H() is the cross-entropy function, P may be the target distribution and Q is the approximation of the target distribution. Log is the base-2 logarithm, meaning that the results are in bits. If the base-e or natural logarithm is used instead, the result will have the units called nats.

The result will be a positive number measured in bits and will be equal to the entropy of the distribution if the two probability distributions are identical.

# Binary Cross Entropy and Categorical Cross Entropy

Cross-entropy is widely used as a loss function when optimizing classification models.

Two examples that you may encounter include the logistic regression algorithm (a linear classification algorithm), and artificial neural networks that can be used for classification tasks.

$$ -y*log(p)-(1-y)*log(1-p)$$

The use of cross-entropy for classification often gives different specific names based on the number of classes, mirroring the name of the classification task; for example:

- Binary Cross-Entropy: Cross-entropy as a loss function for a binary classification task.
- Categorical Cross-Entropy: Cross-entropy as a loss function for a multi-class classification task.

# KL Divergence

The KL divergence is often referred to as the “relative entropy.”

- Cross-Entropy: Average number of total bits to represent an event from Q instead of P.
- Relative Entropy (KL Divergence): Average number of extra bits to represent an event from Q instead of P.

$$KL(P || Q) = – \sum_{x \in X} P(x) * log(\dfrac{P(x)} {Q(x)})$$

The value within the sum is the divergence for a given event.

As such, we can calculate the cross-entropy by adding the entropy of the distribution plus the additional entropy calculated by the KL divergence. This is intuitive, given the definition of both calculations; for example:

$$H(P, Q) = H(P) + KL(P || Q)$$

Where H(P, Q) is the cross-entropy of Q from P, H(P) is the entropy of P and KL(P || Q) is the divergence of Q from P.

Like KL divergence, cross-entropy is not symmetrical, meaning that:

$$H(P, Q) \neq H(Q, P)$$
