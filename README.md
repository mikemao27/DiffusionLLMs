## Welcome to Diffusion LLMs
This is an open-source project exploring the usage of diffusion in training-free acceleration of diffusion LLMs as denoted by **nvlabs.github.io/Fast-dLLM/#**. More information can be found in **github.com/NVlabs/Fast-dLLM** as well as at **docs.vllm.ai/en/stable/getting_started/quickstart/**. All credit goes to the above listed resources. This is an implementation of such diffusion technology for sLLMs, and for understanding the role of diffusion in machine learning.

Additional research papers and resources are provided within this GitHub repository, primarily under the section titled: "Research Papers." While this is an attempt to understand diffusion LLMs, some implementation of such LLMs will be included in the GitHub repository, and may be a re-implementation of existing work. Thus, we do not claim any exclusive rights to any of the code, papers, nor knowledge provided here.

Additional links:
**https://nvlabs.github.io/Fast-dLLM/**
**https://github.com/ZHZisZZ/dllm**

## The Problem with Autoregressive Models
Every LLM you've ever heard of: ChatGPT, LLaMA, Claude, Gemini, etc. is **autoregressive**. This means, that it generates one token at a time, from left to right. Mathematically, this means that the LLM model with parameters $\theta$ models the probability distribution $P(x, \theta) = \prod P(x_n | x < n; \theta)$. While this formula seems complicated, this is essentially the product rule in discrete mathematics, counting, and probability, which states that if, for $n_1$ ways to complete a task, $T_1$, there are $n_2$ ways to complete a task, $T_2$, there are $n_1 \cdot n_2$ ways to complete both tasks. Obviously, this continues for more tasks. This is applied to output generation as the probability of producing some text $x = \text{This is an open source repository}$ is $P(\text{this}) \cdot P(\text{is} | \text{this}) \cdot P(\text{an} | \text{this is})...$.

This type of LLM has real disadvantages as, if you make a mistake predicting the optimal current token, you can't go back to fix it. You would have to regenerate everything from the very beginning. Furthermore, token-by-token generation forces the model to wait for previous tokens to be generated before it computes the conditional probability of the next token.

Diffusion offers a solution. Instead of generating an output token-by-token, it iteratively refines and predicts the **whole** sequence from a noisy starting point. The mathematics for this is: $x_{t_{-1}} ~ p\theta(x_{t_{-1}}, t)$. This means that we have some token $x_{t_{-1}}$ that is sampled from the distribution $p\theta(x_{t_{-1}}, t)$. This is a reverse diffusion step: given the current noisy sequence at some time $t$, produce a less noisy version at time $t-1$. In diffusion, $t$ is a continuous time index running from 0 to 1 (or from 0 to T). If $t = 1$, the output sequence is totally corrupted, pure noise, and utter bogus. If $t = 0$, then the sequence is clean and we have the original text. $p\theta$ means that the distribution is parameterized by a neural network with some weights $\theta$. Essentially, this is the model itself. So, we are passing the current noisy sequence into the model at the current time to clean it up and output a less noisy version.

The intuition here is as follows: imagine that we have a painting that starts off completely blacked out, pretty useless right? At each step, we reveal a few more patches of the painting until the full painting is visible at the end. Diffusion generation works similarly. We start off with the blacked-out version (all ```[MASK]``` tokens) and we reveal a few more tokens at each step until the full response is available. The formula $x_{t_{-1}} ~ p\theta(x_{t_{-1}}, t)$ just means: "given the current partially-revealed painting at step $t$, what would the slightly-more-revealed version look like?"

## Diffusion Pipeline
As seen, we have the reverse diffusion process characterized by $x_{t_{-1}} ~ p\theta(x_{t_{-1}}, t)$. The forward diffusion process is $q(z_t | x) = Cat(z_t; {\alpha}_t \cdot x + (1 -  {\alpha}_t) \cdot m)$.
* $q$: the forward process, corruption, no learned parameters.
* $z_t$: the corrupted version of the token, x, at a timestep, t.
* x: the original clean token, represented as a one-hot vector. 
* Cat(...): short for categorical distribution; it is a probability distribution over a finite set of categories.
* ${\alpha}_t$: the noise schedule; a number between 0 and 1 that decreases as t increases; when t = 0, $\alpha$ = 1 (mostly clean), when t = 1, $\alpha$ = 0 (mostly noisy).
* m: the one-hot vector for the ```[MASK]``` token.
* $(1 - \alpha) \cdot m$: the probability mass placed on the mask token.

Basically, at any timestep $t$, each token independently has a $1 - \alpha$ chance of being replaced by ```[MASK]```. Notice that since this is forward diffusion, we strt from the clean text and move towards a mask.

There is actually another reverse diffusion process formula: $Cat(zs; [(1 - \alpha s) \cdot m + (\alpha s - \alpha t) \cdot x\theta(z_t, t)] / (1 - \alpha t))$
* $x\theta(z_t, t)$: the neural network's prediction; given the current masked sequence $z_t$ at time, t, output the probability distribution for what this token was originally.
* $(\alpha s - \alpha t)$: how much "unmasking" happens between step t and step s, this is a positive number representing "progress made."
* $(1 - \alpha s) \cdot m$: keeps some probability mass on staying masked because we aren't at t = 0 yet.
* Dividing by $(1 - \alpha t)$ is just normalizing so that the probabilities sum to 1.

The intuition here is as follows: at each step, take the masked positions and partially unmask them based on how confident the model is and where we are in the schedule. The closer we are to t = 0, the more we commit to the model's top prediction.

## MDLM Loss Function
The MLDM loss function is $\mathcal{L} = \mathbb{E}q \int_0^1 [\alpha_t' / (1 - \alpha_t)] \cdot \sum \ell log \langle x \theta \ell (z_t), x \ell \rangle dt$.
* $log \langle x \theta \ell (z_t), x \ell \rangle$: this is the cross-entropy loss.
* $\sum \ell$: add up all of the cross-entropy losses for every masked token.
* $\mathbb{E}q$: take the expectation (average) over all possible corrupted sequences $z_t$ that could be drawn from the forward process; randomly corrupt each example in training and average the loss over those random corruptions.
* $\int_0^1 ... dt$: integrate over all possible timesteps from 0 to 1.
* $\alpha_t' / (1 - \alpha_t)$: the weighting term.