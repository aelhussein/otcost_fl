# Appendix: Theoretical Foundations and Proofs

This appendix provides rigorous mathematical foundations for the heterogeneity metric proposed in the main paper. We establish key theoretical properties that justify our design choices and ensure the metric's validity.

**Notation.** Throughout this appendix, we denote vectors by bold lowercase letters (e.g., **z**) and matrices by bold uppercase letters (e.g., **W**). The Euclidean norm is denoted by $\|\cdot\|_2$ and the Frobenius norm by $\|\cdot\|_F$. The unit sphere in $\mathbb{R}^d$ is $S^{d-1} = \{z \in \mathbb{R}^d : \|z\|_2 = 1\}$.

## A.1 Concentration of Inner Products

We first establish that inner products between independent high-dimensional unit vectors concentrate around zero, formalizing the intuition that activations from unrelated features should be nearly orthogonal.

**Proposition 1** (Concentration of Inner Products). *Let $\mathbf{z}_i, \mathbf{z}_j \in \mathbb{R}^d$ be vectors drawn independently and uniformly from the unit sphere $S^{d-1}$ with $d \geq 2$. Then for any $t \in [0,1]$,*

$$\Pr(|\mathbf{z}_i^\top \mathbf{z}_j| > t) \leq 2\exp\left(-\frac{(d-1)t^2}{2}\right).$$

*Proof.* By rotational invariance, we may fix $\mathbf{z}_j$ and consider the function $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{z}_j$ for $\mathbf{x} \in S^{d-1}$. 

By symmetry, $\mathbb{E}[f(\mathbf{z}_i)] = \mathbb{E}[\mathbf{z}_i]^\top \mathbf{z}_j = 0$.

The function $f$ is 1-Lipschitz: for any $\mathbf{x}_1, \mathbf{x}_2 \in S^{d-1}$,
$$|f(\mathbf{x}_1) - f(\mathbf{x}_2)| = |(\mathbf{x}_1 - \mathbf{x}_2)^\top \mathbf{z}_j| \leq \|\mathbf{x}_1 - \mathbf{x}_2\|_2 \|\mathbf{z}_j\|_2 = \|\mathbf{x}_1 - \mathbf{x}_2\|_2.$$

Applying Lévy's concentration inequality for Lipschitz functions on the sphere [1, Theorem 3.3.9],
$\Pr(|f(\mathbf{z}) - \mathbb{E}[f(\mathbf{z})]| > t) \leq 2\exp\left(-\frac{(d-1)t^2}{2L^2}\right)$
for any $L$-Lipschitz function $f$. Substituting $L = 1$ and $\mathbb{E}[f(\mathbf{z}_i)] = 0$ yields the result. $\square$

## A.2 Gradient Dissimilarity Bounds

Next, we bound the dissimilarity between gradient updates from same-class samples, connecting activation geometry to federated learning dynamics.

**Proposition 2** (Gradient Dissimilarity Bound). *Consider two same-class samples: $(\mathbf{z}_c, y_{\text{target}})$ from client A and $(\mathbf{z}_k, y_{\text{target}})$ from client B, where $\mathbf{z}_c, \mathbf{z}_k \in S^{d-1}$. Let $\mathbf{W} \in \mathbb{R}^{m \times d}$ be the final layer weights, with softmax outputs $\mathbf{p}_A(\mathbf{z}_c) = \text{softmax}(\mathbf{W}\mathbf{z}_c)$ and $\mathbf{p}_B(\mathbf{z}_k) = \text{softmax}(\mathbf{W}\mathbf{z}_k)$. Let $\mathbf{e}_{y_{\text{target}}}$ be the one-hot encoding of the target class. The gradient contributions are*
$\mathbf{G}_c = (\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}})\mathbf{z}_c^\top, \quad \mathbf{G}_k = (\mathbf{p}_B(\mathbf{z}_k) - \mathbf{e}_{y_{\text{target}}})\mathbf{z}_k^\top.$

*Then*
$\|\mathbf{G}_c - \mathbf{G}_k\|_F \leq \|\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}}\|_2 \|\mathbf{z}_c - \mathbf{z}_k\|_2 + \|\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k)\|_2.$

*Proof.* We decompose the difference by adding and subtracting an intermediate term:
\begin{align}
\mathbf{G}_c - \mathbf{G}_k &= (\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}})(\mathbf{z}_c - \mathbf{z}_k)^\top + (\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k))\mathbf{z}_k^\top.
\end{align}

Applying the triangle inequality and the identity $\|\mathbf{u}\mathbf{v}^\top\|_F = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$ for outer products:
\begin{align}
\|\mathbf{G}_c - \mathbf{G}_k\|_F &\leq \|(\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}})(\mathbf{z}_c - \mathbf{z}_k)^\top\|_F + \|(\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k))\mathbf{z}_k^\top\|_F \\
&= \|\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}}\|_2 \|\mathbf{z}_c - \mathbf{z}_k\|_2 + \|\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k)\|_2 \|\mathbf{z}_k\|_2.
\end{align}

Since $\mathbf{z}_k \in S^{d-1}$, we have $\|\mathbf{z}_k\|_2 = 1$, completing the proof. $\square$

## A.3 Metric Properties and Cost Function

We verify that our cost function components define a valid optimal transport problem. For samples $i$ and $j$ of the same class $c$, the cost is:
$$C_{ij}^{(c)} = w_f \cdot d_S(\mathbf{z}_i, \mathbf{z}_j) + w_l \cdot H(S_{A,c}, S_{B,c})$$
where $w_f, w_l > 0$ are weighting parameters.

### Spherical Distance
The term $d_S(\mathbf{z}_i, \mathbf{z}_j) = \|\mathbf{z}_i - \mathbf{z}_j\|_2$ is the Euclidean distance between unit vectors, which satisfies:
$$d_S(\mathbf{z}_i, \mathbf{z}_j) = \sqrt{2(1 - \mathbf{z}_i^\top \mathbf{z}_j)}.$$
This defines a metric on $S^{d-1}$.

### Hellinger Distance
We model per-class activation distributions as Gaussians: $S_{A,c} = \mathcal{N}(\boldsymbol{\mu}_{A,c}, \boldsymbol{\Sigma}_{A,c})$ using empirical moments. The Hellinger distance between two multivariate Gaussians is:
$$H^2(\mathcal{N}_1, \mathcal{N}_2) = 1 - \frac{[\det(\boldsymbol{\Sigma}_1)\det(\boldsymbol{\Sigma}_2)]^{1/4}}{\det\left(\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right)^{1/2}} \exp\left(-\frac{1}{8}\boldsymbol{\delta}^\top\left(\frac{\boldsymbol{\Sigma}_1+\boldsymbol{\Sigma}_2}{2}\right)^{-1}\boldsymbol{\delta}\right)$$
where $\boldsymbol{\delta} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$. The Hellinger distance is a true metric on probability measures [2].

### Cost Function Validity
Since both $d_S$ and $H$ are metrics and $w_f, w_l > 0$, their weighted sum defines a symmetric, non-negative **cost function** for optimal transport. However, $C_{ij}^{(c)}$ is not itself a metric on sample pairs: identity of indiscernibles fails since $C_{ii}^{(c)} = w_l \cdot H(S_{A,c}, S_{B,c}) > 0$ whenever $H(S_{A,c}, S_{B,c}) > 0$ (the Hellinger term is constant across all pairs within class $c$). 

Nevertheless, the 1-Wasserstein distance constructed from this ground cost remains a pseudo-metric between clients, which is the quantity we use to measure heterogeneity in practice.

---

**References**
1. Vershynin, R. (2018). *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press.
2. Alvarez-Melis, D., & Fusi, N. (2020). Geometric dataset distances via optimal transport. *Advances in Neural Information Processing Systems*, 33.
