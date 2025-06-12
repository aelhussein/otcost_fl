# Theoretical Insights

This appendix provides the full proofs for the propositions proposed in the main paper. We establish key theoretical properties that justify our design choices and ensure the metric's validity.

**Notation.** Throughout this appendix, we denote vectors by bold lowercase letters (e.g., $\mathbf{z}$) and matrices by bold uppercase letters (e.g., $\mathbf{W}$). The Euclidean norm is denoted by $\lVert\cdot\rVert_2$ and the Frobenius norm by $\lVert\cdot\rVert_F$. The unit sphere in $\mathbb{R}^d$ is $\mathbb{S}^{d-1} = \{\mathbf{z} \in \mathbb{R}^d : \lVert\mathbf{z}\rVert_2 = 1\}$.

## A.1 Concentration of Inner Products

We first establish that inner products between independent high-dimensional unit vectors concentrate around zero, formalizing the intuition that activations from unrelated features should be nearly orthogonal.

**Proposition 1** (Concentration of Inner Products). Let $\mathbf{z}_i, \mathbf{z}_j \in \mathbb{R}^d$ be vectors drawn independently and uniformly from the unit sphere $\mathbb{S}^{d-1}$ with $d \geq 2$. Then for any $t \in [0,1]$,
$$
\Pr(\lvert\mathbf{z}_i^\top \mathbf{z}_j\rvert > t) \leq 2\exp\left(-\frac{(d-1)t^2}{2}\right).
$$

**Proof.** By rotational invariance, we may fix $\mathbf{z}_j$ and consider the function $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{z}_j$ for $\mathbf{x} \in \mathbb{S}^{d-1}$.

By symmetry, $\mathbb{E}[f(\mathbf{z}_i)] = \mathbb{E}[\mathbf{z}_i]^\top \mathbf{z}_j = \mathbf{0}$.

The function $f$ is 1-Lipschitz: for any $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{S}^{d-1}$,
$$
\lvert f(\mathbf{x}_1) - f(\mathbf{x}_2) \rvert = \lvert (\mathbf{x}_1 - \mathbf{x}_2)^\top \mathbf{z}_j \rvert \leq \lVert\mathbf{x}_1 - \mathbf{x}_2\rVert_2 \lVert\mathbf{z}_j\rVert_2 = \lVert\mathbf{x}_1 - \mathbf{x}_2\rVert_2.
$$
Applying Lévy's concentration inequality for Lipschitz functions on the sphere,
$$
\Pr(\lvert f(\mathbf{z}) - \mathbb{E}[f(\mathbf{z})] \rvert > t) \leq 2\exp\left(-\frac{(d-1)t^2}{2L^2}\right)
$$
for any $L$-Lipschitz function $f$. Substituting $L = 1$ and $\mathbb{E}[f(\mathbf{z}_i)] = 0$ yields the result. $\square$

## A.2 Gradient Dissimilarity Bounds

Next, we bound the dissimilarity between gradient updates from same-class samples, connecting activation geometry to federated learning dynamics.

**Proposition 2** (Gradient Dissimilarity Bound). Consider two same-class samples: $(\mathbf{z}_c, y_{\text{target}})$ from client A and $(\mathbf{z}_k, y_{\text{target}})$ from client B, where $\mathbf{z}_c, \mathbf{z}_k \in \mathbb{S}^{d-1}$. Let $\mathbf{W} \in \mathbb{R}^{m \times d}$ be the final layer weights, with softmax outputs $\mathbf{p}_A(\mathbf{z}_c) = \text{softmax}(\mathbf{W}\mathbf{z}_c)$ and $\mathbf{p}_B(\mathbf{z}_k) = \text{softmax}(\mathbf{W}\mathbf{z}_k)$. Let $\mathbf{e}_{y_{\text{target}}}$ be the one-hot encoding of the target class. The gradient contributions are
$$
\mathbf{G}_c = (\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}})\mathbf{z}_c^\top, \quad \mathbf{G}_k = (\mathbf{p}_B(\mathbf{z}_k) - \mathbf{e}_{y_{\text{target}}})\mathbf{z}_k^\top.
$$
Then
$$
\lVert\mathbf{G}_c - \mathbf{G}_k\rVert_F \leq \lVert\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}}\rVert_2 \lVert\mathbf{z}_c - \mathbf{z}_k\rVert_2 + \lVert\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k)\rVert_2.
$$

**Proof.** We decompose the difference by adding and subtracting an intermediate term:
$$
\mathbf{G}_c - \mathbf{G}_k = (\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}})(\mathbf{z}_c - \mathbf{z}_k)^\top + (\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k))\mathbf{z}_k^\top.
$$
Applying the triangle inequality and the identity $\lVert\mathbf{u}\mathbf{v}^\top\rVert_F = \lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2$ for outer products:
$$
\begin{aligned}
\lVert\mathbf{G}_c - \mathbf{G}_k\rVert_F &\leq \lVert(\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}})(\mathbf{z}_c - \mathbf{z}_k)^\top\rVert_F + \lVert(\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k))\mathbf{z}_k^\top\rVert_F \\
&= \lVert\mathbf{p}_A(\mathbf{z}_c) - \mathbf{e}_{y_{\text{target}}}\rVert_2 \lVert\mathbf{z}_c - \mathbf{z}_k\rVert_2 + \lVert\mathbf{p}_A(\mathbf{z}_c) - \mathbf{p}_B(\mathbf{z}_k)\rVert_2 \lVert\mathbf{z}_k\rVert_2.
\end{aligned}
$$
Since $\mathbf{z}_k \in \mathbb{S}^{d-1}$, we have $\lVert\mathbf{z}_k\rVert_2 = 1$, completing the proof. $\square$

## A.3 Metric Properties and Cost Function

We verify that our cost function components define a valid optimal transport problem. For samples $i$ and $j$ of the same class $c$, the cost is:
$$
C_{ij}^{(c)} = w_f \cdot d_S(\mathbf{z}_i, \mathbf{z}_j) + w_l \cdot H(\mathbf{S}_{A,c}, \mathbf{S}_{B,c})
$$
where $w_f, w_l > 0$ are weighting parameters.

### Spherical Distance
The term $d_S(\mathbf{z}_i, \mathbf{z}_j) = \lVert\mathbf{z}_i - \mathbf{z}_j\rVert_2$ is the Euclidean distance between unit vectors, which satisfies:
$$
d_S(\mathbf{z}_i, \mathbf{z}_j) = \sqrt{2(1 - \mathbf{z}_i^\top \mathbf{z}_j)}.
$$
This defines a metric on $\mathbb{S}^{d-1}$.

### Hellinger Distance
We model per-class activation distributions as Gaussians: $\mathbf{S}_{A,c} = \mathcal{N}(\boldsymbol{\mu}_{A,c}, \boldsymbol{\Sigma}_{A,c})$ using empirical moments. The Hellinger distance between two multivariate Gaussians is:
$$
H^2(\mathcal{N}_1, \mathcal{N}_2) = 1 - \frac{[\det(\boldsymbol{\Sigma}_1)\det(\boldsymbol{\Sigma}_2)]^{1/4}}{\det\left(\frac{\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2}{2}\right)^{1/2}} \exp\left(-\frac{1}{8}\boldsymbol{\delta}^\top\left(\frac{\boldsymbol{\Sigma}_1+\boldsymbol{\Sigma}_2}{2}\right)^{-1}\boldsymbol{\delta}\right)
$$
where $\boldsymbol{\delta} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$. The Hellinger distance is a true metric on probability measures.

### Cost Function Validity
Since both $d_S$ and $H$ are metrics and $w_f, w_l > 0$, their weighted sum defines a valid ground cost for optimal transport. The resulting heterogeneity score is symmetric and satisfies the triangle inequality, providing a principled geometric measure of client dissimilarity.
