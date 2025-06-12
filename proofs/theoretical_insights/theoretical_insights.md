# Theoretical Insights Proof

This appendix provides the full proof proposed in the main paper. We establish key theoretical properties that justify our design choices and ensure the metric's validity.

**Notation.** Throughout this appendix, we denote vectors by bold lowercase letters (e.g., **z**) and matrices by bold uppercase letters (e.g., **W**). The Euclidean norm is denoted by $\|\cdot\|_2$ and the Frobenius norm by $\|\cdot\|_F$. The unit sphere in $\mathbb{R}^d$ is $S^{d-1} = \{z \in \mathbb{R}^d : \|z\|_2 = 1\}$.

## A.1 Concentration of Inner Products

We first establish that inner products between independent high-dimensional unit vectors concentrate around zero, formalizing the intuition that activations from unrelated features should be nearly orthogonal.

**Proposition 1** (Concentration of Inner Products). Let $z_i, z_j \in \mathbb{R}^d$ be vectors drawn independently and uniformly from the unit sphere $S^{d-1}$ with $d \geq 2$. Then for any $t \in [0,1]$,

$\Pr(|z_i^T z_j| > t) \leq 2\exp\left(-\frac{(d-1)t^2}{2}\right).$

**Proof.** By rotational invariance, we may fix $z_j$ and consider the function $f(x) = x^T z_j$ for $x \in S^{d-1}$. 

By symmetry, $\mathbb{E}[f(z_i)] = \mathbb{E}[z_i]^T z_j = 0$.

The function $f$ is 1-Lipschitz: for any $x_1, x_2 \in S^{d-1}$,

$|f(x_1) - f(x_2)| = |(x_1 - x_2)^T z_j| \leq \|x_1 - x_2\|_2 \|z_j\|_2 = \|x_1 - x_2\|_2.$

Applying Lévy's concentration inequality for Lipschitz functions on the sphere,

$\Pr(|f(z) - \mathbb{E}[f(z)]| > t) \leq 2\exp\left(-\frac{(d-1)t^2}{2L^2}\right)$

for any $L$-Lipschitz function $f$. Substituting $L = 1$ and $\mathbb{E}[f(z_i)] = 0$ yields the result. $\square$

## A.2 Gradient Dissimilarity Bounds

Next, we bound the dissimilarity between gradient updates from same-class samples, connecting activation geometry to federated learning dynamics.

**Proposition 2** (Gradient Dissimilarity Bound). Consider two same-class samples: $(z_c, y_{target})$ from client A and $(z_k, y_{target})$ from client B, where $z_c, z_k \in S^{d-1}$. Let $W \in \mathbb{R}^{m \times d}$ be the final layer weights, with softmax outputs $p_A(z_c) = \text{softmax}(Wz_c)$ and $p_B(z_k) = \text{softmax}(Wz_k)$. Let $e_{y_{target}}$ be the one-hot encoding of the target class. The gradient contributions are

$G_c = (p_A(z_c) - e_{y_{target}})z_c^T, \quad G_k = (p_B(z_k) - e_{y_{target}})z_k^T.$

Then

$|G_c - G_k|_F \leq \|p_A(z_c) - e_{y_{target}}|_2 |z_c - z_k|_2 + |p_A(z_c) - p_B(z_k)|_2.$

**Proof.** We decompose the difference by adding and subtracting an intermediate term:

$G_c - G_k = (p_A(z_c) - e_{y_{target}})(z_c - z_k)^T + (p_A(z_c) - p_B(z_k))z_k^T.$

Applying the triangle inequality and the identity $\|uv^T\|_F = \|u\|_2 \|v\|_2$ for outer products:

$\|G_c - G_k\|_F \leq \|(p_A(z_c) - e_{y_{target}})(z_c - z_k)^T\|_F + \|(p_A(z_c) - p_B(z_k))z_k^T\|_F$

$= \|p_A(z_c) - e_{y_{target}}\|_2 \|z_c - z_k\|_2 + \|p_A(z_c) - p_B(z_k)\|_2 \|z_k\|_2.$

Since $z_k \in S^{d-1}$, we have $\|z_k\|_2 = 1$, completing the proof. $\square$

## A.3 Metric Properties and Cost Function

We verify that our cost function components define a valid optimal transport problem. For samples $i$ and $j$ of the same class $c$, the cost is:

$C_{ij}^{(c)} = w_f \cdot d_S(z_i, z_j) + w_l \cdot H(S_{A,c}, S_{B,c})$

where $w_f, w_l > 0$ are weighting parameters.

### Spherical Distance
The term $d_S(z_i, z_j) = \|z_i - z_j\|_2$ is the Euclidean distance between unit vectors, which satisfies:

$d_S(z_i, z_j) = \sqrt{2(1 - z_i^T z_j)}.$

This defines a metric on $S^{d-1}$.

### Hellinger Distance
We model per-class activation distributions as Gaussians: $S_{A,c} = \mathcal{N}(\mu_{A,c}, \Sigma_{A,c})$ using empirical moments. The Hellinger distance between two multivariate Gaussians is:

$H^2(\mathcal{N}_1, \mathcal{N}_2) = 1 - \frac{[\det(\Sigma_1)\det(\Sigma_2)]^{1/4}}{\det\left(\frac{\Sigma_1 + \Sigma_2}{2}\right)^{1/2}} \exp\left(-\frac{1}{8}\delta^T\left(\frac{\Sigma_1+\Sigma_2}{2}\right)^{-1}\delta\right)$

where $\delta = \mu_1 - \mu_2$. The Hellinger distance is a true metric on probability measures.

### Cost Function Validity
Since both $d_S$ and $H$ are metrics and $w_f, w_l > 0$, their weighted sum defines a symmetric, non-negative **cost function** for optimal transport.
