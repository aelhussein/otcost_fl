# Privacy Analysis Proof

## Setup

Let sites 1 and 2 hold datasets $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{n\times d}$ where $n$ is the number of samples and $d$ is the dataset dimension, respectively. All samples in $X$ and $Y$ belong to a specific label class $c \in C$ where $C = \{c_1,..,c_l\}$. We use $X_c$ and $Y_c$ to denote the samples in $X$ and $Y$ that belong to label class $c$. Algorithm 1 releases the following information to the server:

1. $XY^T$, the cosine similarity matrix calculated using MPC (Section 3.2)
2. $\mu_{X_{c}'}$ and $\mu_{Y_{c}'}, \forall c \in C$, the noised column means of $X_c$ and $Y_c$ such that $\mu_{X_c'} = \mu_{X_c} + e_1$ and $\mu_{Y_c'} = \mu_{Y_c} + e_2$ where $e_1, e_2$ are noise vectors generated using a $\rho$-zCDP mechanism.
3. $X_c'^TX_c'$ and $Y_c'^TY_c', \forall c \in C$, the noised covariance matrices of $X_c$ and $Y_c$ such that $X_c'^TX_c' = X_c^TX_C + E_1$ and $Y_c'^TY_c' = Y_c^TY_c + E_2$ where $E_1, E_2$ are symmetric noise matrices generated using a $\rho$-zCDP mechanism.

## Threat Model

In our threat model we assume the adversary has access to all released information ($XY^T$, $\mu_{X_{c}'}$, $\mu_{Y_{c}'}$, $X_c'^TX_c'$, $Y_c'^TY_c'$) and aims to reconstruct $X$ or $Y$.

## Assumptions

We make the following assumptions on $X$ and $Y$:

- All samples in $X$ and $Y$ are independent.
- All samples in $X$ and $Y$ belong to one class such that $X_c = X$ and $Y_c = Y$. Consequently, the summary statistics of the full datasets are released. This represents a **worst-case** scenario.
- $X$ and $Y$ are column-centered and row-normalized independently. This pre-processing is a requirement of Algorithm 1. This has the following implications:
  1. Row-normalization ensures that each entry $x_{ij}$ of $X$ satisfies $|x_{ij}| \leq 1$.
  2. As a consequence of 1, the entries of $XX^T$ are bounded by -1 and 1, with the diagonal entries being 1.
  3. As a consequence of 1, the diagonal entries of $X^TX$, which represent the variance of the features, is at most $\frac{1}{3}$. This arises under a uniform distribution with values $x_{ij}$ between -1 and 1.
  4. Column centering prior to calculating $\mu_{X'}$ and $\mu_{Y'}$ ensures that $\mu_{X'}$ and $\mu_{Y'}$ do not release information.

## Proof Outline

Assuming that an adversary has access to $X'^TX'$, $Y'^TY'$ and $XY^T$, our goal is to determine what information regarding datasets $X$ or $Y$ can be leaked. Without loss of generality, we focus on $X$. To quantify the leakage, we bound the ability of an adversary to reconstruct $X$ using $X'^TX'$ and $XY^T$. In summary, an adversary can reconstruct $X$ up to an orthonormal basis if the singular vectors of $X'^TX'$ and $XY^T$ are sufficiently close to $X^TX$ and $XX^T$, respectively. In the proof, **we bound how closely the singular vectors of $X'^TX'$ and $XY^T$ can approximate the singular vectors of $X^TX$ and $XX^T$**.

The proof relies on matrix perturbation theory framing $X'^TX'$ and $XY^T$ as perturbed versions of $X^TX$ and $XX^T$. Under this framework, the singular vectors of $X'^TX'$ and $XY^T$ matrix can only approximate $X^TX$ and $XX^T$ when the spectral gap of $X^TX$ and $XX^T$, $\delta := \sigma_1 - \sigma_2$ is larger than the spectral norm of the perturbation. We show that, under some assumptions and constraints on the data, the perturbation is larger than the upper bound of the spectral gaps in both $X'^TX'$ and $XY^T$.

## Preliminaries

We use the following preliminaries:

1. **Singular value decomposition (SVD):**
   - The SVD of $X$ is expressed as $X=U_X\Sigma_XV_X$ which are the left singular vectors, singular values, and right singular vectors, respectively.
   - The SVD decomposition of $XX^T$ yields $U_X$ and $\Sigma_X$ and the SVD decomposition of $X^TX$ yields $V_X$ and $\Sigma_X$.
   - With $U_X$, $\Sigma_X$, and $V_X$ it is possible to reconstruct $X$ up to an orthonormal basis.

2. **Matrix trace:** The trace of a matrix $A$ is the sum of the diagonal elements, $tr(A) = \sum^n_{i=1} a_{i,i}$. The trace is also sum of the eigenvalues, $\sigma$, of $A$, $tr(A) = \sum^n_{i=1} \sigma_i$.

3. **Norms:**
   - The Frobenius norm of matrix A is $||A||_F = \sqrt{\sum_{i=1}\sum_{j=1} a_{ij}^2}$
   - The spectral norm of a matrix, $||.||$, is the maximum singular value of the matrix, $\max\{\sigma_i\}^n_{i=1}$ for a matrix with singular values $\sigma_1,...,\sigma_n$
   - The spectral norm is upper bounded by the Frobenius norm, $||.|| \leq ||.||_F$

4. **Spectral norm of random matrix**: Let $E$ be a random symmetric matrix in $\mathbb{R}^{d \times d}$ where each entry $e_i \sim N(0, \sigma)$ then:
   $$2\sigma\sqrt{d} - c\ln d \geq ||E|| \leq 2\sigma\sqrt{d} + cd^{\frac{1}{4}}\ln d$$
   for some constant $c$. As $n\rightarrow \infty$ then $||E||$ is expected to be at least $2\sigma\sqrt{n}$.

5. **Spectral gap:** The spectral gap of a matrix, $\delta_i$, is defined as $\delta_i := \sigma_i - \sigma_{i+1}$ where $\sigma_i$ is the $i^{th}$ singular value.

6. **Wedin Sin theorem**: This theorem provides a bound on how closely the singular vectors of a perturbed matrix, $A'$ approximates the singular vectors of the original matrix, $A$
   $$\sin\Theta(V_{A}, V_{A'}) \leq C\frac{||E||}{\delta}$$
   where $\delta$ is the first spectral gap, $V_A, V_A'$ are the singular vectors, $||E||$ is the spectral norm of the perturbation matrix $E$ and $C$ is a constant $>0$. If $\delta \geq C||E||$ then:
   $$\sin\Theta(V_{A}, V_{A'}) \leq \epsilon.$$
   However, if $\delta < C||E||$ then it no longer holds that $V_{A'}$ closely approximates $V_{A}$.

7. **Jensen's inequality**: Jensen's inequality for a convex function $f(.)$ states
   $$f \left( \frac{1}{n} \sum^n_{i=1} x_i \right) \leq \frac{1}{n}\sum^n_{i=1} f(x_i).$$

### Lemma 1: The spectral gaps of $XX^T$ and $X^TX$ are $n$ and $\frac{d}{3}$, respectively

Let $XX^T \in \mathbb{R}^{n\times n}$ and $X^TX \in \mathbb{R}^{d\times d}$ where $n$ and $d$ are the number of samples and features, respectively. This proof relies on showing that the spectral gaps of $XX^T$ and $X^TX$, $\delta_{XX^T}$ and $\delta_{X^TX}$ are less than the spectral norms of the perturbations. To do this, we first establish upper bounds on $\delta_{XX^T}$ and $\delta_{X^TX}$, leveraging the fact that both $XX^T$ and $X^TX$ are symmetric matrices and $X$ is row-normalized. In the case of $XX^T$, the bound is

$$\delta_{XX^T} < n.$$

The upper bound arises as $\sum^n_{i=1} \sigma_i = tr(XX^T)$, and under the assumption that all of this is concentrated on $\sigma_1$ with $\sigma_2 \rightarrow 0$. Since the diagonal entries of $XX^T$ are all one (due to row-normalization), $tr(XX^T)=n$.

For $X^TX$, $\delta_{X^TX}$ is bounded as

$$\delta_{X^TX} < \frac{d}{3}.$$

This upper bound is derived from the fact that the diagonal entries of $X^TX$, which represent the variance of the features, are $\leq \frac{1}{3}$ (see Assumptions section) and all of this concentrated in $\sigma_1$. Consequently, $tr(X^TX) = \frac{d}{3}$. Note, these are upper bounds the gap is likely lower in real-world datasets with low intrinsic rank.

### Proof 1: Bounding the Information from Releasing $X'^TX'$

Next, we bound how closely the $V_{X'^TX'}$ approximates $V_{X^TX}$. Let $E$ be the perturbation matrix where each entry $e_i \sim N(0,\frac{\Delta^2}{2\rho^2})$ and $\Delta=\frac{\sqrt{2}}{n}\left(d+2\sqrt{d\log(n/\beta_s)} + 2\log(n/\beta_s)\right)$ where $\rho$ and $\beta_s$ are user defined parameters based on the privacy-budget (we set $\beta=0.01, \rho=0.1$). We can set a lower bound for $\Delta$ and $\sigma$ as $\Delta \geq \frac{\sqrt{2}d}{n}$ and $\sigma \geq \frac{d}{\rho n}$. Using the result for a spectral norm of a random matrix, $E$, we get

$$||E|| \geq \frac{2d^{\frac{3}{2}}}{\rho n}.$$

Thus, as long as the privacy parameters are appropriately set, then $||E|| > \delta_{X^TX} = \frac{d}{3}$. Specifically, the bound on the privacy parameters is

$$\frac{6d^{\frac{1}{2}}}{n} > \rho,$$

then $||E|| > \delta_{X^TX}$. Appropriately setting $\rho$ should ensure sufficient protection. Across all our datasets, $\rho=0.1$ met this criterion but this will vary depending on the specific datasets.

### Proof 2: Bounding the Information from Releasing $XY^T$

Next, we bound how closely the $V_{XY^T}$ approximates $V_{XX^T}$. To do this, we quantify how much $X$ and $Y$ must differ such that $XY^T$ does not reveal $U_{XX^T}$. This is an argument on the overlap in samples of $X$ and $Y$. For example, in an extreme case, if all samples overlap then $XX^T = XY^T$ and $U_{XX^T}$ is perfectly approximated by $U_{XY^T}$. To estimate the minimum difference, we leverage the fact that $XX^T$ and $XY^T$ are bounded -1,1. We define the perturbation matrix $P=XY^T - XX^T$. When samples in $X$ and $Y$ overlap, entries in $P$ will be $0$, otherwise entries will be non-zero. We can bound spectral norm of $P$ using the Frobenius norm of $P$ by using the fact that $||P|| \leq ||P||_F$ and $||P||_F$ is itself bounded as all entries in $XY^T$ are between -1,1

$$0 \leq ||P|| \leq ||P||_F \leq 2n.$$

The lower bound arises when $XY^T$ and $XX^T$ are identical and the upper bound occurs when every $XY^T$ and $XX^T$ maximally differ and entries $p_{ij}=-2\text{ or }2$. To obtain how much $X$ and $Y$ can differ for $||P|| > \delta_{XX^T} = n$, we derive a bound on $||P||_F$. Let $\alpha_F=\frac{1}{n(n-1)}\sum p_{ij}^2$ such that $\alpha_F$ represents the average squared contribution of $p_{ij}$ to $||P||_F$. Let $\alpha = \left( \frac{1}{n(n-1)}\sum p_{ij} \right)^2$ such that $\alpha$ represents the mean entry of $P$ squared. As entries in $P$ are between -2,2, $\alpha$ ranges from 0,4. We are actually interested in $\alpha_F$ but invoke Jensen's inequality to lower bound $\alpha_F$ by $\alpha$. In this case, a threshold is needed such that $XY^T$ does not reveal $U_{XX^T}$. This threshold is

$$\sqrt{\alpha_F}n \geq \sqrt{\alpha} n > n.$$

It is clear that $\alpha$ must be $\geq 1$ for $||P|| > \delta_{XX^T}$. As $\alpha$ ranges from 0,4 (as entries in $P$, or $\sqrt{\alpha}$, range from -2,2), a requirement of $\alpha \geq 1$ suggests $Y$ must differ by at least a 25% difference in entries between $XY^T$ and $XX^T$. Note, this bound assumes $\delta_{XY^T}=n$ which is conservative. In reality, $\delta_{XY^T}$ is smaller and $\alpha$ can be less than 1.

## Limitations

We assume worse-case spectral gaps of $XX^T$ and $X^TX$. A tighter bound on the spectral gaps would *reduce the level noise added to $X'^TX'$ and the necessary deviation between $XX^T$ and $XY^T$* to prevent reconstruction. We have not pursued these tighter bounds as they are dataset specific. Further, we do not consider repeated release of different matrices (e.g., repeated release with different differential privacy noise) as our setting differs from querying databases. $XY^T$ and $X'^TX'$ are released only once in our scenario.

## Conclusion

We have demonstrated that the release of $X'^TX'$ and $XY^T$ will not allow for the reconstruction of $X$.
