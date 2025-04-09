# Unfolding

For complex processes the physical properties of any experiment introduce inherent systematic errors to the measurement of the underlying probability distribution.
The distortion may range from a blurring effect to the significant shift of resulting measurements.
The process of disconvoluting the detector effects from a probability distribution is calles "unfolding". Unfolding as a problem does not have a single well defined solution, but has rather spawned a whole debate about the different approaches to effeciently removing systematic errors from measurements.

## The simplest possible case.

Take a binned measurement of a variable x. The measurement $m_i$ may be defined as
$$
    m_i = \underbrace{\sum_j Q_{ij}x_j}_{\text{Convolution}} + \overbrace{c_i}^{\text{Background}}
$$

Where the first part is the convolution and the latter is the background.
The background $c_i$ is subtracted from the measurement before unfolding and can be disregarded.
To correct the binned measurement for detector biases the matrix $Q_{ij}$ must be measured. By expressing
$$
\vec m = Q\cdot\vec x \implies \vec x = Q^{-1}\vec m
$$
if Q is invertible.
The Matrix $Q$ is calculated by using the dual vector $\vec k^*$ of a known signal. Let $\vec k$ be an element of an orthonormal base. Then
$$
\begin{aligned}
    &&\vec k'^* Q \vec k &= Q_{k'k} \\
    \Leftrightarrow && \bra{k'}Q\ket{k} &= Q_{k'k}\\
    \Leftrightarrow && \bra{m}Q\ket{k} &= \braket{m|m}
\end{aligned}
$$

Note that $m_i$ does not only depend on $x_i$ but on all signal bins. To concretely calculate the matrix elements of Q the detector is exposed to an input signal of a known basis vector $\vec k$ and the measured vector $\vec m$ then corresponds to the $k$-th column of Q. This is usually done via Monte Carlo simulations.

## Continuous Generalization
While determining the matrix elements from the measurement of a known signal for a discrete number of bins is numerically feasible it becomes an ill defined problem for continuous probability density functions.

The generalization
$$
m(x) = \int Q(\lambda)p(x-\lambda) d\lambda
$$
describes the measured probability density function for a signal with the pdf $p(x)$. Note that $Q$ is assumed to be independent of $x$. $Q(\lambda)$ is calculated via the Fourier-transform

$$\begin{aligned}

\hat m(k) &=\int\text{d}x m(x) e^{-2\pi i k x}\\
&=\int \text{d}x \int \text{d}\lambda Q(\lambda) p(x-\lambda) e^{-2\pi i k x}\\
&=\int \text{d}\lambda \int \text{d}x Q(\lambda) p(x-\lambda) e^{-2\pi i k x}\\
&=\int \text{d}\lambda \int \text{d}x Q(\lambda) \underbrace{e^{-2\pi i k \lambda} e^{+2\pi i k \lambda}}_{=1}  p(x-\lambda) e^{-2\pi i k x}\\
&=\int \text{d}\lambda \int \text{d}x \underbrace{Q(\lambda) e^{-2\pi i k \lambda}}_\text{Independent of x\ } p(x-\lambda) e^{-2\pi i k (x - \lambda)}\\
&=\int \text{d}\lambda Q(\lambda) e^{-2\pi i k \lambda} \int \text{d}x p(x-\lambda) e^{-2\pi i k (x - \lambda)}\\
\end{aligned}
$$
Let $\varphi = x-\lambda \implies \text d \varphi = \text d x$.
Thus:
$$
\begin{aligned}
&& \hat m(k) &=\int \text{d}\lambda Q(\lambda) e^{-2\pi i k \lambda} \int \text{d}\varphi p(\varphi) e^{-2\pi i k \varphi}\\
&& &= \hat Q(k)\hat p(k)\\
\implies && \hat Q(k) &=\frac{\hat m(k)}{\hat p(k)}\\
\Leftrightarrow && \hat p(k) &= \frac{\hat m(k)}{\hat Q(k)}\\
\end{aligned}\\
$$

The functions $p(x)$ and $Q(\lambda)$ are obtained via inverse Fourier transformation.

## Disadvantages of this Approach

Unfolding is inherently an ill-defined inversion problem: small fluctuations in the measured data $\vec{m}$ or $m(x)$ can lead to large variations in the unfolded result $\vec{x}$ or $p(x)$. While this can partially be mitigated by first fitting a function to the calibration measurements, for one divisions of small values of $\hat Q(k)$ in Fourier space can amplify noise significantly.
Especially in high-frequency components (large $k$), $\hat{p}(k)$ becomes unreliable due to division by potentially small $\hat{Q}(k)$.

The method assumes a linear, shift-invariant system where detector effects can be modeled as a convolution.
However the first order approximation may be insufficient in real detectors. Response might be non-linear, position-dependent, or involve more complex smearing behaviors.

## Aknowledgements
The author thanks Dr. S. Neubert and Dr. J. Kroha for productive discussions and major guidance.