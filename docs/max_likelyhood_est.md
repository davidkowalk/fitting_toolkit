# Maximum Likelyhood Estimation

This document describes the maximum likelyhood curve fit implemented by the fitting toolkit.

Maximum likelyhood estimation is separated into the following steps
1. Compute probability density that the function using the parameters $\theta$ produce a datapoint
2. Compute log probability density that set of data points are produced by fitted function using parameters $\theta$
3. Find point $\theta$ in parameter space that maximizes log likelyhood from step 2
4. Find covariance by computing the inverse hessian matrix at optimal point.

## 1. Computing PD of Data Point 

To compute the prabability density that a function produces on specific datapoint. The two main approaches to do this differentiate between data with only y-uncertainty and points that have both x- and y- uncertainty. Consider the normal distribution:

$$
p(x)_{\mu, \sigma} = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2\right)
$$

From that assuming the mean is defined by the fitted function $f$ and using the y-uncertainty, compare the probabillity density for a point at y given the paramteters $\theta$ [citation needed]:

$$
p(y_i | \theta)_{f, \sigma_y^i} = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\left(-\frac{1}{2} \left(\frac{y_i-f(x_i, \theta)}{\sigma_{y,i}}\right)^2\right)
$$

When also considering the x-uncertainty the calculations become more complicated. [citation needed]

$$
p(x_i, y_i |\theta)_{f, \sigma_{x,i} \sigma{y_i}} = \int_{-\infty}^{\infty} du_i 
\frac{1}{2\pi\sigma_{y,i}\sigma_{x,i}}

\exp\left(-\frac{(y_i-f(u_i, \theta))^2}{2\sigma_{y,i}^2} -\frac{(x_i - u_i)^2}{2\sigma_{x,i}^2}\right)
$$

Where $u_i$ represents the "true" value of x which is measured with a defined uncertainty. Internally this integration is performed by `scipy.optimize.quad`

It is often advantegious to take the natural logarithm of these functions. When only considering x-errors this simplifies to:

$$
-\ln p(y_i | \theta)_{f, \sigma_y^i} =
\frac{1}{2}\ln\left({{2\pi\sigma_{y,i}^2}}\right) +
\frac{1}{2} \left(\frac{y_i-f(x_i, \theta)}{\sigma_{y,i}}\right)^2
$$

This simplification does not work when considering uncertainties in x as the lofarithm does not commutate with the integral as it is not a linear operator.

## 2. Computing Probability Density of Measurement set

The probabillity density that a set of points is produced by $f(x, \theta)$ is the product of the probability density of each point:

$$
p(x, y|\theta) = \prod_i p(x_i, y_i | \theta)
$$

It follows, that the logarithmic density is
$$
\ln p(x, y|\theta) = \sum_i \ln p(x_i, y_i | \theta)
$$

This is often advantageous as the the product of small numbers often quickly exceed a computers abillities to accurately compute.

## 3. Finding the Optimal Set of Parameters

There are different algorithms available for maximizing $p(x, y|\theta)$. Note that since the natural logarithm is bijective on the natural numbers bigger than 0 this is equivalent to minimizing

$$
-\ln p(x, y|\theta) = \sum_i - \ln p(x_i, y_i | \theta)
$$

Internally this is handled by `scipy.minimize` which is documented [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

## 4. Finding the Covariance Matrix

The covariance matrix of the fitted parameters is equivalent to the inverse of the hessian matrix [1, p. 6181] if:

1. The parameters are normally distributed around the fit. If this is violated the hessian matrix provides a good local approximation, but does not strictly converge onto the true values.
2. The cost function ($-\ln p(x, y|\theta)$) is quadratic in $\theta$. However Thacker notes

    >As long as the model is not too nonlinear, the inverse of the Hessian should provide a good approximation to the error-covariance matrix even in the nonlinear case. [1, p. 6177/6178]

3. The Hessian matrix must be invertible. This requires:
    1. The cost function ($-\ln p(x, y|\theta)$) must have a unique local minimum in $\theta$.
    2. The data must contain sufficient information to estimate all the parameters (no singularities in the Hessian).

4. The data must be sufficiently informative to constrain the model parameters. If the hessian is near a singularity the estimate becomes unreliable. [1, p. 6182]

5. The statistical model used must correctly describe the data-generating process. Misspecification of the model can lead to biased estimates, and the Hessian may not accurately represent the true parameter uncertainty.

The result object produced by `scipy.minimize` already contains the inverse hessian matrix and thus this can simply be returned.

# Literature

[1] “The Role of the Hessian Matrix in Fitting Models to Measurements.” Journal of geophysical research JGR ; publ. by American Geophysical Union 94.C5 (1989): 6177–6196. Web. [Paywalled Version]( https://doi.org/10.1029/JC094iC05p06177) (doi: /10.1029/JC094iC05p06177)
