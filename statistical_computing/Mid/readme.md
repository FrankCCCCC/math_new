## Expectation Maximization

EM algorithm is useful for the model containing latent variables.

Iterate until $\theta$ converge
- E Step
  
  Evaluate $\mathbb{E}[p(z|x; \theta^{OLD})]$
- M Step
  
  $\arg \max_{\theta} \ \int_z p(z|x; \theta^{OLD}) ln \ p(x, z; \theta) dz$

## On the perspective of variational inference 

### Variational Lower Bound(VLBO/ELBO)

$$
log \ p(X) = log (\int_Z \ p(X, Z))
$$

$$
= log \int_Z \ p(X, Z) \frac{q(Z)}{q(Z)} \tag{2}
$$

$$
= log \int_Z \ q(Z) \frac{p(X, Z)}{q(Z)}
$$

$$
= log ( E_q[\frac{p(X, Z)}{q(Z)}] )
$$

$$
\geq E_q[log \ \frac{p(X, Z)}{q(Z)}] \tag{4}
$$

$$
= \int_Z q(Z)log \ \frac{p(X, Z)}{q(Z)} dZ
$$

$$
= - \int_Z q(Z)log \ \frac{q(Z)}{p(Z|X)p(X)} dZ
$$

$$
= log \ p(X) - \int_Z q(Z) \ log \ \frac{q(Z)}{p(Z|X)} dZ
$$

$$
= log \ p(X) - KL[q(Z) || p(Z|X)] \ \tag{5}
$$

Where $q(Z)$ in Eq. (2) is the **approximation of the true posterior distribution $p(Z|X)$**, since we don't know the distribution of the $p(Z|X)$ of hidden state $Z$. To derive the lower bound, we apply Jensen's inequality in Eq. (4).

Also, the Eq. (5) is the ELBO.

Then, we denote L as ELBO as following

$$
L = log \ p(X) - KL[q(Z) || p(Z|X)]
$$

### Variational Expectation Maximization
