## Expectation Maximization

EM algorithm is useful for the model containing latent variables $Z$ when the maximum likelihood is hard to derive from the observed data $Y$. We can write the maximum likelihood of $Y$ like following

$$
\mathcal{L}(Y; \theta) = \arg \max_{\theta} log(p(Y; \theta))
$$

The Expectation Maximization rewrites the question as the following

$$
\arg \max_{\theta} \ log \int_{Z} p(Y, Z; \theta) dZ
$$

Thus, we can derive the EM with an approximation $q(Z; \gamma)$ for $p(Z|Y)$ to avoid evaluating such complex distribution directly

$$
= \arg \max_{\theta} \ log \int_{Z} \frac{q(Z; \gamma)}{q(Z; \gamma)} p(Y, Z; \theta) dZ
$$

$$
= \arg \max_{\theta} \ log \ \mathbb{E}_{q} [\frac{p(Y, Z; \theta)}{q(Z; \gamma)}]
$$

Since the $log$ function is concave, $log(\mathbb{E}_{p}[X]) \geq \mathbb{E}_{p}[log(X)]$

$$
\geq \arg \max_{\theta} \ \mathbb{E}_{q} [log(\frac{p(Y, Z; \theta)}{q(Z; \gamma)})]
$$

$$
= \arg \max_{\theta} \ \int_Z q(Z; \gamma) log \ p(Y, Z; \theta) dZ - \int_Z q(Z; \gamma) log \ q(Z; \gamma) dZ
$$

$$
= \arg \max_{\theta} \ \int_Z q(Z; \gamma) log \ p(Y, Z; \theta) dZ - H_q[Z]
$$

Where $H_q[Z]$ is the entropy of $Z$ over distribution $q$

So far, we can express the EM algorithm in a simpler way as

---
Iterate until $\theta$ converge
- E Step
  
  Evaluate $q(Z; \gamma) = p(Z|Y)$
- M Step
  
  $\arg \max_{\theta} \ \int_Z q(Z; \gamma) log \ p(Y, Z; \theta) dZ$
---

## EM In General Form

Actually, we can represent the EM algorithm with variational lower bound $\mathcal{L}(\theta, \gamma)$

$$
\mathcal{L}(\theta, \gamma) = \arg \max_{\theta} \ \mathbb{E}_{q} [log(\frac{p(Y, Z; \theta)}{q(Z; \gamma)})]
$$

$$
= \int_Z q(Z; \gamma)log \ \frac{p(Y, Z; \theta)}{q(Z; \gamma)} dZ
$$

$$
= - \int_Z q(Z; \gamma)log \ \frac{q(Z; \gamma)}{p(Z|Y)p(Y)} dZ
$$

$$
= log \ p(Y) - \int_Z q(Z; \gamma) \ log \ \frac{q(Z; \gamma)}{p(Z|Y)} dZ
$$

$$
= log \ p(Y) - KL[q(Z; \gamma) || p(Z|Y)] \ \tag{5}
$$

Since we've known the KL-divergnce is always greater or equal to 0, when $KL[q(Z; \gamma) || p(Z|Y)] = 0$, the result of EM algorithm will be equal to the maximum likelihood $\mathcal{L}(\theta, \gamma) = \mathcal{L}(Y; \theta)$. In the mean time, minimizing the KL-divergence is actually find the best approximation $q(Z; \gamma)$ for $p(Z|Y)$. 

Thus, we can also represent the EM algorithm as

---
Iterate until $\theta$ converge
- E Step at k-th iteration
  
  $\gamma_{k+1} = \arg \max_{\gamma} \mathcal{L}(\theta_{k}, \gamma_{k})$
- M Step at k-th iteration
  
  $\theta_{k+1} = \arg \max_{\theta} \mathcal{L}(\theta_{k}, \gamma_{k+1})$
---

## On the perspective of variational inference 



### Variational Expectation Maximization

