
**Algorithm: Composite Loss Computation for Seismic Trace VAE**

**REQUIRE:** 
- Clean seismic traces $\{\mathbf{x}_i\}_{i=1}^{N} \in \mathbb{R}^{T \times 1}$
- Reconstructed traces $\{\hat{\mathbf{x}}_i\}_{i=1}^{N} \in \mathbb{R}^{T \times 1}$
- Latent distribution parameters $\{\mu_j, \sigma_j^2\}_{j=1}^{d}$ from VAE module
- Batch size $N$, trace length $T$, latent dimension $d$
- Hyperparameters $\lambda_1$ ($L_1$ weight), $\beta$ (KL weight)

**ENSURE:** Total loss $\mathcal{L}_{\text{total}}$ for backpropagation

**RECONSTRUCTION LOSS (MSE):**

1. $\mathcal{L}_{\text{MSE}} \leftarrow 0$ ▷ Initialize mean squared error loss
2. **for** $i = 1$ to $N$ **do**
3. $\quad \mathcal{L}_{\text{MSE}} \leftarrow \mathcal{L}_{\text{MSE}} + \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2$ ▷ Accumulate squared $L_2$ norm
4. **end for**
5. $\mathcal{L}_{\text{MSE}} \leftarrow \mathcal{L}_{\text{MSE}} / N$ ▷ Average over batch

**SPARSITY REGULARIZATION ($L_1$):**

6. $\mathcal{L}_{L_1} \leftarrow 0$ ▷ Initialize $L_1$ regularization term
7. **for** $i = 1$ to $N$ **do**
8. $\quad \mathcal{L}_{L_1} \leftarrow \mathcal{L}_{L_1} + \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_1$ ▷ Accumulate $L_1$ norm
9. **end for**
10. $\mathcal{L}_{L_1} \leftarrow \mathcal{L}_{L_1} / N$ ▷ Average over batch

**VAE LATENT REGULARIZATION (KL DIVERGENCE):**

11. $\mathcal{L}_{\text{KL}} \leftarrow 0$ ▷ Initialize KL divergence term
12. **for** $j = 1$ to $d$ **do**
13. $\quad \mathcal{L}_{\text{KL}} \leftarrow \mathcal{L}_{\text{KL}} + (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$ ▷ Per latent dimension
14. **end for**
15. $\mathcal{L}_{\text{KL}} \leftarrow -0.5 \times \mathcal{L}_{\text{KL}}$ ▷ Scale for $KL(q\|p)$ with $p = \mathcal{N}(0,I)$

**COMPOSITE LOSS COMPUTATION:**

16. $\mathcal{L}_{\text{total}} \leftarrow \mathcal{L}_{\text{MSE}} + \lambda_1 \cdot \mathcal{L}_{L_1} + \beta \cdot \mathcal{L}_{\text{KL}}$ ▷ Weighted combination
17. **return** $\mathcal{L}_{\text{total}}$ ▷ Output total loss for gradient descent
