$$
\begin{align}
&\textbf{REQUIRE: } \text{Bottleneck feature tensor } F \in \mathbb{R}^{128 \times N/4} \text{ after self-attention,} \\
&\qquad\qquad \text{learnable weights } W_\mu, W_\sigma, W_d \in \mathbb{R}^{128 \times 128 \cdot N/4}, \\
&\qquad\qquad \text{bias vectors } \mathbf{b}_\mu, \mathbf{b}_\sigma, \mathbf{b}_d \in \mathbb{R}^{128}, \\
&\qquad\qquad \text{latent dimension } d_z = 128 \\
&\textbf{ENSURE: } \text{Reconstructed feature tensor } F' \in \mathbb{R}^{128 \times N/4} \text{ with probabilistic regularization} \\
\\
&\textbf{FEATURE FLATTENING:} \\
&1: \quad \mathbf{f} \leftarrow \text{Flatten}(F) \quad \triangleright \text{ Convert spatial features to vector } \mathbf{f} \in \mathbb{R}^{128 \cdot N/4} \\
\\
&\textbf{LATENT DISTRIBUTION PARAMETER ESTIMATION:} \\
&2: \quad \mu \leftarrow W_\mu \mathbf{f} + \mathbf{b}_\mu \quad \triangleright \text{ Compute latent mean via FC layer (Equation 9)} \\
&3: \quad \log \sigma^2 \leftarrow W_\sigma \mathbf{f} + \mathbf{b}_\sigma \quad \triangleright \text{ Compute log-variance via FC layer (Equation 9)} \\
&4: \quad \sigma \leftarrow \exp(0.5 \cdot \log \sigma^2) \quad \triangleright \text{ Convert log-variance to standard deviation} \\
\\
&\textbf{STOCHASTIC LATENT SAMPLING (REPARAMETERIZATION TRICK):} \\
&5: \quad \epsilon \sim \mathcal{N}(0, I) \quad \triangleright \text{ Sample standard Gaussian noise } \epsilon \in \mathbb{R}^{128} \\
&6: \quad \mathbf{z} \leftarrow \mu + \sigma \odot \epsilon \quad \triangleright \text{ Reparameterized sampling (Equation 10)} \\
&7: \quad \quad \triangleright \odot \text{ denotes element-wise multiplication} \\
\\
&\textbf{LATENT DECODING AND SPATIAL RECONSTRUCTION:} \\
&8: \quad \mathbf{f}' \leftarrow W_d \mathbf{z} + \mathbf{b}_d \quad \triangleright \text{ Project latent vector back to feature space (Equation 11)} \\
&9: \quad F' \leftarrow \text{Reshape}(\mathbf{f}', [128, N/4]) \quad \triangleright \text{ Restore spatial dimensions for decoder pathway} \\
\\
&\textbf{KL DIVERGENCE REGULARIZATION:} \\
&10: \quad \mathcal{L}_{\text{KL}} \leftarrow -0.5 \cdot \sum_{i}(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2) \quad \triangleright \text{ KL divergence from prior } \mathcal{N}(0, I) \\
&11: \quad \text{Add } \mathcal{L}_{\text{KL}} \text{ to total loss} \quad \triangleright \text{ Enforce structured latent space during training} \\
&12: \quad \textbf{return } F' \quad \triangleright \text{ Output regularized features to decoder}
\end{align}
$$
