$$
\begin{align}
&\textbf{REQUIRE: } \text{Input } x \in \mathbb{R}^{C_{\text{in}} \times T}, \text{ spectral weights } W \in \mathbb{C}^{C_{\text{in}} \times C_{\text{out}} \times M}, \\
&\qquad\qquad\text{pointwise weights } W_{1\times1} \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}, \text{ retained modes } M = 32 \\
&\textbf{ENSURE: } \text{Output } x' \in \mathbb{R}^{C_{\text{out}} \times T} \text{ with long-range dependencies captured} \\
&1: \quad \hat{x} \leftarrow \text{FFT}(x) \quad \triangleright \text{ Forward transform: } \hat{x} \in \mathbb{C}^{C_{\text{in}} \times K}, K = \lfloor T/2 \rfloor + 1 \\
&2: \quad \textbf{for } k = 1 \text{ to } M \textbf{ do} \quad \triangleright \text{ Spectral multiplication on retained modes} \\
&3: \quad\quad \hat{y}_{j,k} \leftarrow \sum_{i=1}^{C_{\text{in}}} W_{i,j,k} \cdot \hat{x}_{i,k} \quad \triangleright \text{ Learn frequency-domain filters} \\
&4: \quad \textbf{end for} \\
&5: \quad \textbf{for } k = M + 1 \text{ to } K \textbf{ do} \quad \triangleright \text{ Truncate high frequencies} \\
&6: \quad\quad \hat{y}_k \leftarrow 0 \\
&7: \quad \textbf{end for} \\
&8: \quad y \leftarrow \text{IFFT}(\hat{y}) \quad \triangleright \text{ Inverse transform: } y \in \mathbb{R}^{C_{\text{out}} \times T} \\
&9: \quad z \leftarrow y + W_{1\times1}(x) \quad \triangleright \text{ Add skip connection via } 1\times1 \text{ convolution} \\
&10: \quad x' \leftarrow \text{ReLU}(z) \quad \triangleright \text{ Apply activation: } x' = \max(0, z) \\
&11: \quad \textbf{return } x' \quad \triangleright \text{ Complete FNO layer} \\
&\textbf{ACTIVATION FUNCTION: } \text{ReLU (Rectified Linear Unit)}
\end{align}
$$
