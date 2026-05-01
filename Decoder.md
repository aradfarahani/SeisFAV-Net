$$
\begin{align}
&\textbf{REQUIRE: } \text{Regularized bottleneck features } F' \in \mathbb{R}^{128 \times N/4}, \\
&\qquad\qquad \text{encoder skip connections } \{E_1 \in \mathbb{R}^{32 \times N}, E_2 \in \mathbb{R}^{64 \times N/2}, E_3 \in \mathbb{R}^{128 \times N/4}\}, \\
&\qquad\qquad \text{noisy input } x_{\text{noisy}} \in \mathbb{R}^{1 \times N}, \\
&\qquad\qquad \text{network parameters } \theta \\
&\textbf{ENSURE: } \text{Denoised seismic signal } \hat{s} \in \mathbb{R}^{1 \times N} \\
\\
&\textbf{PROGRESSIVE UPSAMPLING WITH MULTI-SCALE FEATURE FUSION:} \\
&1: \quad D_1 \leftarrow \text{TransposedConv}(F', \text{kernel}=4, \text{stride}=2) \quad \triangleright \text{ Upsample } N/4 \to N/2 \text{ (first stage, Equation 13)} \\
&2: \quad D_1 \leftarrow \text{Concatenate}(D_1, E_3) \quad \triangleright \text{ Fuse with encoder skip connection from EncBlock 3} \\
&3: \quad D_1 \leftarrow \text{ConvBlock}(D_1) \quad \triangleright \text{ Process concatenated features via convolution block} \\
&4: \quad D_1 \leftarrow \text{SelfAttention}(D_1) \quad \triangleright \text{ Apply self-attention in dec1 for contextual refinement} \\
&5: \quad D_2 \leftarrow \text{TransposedConv}(D_1, \text{kernel}=4, \text{stride}=2) \quad \triangleright \text{ Upsample } N/2 \to N \text{ (second stage, Equation 13)} \\
&6: \quad D_2 \leftarrow \text{Concatenate}(D_2, E_2) \quad \triangleright \text{ Fuse with encoder skip connection from EncBlock 2} \\
&7: \quad D_2 \leftarrow \text{ConvBlock}(D_2) \quad \triangleright \text{ Refine features through convolutional processing} \\
&8: \quad D_3 \leftarrow \text{TransposedConv}(D_2, \text{kernel}=4, \text{stride}=2) \quad \triangleright \text{ Upsample to original resolution } N \text{ (third stage)} \\
&9: \quad D_3 \leftarrow \text{Concatenate}(D_3, E_1) \quad \triangleright \text{ Fuse with encoder skip connection from EncBlock 1} \\
&10: \quad D_3 \leftarrow \text{ConvBlock}(D_3) \quad \triangleright \text{ Final feature refinement at full resolution} \\
\\
&\textbf{RESIDUAL NOISE PREDICTION:} \\
&11: \quad \hat{n} \leftarrow \text{Conv}_{1\times1}(D_3, \text{out channels}=1) \quad \triangleright \text{ Map refined features to noise estimate (Equation 14)} \\
\\
&\textbf{DENOISED SIGNAL RECONSTRUCTION VIA RESIDUAL SUBTRACTION:} \\
&12: \quad \hat{s} \leftarrow x_{\text{noisy}} - \hat{n} \quad \triangleright \text{ Obtain clean signal by subtracting predicted noise (Equation 15)} \\
&13: \quad \textbf{return } \hat{s} \quad \triangleright \text{ Output denoised seismic trace}
\end{align}
$$
