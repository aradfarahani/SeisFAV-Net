[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# SeisFAV-Net: Denoising Framework Architecture

<img width="850" height="496" alt="SeisFAV-Net Architecture" src="https://github.com/user-attachments/assets/de494935-cc66-4b0a-99de-e41d8bb01d72" />

The proposed denoising framework employs an integrated **Fourier Neural Operator Attention U-Net Variational Autoencoder** architecture designed to effectively suppress noise while preserving seismic signal characteristics. The model integrates spectral learning, hierarchical convolutional feature extraction, and probabilistic latent representation within a unified end-to-end framework.

<img width="2520" height="1263" alt="image" src="https://github.com/user-attachments/assets/a72d68f4-cbae-4f25-8262-1747bb8f940d" />
(a) Clean test data (b) Noisy test data (c) MSSA-Net (d) MCA SCUNet (e) DDAE (f) BM3D (g) ADDC-Net (h) SeisFAV-Net.

## Input Specification

The input to the network is a normalized one-dimensional seismic trace $x \in \mathbb{R}^{1 \times N}$, where $N$ denotes the number of temporal samples. Prior to training, all traces are standardized using the global mean and standard deviation of the dataset to stabilize optimization and accelerate convergence.

## Architecture Components

The framework utilizes a **Fourier Neural Operator block** to learn long-range frequency dependencies, followed by an **Attention-augmented U-Net encoder-decoder structure** for multi-scale feature representation. 

Within the U-Net bottleneck, a **VAE** maps features to a stochastic latent space ($d=128$), enhancing the model's ability to generalize and regularize noise distributions. 

The decoder incorporates **self-attention modules** and **skip connections** to reconstruct high-fidelity seismic signals via residual learning, where the network specifically predicts the noise component for subtraction from the raw record.

---

## Overview

SeisFAV-Net is a deep learning framework that integrates **Fourier Neural Operators (FNO)**, **Attention-augmented U-Net**, and **Variational Autoencoders (VAE)** for effective seismic noise suppression while preserving critical signal characteristics. The architecture combines spectral learning, hierarchical feature extraction, and probabilistic latent representation in a unified end-to-end trainable model.

<img width="2498" height="1227" alt="Screenshot from 2026-05-02 15-26-26" src="https://github.com/user-attachments/assets/748e359c-8b22-45fb-b02d-da6b0bf5ebde" />
𝑓-𝑘 spectral analysis (a) Clean test data (b) Noisy test data (c) MSSA-Net (d) MCA SCUNet (e) DDAE (f) BM3D (g) ADDC-Net (h) SeisFAV-Net.

### Key Features

- **Spectral-Spatial Learning**: FNO block captures long-range frequency dependencies in seismic data
- **Multi-Scale Representation**: Attention-augmented U-Net encoder-decoder with skip connections
- **Probabilistic Regularization**: VAE bottleneck ($d=128$) for robust noise modeling and generalization
- **Residual Denoising**: Network predicts noise component for direct subtraction from raw traces
- **End-to-End Training**: Single-stage optimization without preprocessing pipelines

---

## Architecture

The proposed framework processes normalized 1D seismic traces $x \in \mathbb{R}^{1 \times N}$ through three integrated components:

### 1. Fourier Neural Operator (FNO) Block
Learns global frequency-domain representations via spectral convolutions, enabling efficient modeling of long-range temporal dependencies without the quadratic complexity of self-attention.

### 2. Attention U-Net Encoder-Decoder
- **Encoder**: Hierarchical downsampling with convolutional blocks
- **Bottleneck**: VAE module maps features to stochastic latent space $\mathcal{Z} \sim \mathcal{N}(\mu, \sigma^2)$
- **Decoder**: Upsampling with self-attention modules and skip connections for high-fidelity reconstruction

### 3. Residual Learning Strategy
The network outputs the estimated noise $\hat{n}$, and the denoised signal is obtained via:

$$\hat{s}_{\text{clean}} = x - \hat{n}$$

This formulation simplifies optimization by focusing the model on noise characteristics rather than full signal reconstruction.

---

## Evaluation Metrics

The framework reports standard seismic denoising metrics:

- **Signal-to-Noise Ratio (SNR)**: $\text{SNR} = 10 \log_{10} \frac{\|s\|^2}{\|s - \hat{s}\|^2}$
- **Peak Signal-to-Noise Ratio (PSNR)**
- **Structural Similarity Index (SSIM)**
- **Mean Absolute Error (MAE)**

<img width="2438" height="1338" alt="image" src="https://github.com/user-attachments/assets/e248afdf-f371-4623-ba1b-17a7d6606815" />
Time-domain analysis (trace No.475) (a) Clean test data (b) Noisy test data (c) MSSA-Net (d) MCA SCUNet (e) DDAE (f) BM3D (g) ADDC-Net (h) SeisFAV-Net.


## Comparison with Prior Work

| Aspect | Others | SeisFAV-Net (Ours) |
|--------|--------|---------------------|
| **Spectral Learning** | ❌ None | ✅ FNO block |
| **Latent Regularization** | ❌ Deterministic | ✅ VAE bottleneck |
| **Learning Strategy** | Direct reconstruction | Residual noise prediction |
| **Code Availability** | ❌ Not released | ✅ Open source |
| **Reproducibility** | Limited architectural details | Full implementation + configs |

<img width="2528" height="1099" alt="image" src="https://github.com/user-attachments/assets/72d2775b-bf85-40b9-84e2-035941272b2a" />
frequency-amplitude spectrum analysis (trace No.475) (a) Clean test data (b) Noisy test data (c) MSSA-Net (d) MCA SCUNet (e) DDAE (f) BM3D (g) ADDC-Net (h) SeisFAVNet.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Institute of Geophysics, University of Tehran
- Department of Earth Sciences, Kharazmi University

---

## Contact

**Mahdi Farmahini Farahani**  
📧 Email: [aradfarahani@aol.com]  
🔗 GitHub: [@aradfarahani](https://github.com/aradfarahani)

For questions or collaboration inquiries, please open an issue or contact the corresponding author.

<img width="2505" height="1267" alt="image" src="https://github.com/user-attachments/assets/b3bae2bf-2370-452f-ac72-7110d3a7123d" />
(a) Clean test data (b) Noisy test data (c) MSSA-Net (d) MCA SCUNet (e) DDAE (f) BM3D (g) ADDC-Net (h) SeisFAV-Net.


---

**Note**: This implementation is provided for research purposes. For production deployment on critical seismic processing pipelines, additional validation and domain-specific tuning are recommended.
