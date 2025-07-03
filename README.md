# AudioNoiseRemovalSystem

A comprehensive pipeline for removing background noise from speech recordings using SEGAN (Speech Enhancement GAN). This system preserves the clarity and intelligibility of the original signal while minimizing background noise through an end-to-end generative adversarial approach.

---

## Dataset 

We used the 28spk "datashare" dataset, which contains paired noisy and clean speech recordings from 28 different speakers. Each sample is in WAV format, recorded at a 48,000 Hz sampling rate. The dataset includes both the original clean version and a version with added background noise, making it ideal for supervised speech enhancement tasks.

---

 ## Features

- **Exploratory Data Analysis (EDA):** Visualize waveforms and spectrograms to understand noise characteristics.
- **Preprocessing:** Normalize audio, trim silences, and extract features.
- **Model Training with SEGAN:** Train a generative adversarial network for end-to-end speech enhancement.
- **Evaluation:** Compute objective metrics (SNR, PESQ, STOI) to quantify performance.
- **Flask Web App:** Upload or record audio directly in the browser and download the cleaned result.
