# 🧠 Brain Tumor Images Classification with Bayesian CNN

This repository contains the official implementation for the study **"Brain Tumor Classification with Bayesian Convolutional Neural Network: A Comparison of Variational Inference and Monte Carlo Dropout"**.

Standard Deep Learning (DL) models often produce overconfident predictions without indicating their reliability, which poses a significant risk in medical diagnosis. This project integrates **Bayesian Approximation Methods**—specifically **Variational Inference (VI)** and **Monte Carlo (MC) Dropout**—into a custom VGG-style Convolutional Neural Network (CNN) to classify brain MRI scans into four distinct categories while providing reliable **Uncertainty Quantification (UQ)**.

---

## 📌 Background

Deep learning models are notoriously known as "black boxes" that can yield overconfident wrong predictions. In clinical settings, misclassifying a tumor can lead to incorrect treatment plans. 

By applying Bayesian principles, model parameters (weights) are treated as probability distributions rather than fixed point values. This allows the network to:
- Output a probability distribution over predictions rather than just a single class label.
- Quantify **predictive uncertainty** (via predictive entropy and standard deviation).
- Automatically flag ambiguous or difficult cases for secondary review by radiologists.

---

## 📁 Dataset

The dataset used in this project is a curated combination of publicly available datasets (such as Figshare, SARTAJ, and Br35H), consisting of **7,200 labeled brain MRI scans** distributed evenly across 4 classes:

1. 🟡 **Glioma** (1,800 images)
2. 🔵 **Meningioma** (1,800 images)
3. 🟢 **No Tumor** (1,800 images)
4. 🔴 **Pituitary** (1,800 images)

### ✂️ Data Splitting & Preprocessing
* **Split Ratio (~7:1:1):** 
  * **Train Set:** 5,600 images (1,400 per class)
  * **Validation Set:** 800 images (200 per class)
  * **Test Set:** 800 images (200 per class)
* **Preprocessing Pipeline:**
  * Resized to a uniform dimension of **$224 \times 224$ pixels**.
  * Pixel intensity normalized to the **$[0, 1]$** range.
  * **Data Augmentation** (applied exclusively to the training set): Random rotation ($\pm 10^\circ$), random zoom ($\pm 10\%$), horizontal flip ($50\%$ probability), and brightness adjustment ($\pm 10\%$).

---

## 🛠️ Methodology & Architectures

We evaluated and compared three model variations built on top of a custom 4-block VGG-style backbone:

1. 🔹 **Baseline CNN (Deterministic):** Standard VGG-style architecture with 4 convolutional blocks ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ filters), Max-Pooling, Batch Normalization, Dropout, Global Average Pooling, and Dense layers trained via standard backpropagation.
2. 🔹 **BCNN with MC Dropout:** Retains the same architecture as the baseline CNN, but keeps Dropout active during both training and inference (`training=True`) to sample predictions over $T=30$ stochastic forward passes.
3. 🔹 **BCNN with Variational Inference (VI):** Replaces standard layers with probabilistic `Conv2DFlipout` and `DenseFlipout` layers from **TensorFlow Probability**. Each weight is modeled as a normal distribution $q(w) \sim \mathcal{N}(\mu, \sigma^2)$ optimized by minimizing the negative **Evidence Lower Bound (ELBO)** (combining categorical cross-entropy and KL Divergence).

### 📐 Uncertainty Metrics
For Bayesian models, uncertainty is evaluated across $T = 30$ stochastic forward passes using:
* **Predictive Entropy ($H_p$):** Measures the randomness/dispersion of class predictions across passes.
* **Predictive Standard Deviation ($\sigma$):** Measures the variance of the predicted probabilities across stochastic passes.

---

## 📊 Results

### 1. Classification Performance on Test Set

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 92.25% | 0.9248 | 0.9225 | 0.9211 |
| **BCNN-MC Dropout** | 91.75% | 0.9177 | 0.9175 | 0.9164 |
| **BCNN-VI** 🚀 | **94.63%** | **0.9525** | **0.9425** | **0.9450** |

* **BCNN-VI** achieved the highest test accuracy and overall metrics while demonstrating the most stable training and validation curves.
* **Pituitary** and **No Tumor** classes were consistently the easiest to separate ($AUC > 0.99$), whereas **Glioma** showed higher visual overlap with **Meningioma**.

### 2. Uncertainty Quantification Insights

| Model | Mean Entropy (Correct) | Mean Entropy (Incorrect) | Mean Std Dev (Correct) | Mean Std Dev (Incorrect) |
| :--- | :---: | :---: | :---: | :---: |
| **BCNN-MC Dropout** | 0.1249 | 0.5849 | 0.0163 | 0.0785 |
| **BCNN-VI** 🌟 | **0.0718** | **0.4160** | **0.0108** | **0.0648** |

* **Reliable Safety Indicator:** For both Bayesian models, **incorrectly classified samples exhibited substantially higher predictive entropy (~4.7x to 5.8x higher) and standard deviation** than correctly classified samples.
* **Sharper Estimates:** **BCNN-VI** produced lower, more concentrated uncertainty for correct predictions compared to MC Dropout, yielding a clearer separation between confident and ambiguous cases.

---

## 💡 Conclusions & Future Work

### 🎯 Key Conclusions
1. **Performance & Reliability:** Integrating Bayesian inference via **Variational Inference (BCNN-VI)** improves both diagnostic accuracy (94.63%) and generalization over deterministic CNNs without sacrificing performance.
2. **Clinical Safety Mechanism:** Predictive uncertainty effectively acts as an automated "flagging system"—high entropy directly correlates with misclassifications or ambiguous images that require expert human review.

### 🔮 Future Directions
* **Uncertainty-Aware Training:** Incorporating predictive uncertainty as an internal loss-weighting signal during training.
* **Uncertainty Decomposition:** Explicitly separating total uncertainty into *epistemic* (model uncertainty) and *aleatoric* (data noise) components.
* **Hybrid Architectures:** Selectively applying Variational layers only to deeper layers to reduce training runtime while retaining uncertainty estimates.
* **External Validation:** Testing generalizability on multi-institutional and multi-modal clinical MRI datasets.

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python
- **Frameworks:** TensorFlow, TensorFlow Probability, Keras
- **Image Processing & Data Handling:** OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Environment:** Kaggle / GPU-accelerated environment

---

## 📜 Citation & Acknowledgments

If you find this repository or study useful in your research, please cite the corresponding paper or repository.
