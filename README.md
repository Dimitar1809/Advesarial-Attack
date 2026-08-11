# Adversarial Attack and Defense on Audio Models

Evaluating how a spectrogram-based CNN trained for spoken-digit classification holds up against
a Projected Gradient Descent (PGD) attack, and whether a standard adversarial-training defense
can recover its robustness.

Most adversarial-robustness research targets image models. This project asks the same question
for audio: how vulnerable is a spectrogram classifier to gradient-based perturbations, and how
far does a common defense actually go?

Course project for the MSc Deep Learning course, University of Twente.

---

## Research question

How effective is adversarial training at mitigating a PGD attack on spectrogram-based audio
models?

## Approach

**Dataset.** [Audio MNIST](https://arxiv.org/abs/1807.03418) — 30,000 spoken-digit recordings
(0–9) from 60 speakers. Each `.wav` file is converted to a spectrogram via Short-Time Fourier
Transform before being fed to the model.

**Model.** A CNN with the same architecture as AlexNet (same layers, kernels, padding, strides),
trained on the spectrograms in TensorFlow. Clean-data accuracy: **97%**.

**Attack.** Multi-step PGD, implemented from scratch for spectrogram inputs (normalized to
[0, 1]):
- Random perturbation initialization within an ε-ball
- 40 iterative steps, step size α = 0.01
- Perturbation projected back onto an L∞ ball of radius ε = 0.1 after every step

**Defense.** Single-step FGSM-based adversarial training: each training batch is perturbed
in the direction of the loss gradient before the model is trained on it, with the goal of
learning representations less sensitive to small perturbations. (Defensive distillation is
covered as a theoretical mitigation in the report but was not the defense that was implemented
and evaluated here — the evaluated defense is the FGSM adversarial-training loop above.)

## Results

| | Clean accuracy | Accuracy under PGD attack |
|---|---|---|
| Before defense | 97% | ~10% |
| After FGSM adversarial training | — | ~10% (no meaningful improvement) |

The PGD attack collapses classification accuracy from 97% to roughly 10% on 1,000 validation
examples. Retraining the model with single-step FGSM adversarial examples did **not** recover
robustness against the stronger, multi-step PGD attack — final adversarial accuracy stayed
around 10%. The takeaway: a single-step defense is a mismatch for a multi-step attack, and
closing that gap is exactly where this project goes next.

## Future work

The defense is the weak point of this project, and it's what I want to improve next:

- **Match the defense to the attack.** Retrain using multi-step PGD adversarial examples
  instead of single-step FGSM ones, since the evaluation attack is itself multi-step.
- **Actually implement and evaluate defensive distillation**, rather than leaving it at the
  theoretical-analysis stage, and test it in combination with adversarial training.
- **Add preprocessing-based defenses** — feature squeezing (median filtering / bit-depth
  reduction) and randomized smoothing at inference time — as cheap additions on top of
  adversarial training.
- **Try ensemble and randomized-transformation defenses** to reduce reliance on any single
  model's decision boundary.
- **Report attack/defense strength with more than accuracy**: attack success rate, L2/L∞
  perturbation norms, and ideally a certified-robustness bound (e.g. via interval bound
  propagation) rather than empirical accuracy alone.
- **Strengthen the attack side too**, with an adaptive epsilon schedule and Expectation-Over-
  Transformation (EOT), so the defense is tested against a harder adversary.

## Repository contents

```
Covert_wav_to_spectogram.ipynb   Audio preprocessing: .wav -> spectrogram (STFT)
Train_model.ipynb                Trains the AlexNet-style CNN on the spectrograms
PGD Attack Updated.py            PGD attack implementation and evaluation
main.ipynb                       End-to-end pipeline: attack + FGSM adversarial-training defense
```

## Running it

There's no `requirements.txt` yet. Based on what the pipeline uses, you'll need:

```
tensorflow  numpy  scipy  matplotlib
```

Order: convert the Audio MNIST `.wav` files to spectrograms (`Covert_wav_to_spectogram.ipynb`),
train the classifier (`Train_model.ipynb`), then run the attack and defense evaluation
(`PGD Attack Updated.py` / `main.ipynb`).

## Reference

Becker, S., Ackermann, M., Lapuschkin, S., Müller, K.-R., & Samek, W. (2018). Interpreting and
explaining deep neural networks for classification of audio signals. *arXiv:1807.03418*.
