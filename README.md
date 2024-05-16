## Custom-FINT: Enhanced Click-Through Rate (CTR) Prediction

**Updates:**

* **MLP Model Added:**  A new Multilayer Perceptron (MLP) model has been integrated for CTR prediction, offering an alternative architecture to FINT.
* **Prediction Script:**  `inference.py` facilitates CTR predictions on your test data using trained models.
* **Hyperparameter Optimization:** A hyperparameter optimization process is now available to fine-tune your models and achieve the best possible CTR prediction performance.

### Important Notes

* **Dataset Compatibility:** Currently, custom-FINT is optimized for use with the Avazu dataset. Compatibility with other datasets will be added in future releases.
* **Environment:** To ensure seamless compatibility with the original FINT repository, please use Python 3.7 and TensorFlow 1.15.5.

## Getting Started

### Usage:

#### FINT Model:

```bash
./run_fint.sh
```

#### MLP Model:

```bash
./run_mlp.sh
```

#### Prediction:

Make sure you have a `test.csv` file prepared in the correct format. Then run:

```bash
python inference.py
```

#### FINT Hyperparameter Optimization:

```bash
python train-opt.py
```

### Additional Notes:

* The `run_fint.sh` and `run_mlp.sh` scripts handle both model training and evaluation.
* Hyperparameter optimization is currently available only for the FINT model.

Let me know if you'd like any further adjustments or have specific sections you'd like to expand upon.
















**Source code for our paper:** [FINT: Field-aware INTeraction Neural Network For CTR Prediction](https://arxiv.org/pdf/2107.01999.pdf)

**If the code is helpful to you, please kindly cite our paper:**
Zhao, Z., Yang, S., Liu, G., Feng, D., & Xu, K. (2022, May). FINT: Field-Aware Interaction Neural Network for Click-Through Rate Prediction. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3913-3917). IEEE. 
