# Multi-Class Classification of Tifinagh Characters Using a Multilayer Perceptron

## Abstract

This report presents the implementation of a Multilayer Perceptron (MLP) for multi-class classification of Tifinagh handwritten characters. The model is trained on the Amazigh Handwritten Character Database (AMHCD), which contains 6,240 grayscale images (32×32 pixels) across 8 classes representing Tifinagh alphabet characters. The MLP architecture consists of an input layer (1024 neurons), two hidden layers (64 and 32 neurons with ReLU activation), and an output layer with softmax activation. The model achieves 89% accuracy on the test set using mini-batch stochastic gradient descent with categorical cross-entropy loss.

**Keywords:** Tifinagh, Handwritten Character Recognition, MLP, Deep Learning, AMHCD

---

## 1. Introduction

### 1.1 Background

Handwritten character recognition is a fundamental task in computer vision, particularly for less-studied scripts such as Tifinagh, used by Amazigh communities. The Amazigh Handwritten Character Database (AMHCD), introduced by Benaddy et al. (2020), provides a benchmark dataset for this purpose, containing 28,182 images of 64×64 pixels distributed across 33 classes representing the Tifinagh alphabet.

### 1.2 Problem Statement

The objective is to implement a multi-class neural network classifier capable of distinguishing between Tifinagh characters. The model must learn to map grayscale input images to their corresponding character classes with high accuracy, demonstrating the feasibility of deep learning approaches for low-resource scripts.

### 1.3 Objective

The primary goals are:
1. To implement a Multi-Layer Perceptron with configurable architecture
2. To apply ReLU activation functions in hidden layers and softmax in the output layer
3. To train the model using mini-batch stochastic gradient descent with categorical cross-entropy loss
4. To evaluate performance using accuracy, confusion matrix, and classification report

### 1.4 Dataset

The dataset used in this work is the AMHCD (Amazigh Handwritten Character Database). From the full AMHCD, 8 classes were selected for this exercise, containing 6,240 images total (780 per class). Each image is preprocessed to 32×32 pixels in grayscale and normalized to [0, 1]. The dataset is split into training (60%), validation (20%), and test (20%) sets.

---

## 2. Related Work

The AMHCD dataset was introduced by Benaddy et al. (2020) to support research on Amazigh handwritten character recognition. Previous work on Tifinagh character recognition has explored various machine learning approaches including support vector machines, convolutional neural networks, and template matching methods. This TP builds upon binary classification techniques previously studied, extending them to a multi-class setting.

---

## 3. Mathematical Formulation

### 3.1 Forward Propagation

For a layer $l$, the linear transformation and activation are:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$

where $A^{[0]} = X \in \mathbb{R}^{m \times 1024}$ is the input, $W^{[l]} \in \mathbb{R}^{n^{[l-1]} \times n^{[l]}}$ are the weights, and $b^{[l]} \in \mathbb{R}^{1 \times n^{[l]}}$ are the biases.

For hidden layers ($l = 1, 2$):
$$g^{[l]}(z) = \text{ReLU}(z) = \max(0, z)$$

For the output layer ($l = 3$):
$$A^{[3]} = \text{softmax}(Z^{[3]}), \quad \hat{y}_{i,c} = \frac{e^{z_{i,c}}}{\sum_{j=1}^{33} e^{z_{i,j}}}$$

### 3.2 Loss Function

The categorical cross-entropy loss is:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{33} y_{i,c} \log(\hat{y}_{i,c})$$

where $y_{i,c} = 1$ if class $c$ is correct, 0 otherwise.

### 3.3 Accuracy

$$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}[\arg\max_c(\hat{y}_{i,c}) = \arg\max_c(y_{i,c})]$$

### 3.4 Backpropagation

The initial gradient for the output layer:
$$\frac{\partial J}{\partial Z^{[3]}} = \hat{y} - Y$$

For hidden layers ($l = 2, 1$):
$$\frac{\partial J}{\partial Z^{[l]}} = \left(\frac{\partial J}{\partial Z^{[l+1]}} W^{[l+1]T}\right) \odot \text{ReLU}'(Z^{[l]})$$

where $\text{ReLU}'(z) = 1$ if $z > 0$, else $0$.

Parameter gradients:
$$\frac{\partial J}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^T \frac{\partial J}{\partial Z^{[l]}}$$

$$\frac{\partial J}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial J}{\partial Z^{[l]}}$$

Parameter updates with learning rate $\alpha = 0.01$:
$$W^{[l]} := W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$
$$b^{[l]} := b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}}$$

---

## 4. Implementation

### 4.1 Architecture

The implemented neural network follows the architecture specified in the TP:
- **Input layer:** 1024 neurons (32×32 flattened image)
- **Hidden layer 1:** 64 neurons with ReLU activation
- **Hidden layer 2:** 32 neurons with ReLU activation
- **Output layer:** 8 neurons with softmax activation (one per class)

Weight matrices are initialized using a standard normal distribution scaled by 0.01. Biases are initialized to zero. The random seed is set to 42 for reproducibility.

### 4.2 Training Procedure

The model is trained using mini-batch stochastic gradient descent with the following hyperparameters:
- **Learning rate:** 0.01
- **Epochs:** 100
- **Batch size:** 32
- **Optimizer:** SGD (no momentum)

Each epoch consists of shuffling the training data, iterating over mini-batches, performing forward propagation, computing the loss, and applying backpropagation. Validation metrics are computed after each epoch.

### 4.3 Preprocessing Pipeline

1. **Grayscale conversion:** Images are read as grayscale using OpenCV
2. **Resizing:** All images are resized to 32×32 pixels
3. **Normalization:** Pixel values are scaled to [0, 1] by dividing by 255
4. **Flattening:** Images are flattened into 1024-dimensional vectors
5. **Label encoding:** String labels are encoded as integers, then one-hot encoded

### 4.4 Code Structure

The implementation consists of the following components:
- `relu()`, `relu_derivative()`, `softmax()`: Activation functions
- `MultiClassNeuralNetwork`: Main class containing `forward()`, `backward()`, `train()`, `predict()`, `compute_loss()`, and `compute_accuracy()` methods
- Data loading and preprocessing pipeline
- Training loop with visualization

---

## 5. Results

### 5.1 Training Curves

The training and validation loss decrease steadily over epochs, indicating successful learning without severe overfitting. Training accuracy increases from ~13% (random) to ~90% by epoch 90, while validation accuracy reaches ~88%.

### 5.2 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ya    | 0.99      | 1.00   | 1.00     | 156     |
| yab   | 0.84      | 0.99   | 0.91     | 156     |
| yach  | 0.97      | 0.89   | 0.93     | 156     |
| yad   | 0.82      | 0.63   | 0.71     | 156     |
| yadd  | 0.91      | 0.88   | 0.90     | 156     |
| yae   | 0.71      | 0.97   | 0.82     | 156     |
| yaf   | 0.96      | 0.85   | 0.90     | 156     |
| yag   | 0.97      | 0.90   | 0.93     | 156     |
| **Overall** | **0.90** | **0.89** | **0.89** | **1248** |

### 5.3 Confusion Matrix

The confusion matrix reveals that classes `ya` and `yab` are most easily recognized, while `yad` presents the greatest challenge with only 63% recall. This suggests that certain Tifinagh character forms are visually similar and harder to distinguish.

### 5.4 Analysis

The model achieves good overall performance (89% accuracy) but shows variance across classes. Some classes (like `yad` and `yae`) have lower recall, indicating potential confusion between certain character shapes. The gap between training and validation accuracy is moderate, suggesting some overfitting but not severe.

---

## 6. Discussion

### 6.1 Key Observations

The MLP successfully learns to classify Tifinagh characters despite the limited training data (only 8 classes with 780 samples each). The relatively simple architecture achieves competitive results. However, the performance on visually similar characters suggests that a more sophisticated model might be needed for production use.

### 6.2 Limitations

1. The current implementation uses only 8 of the 33 available Tifinagh classes
2. No regularization is applied, which may lead to overfitting
3. The basic SGD optimizer without momentum converges slowly compared to adaptive methods
4. The fixed learning rate may not be optimal throughout training

### 6.3 Potential Improvements

1. **L2 Regularization:** Adding $\frac{\lambda}{m} W^{[l]}$ to the gradients would penalize large weights and reduce overfitting
2. **Adam Optimizer:** Using adaptive moments (first and second) would provide faster and more stable convergence
3. **K-Fold Cross-Validation:** Would provide a more robust estimate of model performance
4. **Data Augmentation:** Applying rotations and translations to training images would increase dataset diversity and improve generalization
5. **Full Dataset:** Expanding to all 33 Tifinagh classes would make the model more comprehensive

---

## 7. Conclusion

This TP successfully demonstrated the implementation of a multi-layer perceptron for multi-class classification of Tifinagh handwritten characters. The model achieves 89% accuracy on the test set, validating the effectiveness of MLP architectures for character recognition tasks. The mathematical foundations of forward propagation, backpropagation, and gradient descent were implemented from scratch, providing a deep understanding of neural network mechanics. Future work should focus on extending to the full Tifinagh alphabet (33 classes) and incorporating advanced optimization techniques and regularization methods.

---

## References

[1] Benaddy, M., et al. "Amazigh Handwritten Character Database (AMHCD)." Kaggle, 2020. Available at: https://www.kaggle.com/datasets/benaddym/amazigh-handwritten-character-database-amhcd

[2] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

---

*Source code available at: https://github.com/elhatimisanae426-hue/Classification-des-Caract-res-Tifinagh-niveaux-de-gris-avec-un-R-seau-de-Neurones-Multiclasses.git*
