# Transformer-Based Language Model

## Overview
This project implements a transformer-based language model with iterative improvements to reduce overfitting, optimize training, and improve generalization. By leveraging advanced deep learning techniques, the model achieves substantial gains in prediction accuracy and stability across datasets.

---

## Features
- **Multi-dataset Support**:
  
      Shakespeare Dataset (1,115,394 characters)
  
      QnA Dataset (31,985,086 characters)
  
      Ponniyin Selvan Dataset (1,661,635 characters)
  
      Harry Potter Dataset (683,693 characters)
  
- **Transformer Architecture**: Implements GPT-based attention mechanisms with self-attention, feed-forward layers, layer normalization, and dropout.
- **Iterative Improvements**: Refines the model through multiple milestones to address overfitting and enhance learning capacity.

---

## Improvements and Results
### Improvement 1
- **Changes**: Introduced weight decay, dropout, and decreased learning rate.
- **Impact**: 
  - **Training Loss**: Decreased by **up to 61%** at Step 4999.
  - **Validation Loss**: Decreased by **up to 18%**, reducing overfitting.

### Improvement 2
- **Changes**: Added learning rate scheduling and adjusted dropout.
- **Impact**: 
  - Enhanced consistency in loss reduction.
  - Improved training loss by **10.8%** and validation loss by **13.4%** at Step 4999.

### Improvement 3
- **Changes**: Tuned hyperparameters, introduced embedding dropout, and enhanced initialization weights.
- **Impact**:
  - **Training Loss**: Reduced by **up to 77%**.
  - **Validation Loss**: Reduced by **up to 34%**, achieving the best generalization.

---

## Results Summary
| **Model**         | **Training Loss Reduction** | **Validation Loss Reduction** |
|--------------------|-----------------------------|---------------------------------|
| **Improvement 1** | Up to 61.16%               | Up to 18.21%                  |
| **Improvement 2** | Up to 10.82%               | Up to 13.41%                  |
| **Improvement 3** | Up to 77.77%               | Up to 34.28%                  |

---

## Future Directions:
- Implement multilingual support.
- Further optimize computational efficiency for real-time deployment.
- Explore alternative datasets and evaluation metrics for broader generalization.

---

## Contributors: 
Laura, Shiv, Heather, Goutam, Drishtee, Sarah
