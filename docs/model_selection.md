# Final Model Selection Report (Week 4)

## Problem
Churn is the minority class (26.5%, about 2.8 non-churners per churner). The Week 3 models reached 79% accuracy but detected only about half of the churners (recall 0.50-0.53).

## What was tried
1. **Imbalance handling** (class weights, SMOTE, random undersampling) on 4 models.
2. **Stratified 5-fold cross-validation** on the training set to check consistency.
3. **Hyperparameter tuning** (GridSearchCV for Logistic Regression and Decision Tree, RandomizedSearchCV with 25 combinations for Random Forest and Gradient Boosting), objective F1-score, with class weights.

## Impact of imbalance handling (test set, change vs baseline)
| Model | Strategy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | Class weight | -0.132 | +0.278 | +0.048 | 0.000 |
| Gradient Boosting | Class weight | -0.132 | +0.316 | +0.065 | -0.001 |
| Decision Tree | Class weight | -0.148 | +0.278 | +0.025 | -0.003 |
| Random Forest | Class weight | -0.009 | -0.019 | -0.015 | +0.002 |
| Random Forest | Undersampling | -0.118 | +0.278 | +0.062 | +0.004 |

Class weights, SMOTE and undersampling raise recall by about 0.15-0.3 at the cost of 9-15 points of precision. ROC-AUC barely changes because these methods move the decision point, not the ranking of customers. **Class weights were selected**: same F1, simplest, no data deleted or invented. SMOTE was not better and worked worse with trees.

## Cross-validation (F1, mean over 5 folds)
Standard deviations are small (0.01-0.03), so results are consistent. With class weights: Logistic Regression 0.626, Gradient Boosting 0.626, Decision Tree 0.600, default Random Forest 0.542 (the weakest and the most unstable in precision).

## Final comparison (test set, top versions)
| Model version | Precision | Recall | F1 | ROC-AUC | Train ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest, tuned** | 0.547 | 0.778 | **0.642** | **0.845** | 0.907 |
| Decision Tree, tuned | 0.525 | 0.775 | 0.626 | 0.834 | 0.867 |
| Gradient Boosting, class weight | 0.508 | 0.813 | 0.625 | 0.840 | 0.882 |
| Logistic Regression, class weight | 0.504 | 0.791 | 0.616 | 0.842 | 0.851 |
| Logistic Regression, baseline (Week 3) | 0.636 | 0.513 | 0.568 | 0.842 | 0.851 |

## Decision
**Random Forest, tuned, with class weights** (`max_depth=12`, `min_samples_leaf=15`, `max_features=0.5`, `n_estimators=150`, `class_weight="balanced"`).

- Best F1 and ROC-AUC, and the best cross-validated F1 of the search (0.635).
- Detects 291 of the 374 churners in the test set (recall 0.78) with 241 false alarms and 83 missed churners.
- Overfitting is much lower than the default forest (train ROC-AUC 0.91 vs 1.00).
- Business fit: losing a churner costs recurring revenue, while a false alarm only costs a retention offer.

## Limitations
- Differences between the best versions are small (F1 0.615-0.642) and the test set has only 374 churners, so the ranking is not conclusive. Tuned Logistic Regression is a simpler alternative with almost the same ROC-AUC.
- The decision threshold remains 0.5 and could be adjusted to the real cost of a retention action.
- XGBoost could not be loaded on the development machine (missing `libomp`), so sklearn's Gradient Boosting was used.
