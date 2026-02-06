# Flow Model Dynamics Validation Summary

## Configuration
- **Environment**: hopper-medium-v2
- **Max data size**: 50000
- **Training epochs**: 100
- **Batch size**: 256
- **Learning rate**: 0.0003
- **Hidden dims**: [256, 256, 256]
- **Time embed dim**: 64
- **Layer norm**: True
- **Eval pairs**: 50
- **Neighbors (k)**: 30
- **Flow samples**: 30
- **ODE steps**: 30

## Training
- **Final loss**: 0.039185
- **Min loss**: 0.039185

## 1. Pointwise Prediction Accuracy (Most Important)

This measures whether the flow model can predict the true s' for a given (s, a).

- **Average MAE**: 0.0173
- **Average RMSE**: 0.0276
- **Assessment**: GOOD

### Per-Dimension Pointwise Error
| Dimension | MAE | RMSE |
|-----------|-----|------|
| dim_0 | 0.0069 | 0.0088 |
| dim_1 | 0.0069 | 0.0096 |
| dim_2 | 0.0131 | 0.0148 |
| dim_3 | 0.0073 | 0.0088 |
| dim_4 | 0.0090 | 0.0124 |
| dim_5 | 0.0168 | 0.0353 |
| dim_6 | 0.0180 | 0.0332 |
| dim_7 | 0.0425 | 0.0826 |
| dim_8 | 0.0139 | 0.0201 |
| dim_9 | 0.0254 | 0.0334 |
| dim_10 | 0.0300 | 0.0444 |

## 2. Global Marginal Distribution Comparison

This measures whether the overall distribution of flow-generated s' matches
the dataset distribution of s' (when conditioned on random (s, a) pairs).

- **Global Mean Abs Error (avg)**: 0.0161
- **Global Std Ratio (avg)**: 0.9885 (ideal: 1.0)
- **Mean Assessment**: GOOD
- **Std Assessment**: GOOD -- variance is well-matched

### Per-Dimension Global Comparison
| Dimension | DS Mean | Flow Mean | DS Std | Flow Std | Std Ratio |
|-----------|---------|-----------|--------|----------|-----------|
| dim_0 | -0.0000 | -0.0249 | 1.0000 | 0.9882 | 0.9882 |
| dim_1 | 0.0000 | 0.0018 | 1.0000 | 1.0233 | 1.0233 |
| dim_2 | -0.0000 | 0.0311 | 1.0000 | 0.9958 | 0.9958 |
| dim_3 | -0.0000 | 0.0065 | 1.0000 | 0.9726 | 0.9726 |
| dim_4 | -0.0000 | -0.0055 | 1.0000 | 1.0042 | 1.0042 |
| dim_5 | -0.0000 | -0.0259 | 1.0000 | 1.0157 | 1.0157 |
| dim_6 | 0.0000 | 0.0096 | 1.0000 | 0.9931 | 0.9931 |
| dim_7 | 0.0000 | 0.0007 | 1.0000 | 0.9946 | 0.9946 |
| dim_8 | 0.0000 | -0.0283 | 1.0000 | 1.0059 | 1.0059 |
| dim_9 | -0.0000 | 0.0250 | 1.0000 | 0.9146 | 0.9146 |
| dim_10 | -0.0000 | -0.0178 | 1.0000 | 0.9656 | 0.9656 |

## 3. Neighborhood (KNN) Conditional Comparison

This compares the flow model's conditional samples against KNN-based empirical
samples. Note: in near-deterministic environments, the flow model correctly
produces low variance (concentrated around the true s'), while KNN-based
empirical samples have higher variance because neighbors have different (s, a).

- **Neighbor Mean Abs Error (avg)**: 0.0833
- **Neighbor Std Ratio (avg)**: 0.0922

## Conclusion

The conditional flow matching model is a reasonable proposal distribution for p_0(s' | s, a) in offline RL on hopper-medium-v2.

- **Pointwise prediction**: GOOD (MAE = 0.0173, RMSE = 0.0276)
- **Global mean matching**: GOOD (avg error = 0.0161)
- **Global variance matching**: GOOD (std ratio = 0.9885)

Note: The flow model learns a near-deterministic mapping for each (s, a) -> s',
which is appropriate for environments with deterministic (or near-deterministic) dynamics.
The low conditional variance is expected and correct behavior.

## Plots
- `training_loss.png`: Training loss curve
- `pointwise_prediction.png`: Flow mean vs true s' scatter
- `pointwise_error_bars.png`: Per-dimension MAE/RMSE
- `mean_comparison.png`: KNN conditional mean comparison
- `std_comparison.png`: KNN conditional std comparison
- `per_pair_mean_scatter.png`: Per-pair mean scatter
- `global_comparison.png`: Global marginal mean and std
- `marginal_histograms_global.png`: Global marginal distributions
- `marginal_histograms_neighborhood.png`: Neighborhood marginal distributions
