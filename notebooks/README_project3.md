# Project 3 - Surrogate Method Comparison

## What this project is about

Projects 1 and 2 always used a Gaussian Process. But GP 
is not always the right choice. This project puts three 
surrogate methods head to head on the same problem to 
understand when each one makes sense.

---

## The Rosenbrock Function

The Rosenbrock function is a classic optimization benchmark:
```
f(x1, x2) = (1 - x1)² + 100(x2 - x1²)²
```

**Input ranges:** x1 ∈ [-2, 2], x2 ∈ [-1, 3]  
**Global minimum:** f(1, 1) = 0

It has a narrow, curved valley that follows the parabola 
x2 = x1². The valley floor is nearly flat (values near 0) 
while the surrounding walls are extremely steep (values 
in the thousands). This contrast makes it a hard surrogate 
problem — the model must learn both the gentle valley and 
the steep walls from the same training data.

---

## The Three Surrogate Methods

### 1. Response Surface Model (RSM)
A polynomial regression model. The inputs are expanded 
into polynomial terms and a linear regression is fitted:
```
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₁x₂ + β₅x₂²
```

Where β are coefficients learned from training data.

**Strengths:** Extremely fast, simple to interpret  
**Weaknesses:** Cannot capture highly nonlinear responses.
A degree-2 polynomial is a bowl shape — it cannot 
represent a curved valley.

### 2. Gaussian Process (GP)
A probabilistic surrogate that predicts both a mean 
and uncertainty at every point. Uses the RBF kernel:
```
k(x, x') = σ² · exp(-||x - x'||² / (2l²))
```

**Strengths:** Handles nonlinearity well, provides 
uncertainty estimates, very accurate with limited data  
**Weaknesses:** Training cost scales as O(n³) with 
number of training points — becomes slow for large datasets

### 3. Radial Basis Function Interpolant (RBF)
Fits a weighted sum of radial basis functions centered 
at each training point. Uses the multiquadric basis:
```
φ(r) = √(r² + c²)
```

Where r is the distance from a training point and 
c is a shape parameter.

**Strengths:** Fast, smooth interpolation, exact at 
training points  
**Weaknesses:** No uncertainty estimates, can 
extrapolate poorly beyond training data

---

## Results

| Method | R² | MAPE | Train Time |
|--------|-----|------|------------|
| RSM | 0.247 | 1763% | 0.004s |
| GP | 1.000 | 0.55% | 0.146s |
| RBF | 0.895 | 187% | 0.002s |

### Why RSM failed
A degree-2 polynomial is fundamentally the wrong shape 
for Rosenbrock's curved valley. No amount of data would 
fix this — it is a structural limitation of the model.

### Why RBF got R²=0.895 but MAPE=187%
R² measures how well the model captures variance across 
the whole space — and RBF got the overall shape roughly 
right. But MAPE is sensitive to errors at small values. 
Near the valley floor where f ≈ 0, even a small absolute 
error produces a huge percentage error. RBF struggled 
most in exactly that region.

### Why GP won
The RBF kernel gives the GP the flexibility to represent 
arbitrary nonlinear shapes, not just quadratic ones. 
The uncertainty-driven kernel optimization found the 
right length scales for this specific problem.

---

## When to use each method

| Situation | Recommended |
|-----------|-------------|
| Nearly quadratic response, large dataset | RSM |
| Nonlinear response, limited data | GP |
| Large dataset, speed critical, no uncertainty needed | RBF |
| Need to know where to sample next | GP only |

---

## Files
- `03_surrogate_comparison.ipynb` — full implementation
- `../results/surrogate_comparison_clean.png` — comparison plots
