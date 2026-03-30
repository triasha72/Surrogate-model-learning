# Project 1 — Gaussian Process Surrogate for the Branin Function

## What this project is about

The central problem in engineering simulation is cost. A single 
CFD run can take hours, sometimes days. You cannot afford to run 
thousands of them during design optimization. Surrogate models 
solve this by learning the input-output relationship from a small 
number of carefully chosen simulation runs, then predicting the 
output everywhere else — instantly.

This project builds the simplest possible version of that workflow 
using a well-known mathematical benchmark called the Branin function.

---

## The Branin Function

The Branin function is a standard benchmark in surrogate modeling 
literature. It takes two inputs and produces one output:
```
f(x1, x2) = a(x2 - bx1² + cx1 - r)² + s(1-t)cos(x1) + s
```

Where the constants are:
- a = 1
- b = 5.1 / (4π²)
- c = 5 / π
- r = 6
- s = 10
- t = 1 / (8π)

**Input ranges:** x1 ∈ [-5, 10], x2 ∈ [0, 15]

It has three global minima, all at f ≈ 0.398. The landscape 
looks like a hilly terrain with valleys and peaks — complex 
enough to be a meaningful test, simple enough to visualize.

In this project, the Branin function plays the role of an 
expensive simulation. We pretend each evaluation costs hours 
of compute time, so we can only afford 20 of them.

---

## Key Concepts

### Design of Experiments (DOE)
Before running any simulations, you need to decide *where* 
to evaluate. Random sampling wastes budget by clustering 
points. Latin Hypercube Sampling (LHS) divides the input 
space into equal intervals and places exactly one sample 
in each — giving much better coverage with the same budget.

### Gaussian Process (GP)
A Gaussian Process is a probabilistic surrogate model. 
Given training data, it predicts:
1. A **mean prediction** — what it thinks the output is
2. A **standard deviation** — how confident it is

The prediction at any point x follows a normal distribution:
```
f(x) ~ N(μ(x), σ²(x))
```

Where μ(x) is the predicted mean and σ(x) is the uncertainty.

### Kernel Function
The kernel controls how the GP interpolates between training 
points. We use the RBF (Radial Basis Function) kernel:
```
k(x, x') = σ² · exp(-||x - x'||² / (2l²))
```

Where:
- σ² = signal variance (overall scale of the function)
- l = length scale (how quickly the function changes)
- ||x - x'|| = distance between two input points

Intuitively: points that are close together in input space 
should have similar outputs. The length scale controls how 
"close" close needs to be.

### Uncertainty Map
The GP's uncertainty (σ) is highest where there is no 
training data. This map tells you exactly where to run 
your next simulation to improve the surrogate the most — 
a property no classical regression model has.

---

## Results

- **Training points:** 20 (Latin Hypercube Sampling)
- **R²:** 0.9553
- **Method:** Gaussian Process with RBF kernel

The surrogate captured 95.5% of the true function's variance 
from just 20 evaluations. The uncertainty map correctly 
identified the corners of the design space as the regions 
with least confidence — exactly where training data was sparse.

---

## Files
- `01_gp_surrogate_branin.ipynb` — full implementation
- `../results/gp_branin_result.png` — output plots
