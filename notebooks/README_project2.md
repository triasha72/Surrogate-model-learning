# Project 2 — Cantilever Beam Deflection Surrogate

## What this project is about

Project 1 used a mathematical benchmark with no physical 
meaning. This project applies the same surrogate workflow 
to a real structural engineering problem — predicting the 
tip deflection of a cantilever beam.

The goal is to replace an expensive FEA (Finite Element 
Analysis) solver with a GP surrogate that predicts deflection 
in milliseconds across a 4-dimensional design space.

---

## The Physics

A cantilever beam is fixed at one end and free at the other 
— like an aircraft wing attached to the fuselage, or a 
diving board. When a force is applied at the free end, 
the tip deflects downward.

The analytical formula for tip deflection is:
```
δ = (F × L³) / (3 × E × I)
```

Where:
- **δ** = tip deflection (m) — what we want to predict
- **F** = applied force (N) — how hard you push
- **L** = beam length (m) — longer beam deflects more
- **E** = Young's modulus (Pa) — material stiffness
            (steel ≈ 200 GPa, aluminum ≈ 70 GPa)
- **I** = second moment of area (m⁴) — cross-section shape
            (larger I = stiffer beam)

Notice that L appears cubed — doubling the beam length 
increases deflection by 8x. This nonlinearity is what 
makes the surrogate problem interesting.

---

## Input Space

| Variable | Physical meaning | Range |
|----------|-----------------|-------|
| F | Applied force | 100 – 1000 N |
| L | Beam length | 0.5 – 2.0 m |
| E | Young's modulus | 70 – 210 GPa |
| I | Second moment of area | 1×10⁻⁷ – 1×10⁻⁵ m⁴ |

---

## Key Concepts

### Curse of Dimensionality
Going from 2 inputs (Project 1) to 4 inputs dramatically 
increases the space that needs to be covered. With 30 
samples in 4D space, coverage is extremely sparse — this 
is why the first attempt gave R² = 0.64.

### Log Transformation
E spans 70 GPa to 210 GPa. I spans 1×10⁻⁷ to 1×10⁻⁵ m⁴. 
These variables differ by orders of magnitude. Without 
transformation, the GP treats a change from 70 GPa to 
71 GPa the same as a change from 70 GPa to 140 GPa in 
relative terms — which is wrong.

Taking log(E) and log(I) compresses these ranges so 
the GP can treat all variables equally:
```
x_transformed = log(x_raw)
```

This single change improved R² from 0.64 to 1.0.

### StandardScaler
After log-transformation, inputs are rescaled to zero 
mean and unit variance:
```
x_scaled = (x - mean) / std
```

This ensures no single variable dominates the GP's 
distance calculations simply because of its scale.

### Kernel Length Scales
After training, the GP's optimized kernel revealed:
```
RBF(length_scale=[F, L, E, I])
```

Smaller length scale = surrogate is more sensitive to 
that variable. L had the smallest length scale, confirming 
what the physics tells us — deflection is most sensitive 
to beam length because L appears cubed in the formula.

---

## Results

| Attempt | Samples | R² | MAPE |
|---------|---------|-----|------|
| First attempt | 30 | 0.64 | 19.4% |
| After fixes | 80 | 1.00 | 0.26% |

**Key lesson:** Feature engineering (log-transforming inputs 
that span orders of magnitude) mattered more than increasing 
the sample count. The model didn't change — only how the 
data was fed to it.

---

## Files
- `02_beam_deflection_surrogate.ipynb` — full implementation
- `../results/beam_deflection_30samples.png` — output plots