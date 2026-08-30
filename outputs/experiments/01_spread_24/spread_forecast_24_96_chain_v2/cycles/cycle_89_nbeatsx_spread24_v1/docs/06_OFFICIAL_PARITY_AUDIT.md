---
status: active
date: 2026-08-29
scope: Cycle89 NBEATSx official-source parity
verification: locked commit fe116d21785fca55670d258756e7c35fcb613eca; pytest official basis/TCN tests
---

# Official NBEATSx parity audit

Reference repository: https://github.com/cchallu/nbeatsx  
Locked commit: \`fe116d21785fca55670d258756e7c35fcb613eca\`  
Read-only copy: \`third_party/reference_source/\`

| Item | OFFICIAL | LOCAL | Status |
|---|---|---|---|
| Identity basis | theta backcast prefix / forecast suffix | same split | MATCH |
| Trend basis | normalized polynomial grids and einsum | same | MATCH |
| Seasonality basis | zero + harmonic frequencies, cosine/sine templates | same | MATCH |
| Causal TCN | weight-normalized dilated residual blocks, Chomp1d | same computation; local modules are reorganized | NUMERICAL MATCH |
| WaveNet | causal exogenous convolution and basis projection | equivalent paper-style implementation | FORMULA REVIEW |
| Double residual | reverse residual, subtract backcast, additive forecast | same; initial level is last backcast point | MATCH |
| Forecast aggregation | level + sum of block forecasts | same | MATCH |
| Decomposition | block forecasts returned separately | same additive components | MATCH |
| Initialization | official-compatible named initializers | orthogonal/he_normal/glorot_normal paths | MATCH |
| Batch normalization | after activation in official block | supported after activation in local block | MATCH |
| Exogenous filtering | official include_var_dict supports EPF selection | local CORE5 is explicit and auditable | DIFFER: business adaptation |
| Static features | official optional static encoder | omitted from frozen CORE5 profile | DIFFER: not used by v1 |
| Full NBEATS block input | official input filtering depends on include_var_dict | local block receives explicit flattened temporal covariates | DIFFER: documented adaptation |
| Paper training | official StepLR/search protocol | local paper profile and smoke harness | STRUCTURAL ONLY |

The official Identity, Trend, Seasonality and TCN numerical tests compare
float32 tensors with copied/equivalent weights and tolerance \`1e-6\`.
The mapped numerical checks are recorded as SOURCE-EQUATION PASS; a full
official EPF training reproduction is not claimed because the paper search
run remains outside this readiness closure. The business model is explicitly
a strict H34 adaptation.
