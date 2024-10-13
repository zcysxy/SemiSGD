# Comparisons of SemiSGD and FPI-type methods on Ring Road Speed Control

## Project Structure

- Main script to run the simulation:
    - `main.m`
- Algorithm implementations:
    - `qmi.m`: FPI-type methods
    - `gd.m`: SemiSGD
    - `gd_lfa.m`: SemiSGD with linear function approximation
- Reference and performance metric:
    - `opt.m`: calculate reference optimal equilibrium
    - `br.m`: calculate best response
    - `ip.m`: calculate induced population
    - `expl.m` calculate exploitability
- Helper functions:
    - `feat.m`: feature map for linear function approximation
    - `projsplx.m`: projection onto the simplex
- Plotting functions:
    - `plot_results.m`, `varplot.m`, `pretty.m`

## Development

Run the simulation:

```bash
matlab -r "main"
```

All the parameters are set in `main.m`. Modify this file to tweak the simulation settings.

Algorithm implementations are in `qmi.m` and `gd.m` for FPI-type methods and SemiSGD, respectively.
Notably, when the inner number of iterations $T$ is set to 1, FPI reduces to SemiSGD. Thus, you can use one script `qmi.m` for all algorithm implementations. This is implemented in other branches.

Reference and performance metric functions (`opt.m`, `br.m`, `ip.m`, `expl.m`) are for accurate evaluation. 
In the early stage of the development, you can disregard these functions, and evaluate the performance of an algorithm simply by comparing it with a converging FPI-type method with large number of iterations.
