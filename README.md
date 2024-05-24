# Wind Turbine Optimisation Code

This project optimises the desing of a 3D section wind turbine. 

The specifications are 1m swept radius. Inflow wind velocity of 1.5m/s.

To complete this task, Evolutionary Algorithms were employed, primarily the Genetic Algorithm. Performing coordinate descent would result in significant computation time when optimising the objective function due to the several nested loops. The coefficient of performance (Cp) is solved for via the Blade Element Method.

$$ \text{MAX} \frac{Cp}{\text{Volume}} $$

Several subject to constraints were also employed.

![Lithium Bromide - Water Absorption Cooling Cycle](WindTurbine.JPG)
