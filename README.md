# Triangular Transport Toolbox

<img align="left" src="https://github.com/MaxRamgraber/Triangular-Transport-Toolbox/blob/main/figures/spiral_animated.gif" height="300px">

This repository contains the code for my triangular transport implementation. To use it in your own code, simply download the file `transport_map.py`, copy it into your working directory, import the class `transport_map` into your Python code with the line `from transport_map import *`. That's it!

The practical use and capabilities of this toolbox are illustrated in a number of example files:

 - **Examples A - spiral distribution** illustrates the basic use of the map, from the parameterization of a transport map object to its use for forward mapping, inverse mapping, and conditional sampling.
 - **Examples B - statistical inference** builds on previous established basics to illustrate the use of transport methods for statistical inference. The first examples examines statistical dependencies between temperatures in two cities, the second demonstrates Bayesian parameter inference for Monod kinetics.
