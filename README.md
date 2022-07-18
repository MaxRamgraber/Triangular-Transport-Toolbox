# Triangular Transport Toolbox

<img align="left" src="https://github.com/MaxRamgraber/Triangular-Transport-Toolbox/blob/main/figures/spiral_animated.gif" height="300px">

This repository contains the code for my triangular transport implementation. To use it in your own code, simply download the file `transport_map.py`, copy it into your working directory, import the class `transport_map` into your Python code with the line `from transport_map import *`. That's it!

The practical use and capabilities of this toolbox are illustrated in a number of example files:

 - **Examples A - spiral distribution** illustrates the basic use of the map, from the parameterization of a transport map object to its use for forward mapping, inverse mapping, and conditional sampling.
 - **Examples B - statistical inference** builds on previous established basics to illustrate the use of transport methods for statistical inference. The first examples examines statistical dependencies between temperatures in two cities, the second demonstrates Bayesian parameter inference for Monod kinetics.
 - **Examples C - data assimilation** demonstrates the use of transport maps for Bayesian filtering and smoothing, using the chaotic Lorenz-63 system. These examples also introduce the use of map regularization, the possibility of separation of the map update, and the exploitation of conditional independence.

---

If you're curious about other triangular transport libraries, I recommend checking out the the [**M**onotone **Par**ameterization **T**oolbox MParT](https://measuretransport.github.io/MParT/). It is a joint effort by colleagues from MIT and Dartmouth to create an efficient toolbox for the monotone part of the transport map, realized in C++ for computational efficiency, with bindings to a wide variety of programming languages (Python, MATLAB, Julia). While it does not (yet) support as many features as this toolbox at time of writing, it is being actively developed. Keep an eye on it!
