# HytoolPy---Pumping-test-analysis-with-derivative-diagnostic
This package is a translation of some of the functionality of the Matlab HYTOOL code developed by Philippe Renard (Renard, Philippe (2017). Hytool: an open source matlab toolbox for the interpretation of hydraulic tests using analytical solutions. Journal of Open Source Software, 2(19), 441, doi:10.21105/joss.00441) into Python. 

It uses the derivative method to diagnose test curves and choose the right analytical solution. 
Bayesian fitting has been added to the initial code

This package contains analytical solutions used to describe groundwater flow around wells, and functions for importing, displaying, and fitting a model to the data. An example file is joined to the package to show the functionality.

You can install it '''pip install git+https://github.com/Celestin-Dartigues/Hytoolpy.git'''

The following point are on going to better the package :
1) More test
2) Streamlit function on going to do a more "click button version"
3) Implementing Papadopoulos with constant head boundary
4) Implement slug test resolution

