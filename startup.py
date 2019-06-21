from scipy.io import loadmat  # To load .mat files
import matplotlib.pyplot as plt  # Load plotting functions
from matplotlib import rcParams  # Allow us to change appearance of figures
# Import specific plotting functions that we use frequently
from matplotlib.pyplot import plot, subplots, xlim, xlabel, ylim, ylabel, show, title, legend, style, semilogx, semilogy, savefig
import numpy as np  # Commonly used package for numerical computations
from numpy import zeros_like, ones_like, arange  # ... and some specific functions that we use regularly
from IPython.core.display import HTML  # Package for manipulating appearance of notebooks
from IPython.lib.display import YouTubeVideo  # Package for displaying YouTube videos

HTML('../../assets/custom/custom.css')  # Change some style features of notebooks
style.use('ggplot')  # Change default plot appearance
rcParams['backend'] = 'TkAgg'  # Make plots appear 'inline' (equivalent to %matplotlib inline)

def figsize(w=12, h=3):  # Function to make plots default to wide size
    rcParams['figure.figsize'] = (w, h)