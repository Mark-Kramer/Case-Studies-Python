from scipy.io import loadmat
import warnings
from matplotlib import rcParams
from matplotlib.pyplot import plot, subplots, xlim, xlabel, ylim, ylabel, show, title, legend, style, semilogx, semilogy, savefig
import numpy as np
from numpy import zeros_like, ones_like, arange
from IPython.core.display import HTML
from IPython.lib.display import YouTubeVideo

HTML('../../assets/custom/custom.css')
style.use('ggplot')
rcParams['backend'] = 'TkAgg'
subplots()
rcParams['figure.figsize'] = (12, 3)
warnings.simplefilter(action='ignore', category=FutureWarning)

def figsize(w=12, h=3):
    rcParams['figure.figsize'] = (w, h)