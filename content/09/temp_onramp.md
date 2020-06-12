```{code-cell} ipython3
# Import modules ...
from scipy.io import loadmat                     # To load .mat files
import statsmodels.api as sm                     # To fit GLMs
from statsmodels.genmod.families import Poisson  # ... Poisson GLMs
from pandas import DataFrame as df               # Table object for working with data
from pylab import *                              # Numerical and plotting functions
%matplotlib inline

data = loadmat('spikes-1.mat')                             # Load the data,
t = data['t'][:, 0]                                        # Extract the t variable,
X = data['X'][:, 0]                                        # Extract the X variable,
spiketimes = data['spiketimes']                            # ... and the spike times.
spiketrain = histogram(spiketimes, 
                         bins = len(t), 
                         range = (t[0], t[-1]))[0] 
spikeindex = where(spiketrain!=0)[0]                       # Get the spike indices.

bin_edges = arange(-5, 106, 10)                            # Define spatial bins.
spikehist = histogram(X[spikeindex], bin_edges)[0]         # Histogram positions @ spikes.
occupancy = histogram(X, bin_edges)[0]*0.001               # Convert occupancy to seconds.
predictors = df(data={                                     # Create a dataframe of predictors
	'Intercept': ones_like(X),
	'X': X,
	'X2': X**2
	})

# GLM model with Poisson family and identity link function
model3 = sm.GLM(spiketrain, predictors, family=Poisson())  # Create the model
model3_results = model3.fit()                              # Fit model to our data
b3 = model3_results.params                                 # Get the predicted coefficient vector

bins = linspace(0, 100, 11)
bar(bins, spikehist / occupancy, width=8)                  # Plot results as bars.
plot(bins,                                                 # Plot model.
     exp(b3[0] + b3[1] * bins + b3[2] * bins**2) * 1000,
     'k', label='Model')
xlabel('Position [cm]')                                    # Label the axes.
ylabel('Occupancy norm. hist. [spikes/s]')
legend()
show()

```

<div class="question">
    
**Q:** Try to read the code above. Can you see how it loads data, estimates the parameters of a model, and then plots the model over the observations?

**A:** There is a lot happening here. Please continue on to learn this **and more**!

</div>

+++