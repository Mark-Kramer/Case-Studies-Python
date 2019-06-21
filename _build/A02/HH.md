---
redirect_from:
  - "/a02/hh"
interact_link: content/A02/HH.ipynb
kernel_name: python3
title: 'Hodgkin Huxley Model'
prev_page:
  url: /11/spike-field-coherence
  title: 'Analysis of Spike-Field Coherence'
next_page:
  url: /A03/LIF
  title: 'Integrate and Fire Model'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# The Hodgkin-Huxley model

In this module we will use Python to simulate the Hodgkin-Huxley (HH) neuron model.  This model is arguably the *most* important computational model in neuroscience.  We'll focus here on simulating this model and understanding its pieces.

## Background information about the HH model

Here's a video that describes some of the biophysical details of the HH model:



{:.input_area}
```python
from IPython.lib.display import VimeoVideo
VimeoVideo('140084450')
```





<div markdown="0" class="output output_html">

        <iframe
            width="400"
            height="300"
            src="https://player.vimeo.com/video/140084450"
            frameborder="0"
            allowfullscreen
        ></iframe>
        
</div>



Here are some additional usual videos and references:

- [Lecture by Prof. Gerstner, *Detailed Neuron Model (a)*](http://klewel.com/conferences/epfl-neural-networks/index.php?talkID=4)

- [Lecture by Prof. Gerstner, *Detailed Neuron Model (b)*](http://klewel.com/conferences/epfl-neural-networks/index.php?talkID=5)

##  Preliminaries

Before beginning, let's load in the Python packages we'll need:



{:.input_area}
```python
import numpy as np
import math
%matplotlib
import matplotlib.pyplot as plt
```


{:.output .output_stream}
```
Using matplotlib backend: MacOSX

```

In addition, let's import the functions we'll need to simulate the HH model, which are available on this repository:



{:.input_area}
```python
from HH_functions import HH
```


##  Part 1:   The Hodgkin-Huxley (HH) equation code.

To start, let's examine the code for the HH model. We can do so in (at least) two ways.

- Go to the Case Studies repository, and examine the Python file
`HH_functions.py`
- Examine the code inline with `inspect`



{:.input_area}
```python
import inspect
inspect.getsourcelines(HH)
```





{:.output .output_data_text}
```
(['def HH(I0,T0):\n',
  '    dt = 0.01;\n',
  '    T  = math.ceil(T0/dt)  # [ms]\n',
  '    gNa0 = 120   # [mS/cm^2]\n',
  '    ENa  = 115;  # [mV]\n',
  '    gK0  = 36;   # [mS/cm^2]\n',
  '    EK   = -12;  # [mV]\n',
  '    gL0  = 0.3;  # [mS/cm^2]\n',
  '    EL   = 10.6; # [mV]\n',
  '\n',
  '    t = np.arange(0,T)*dt\n',
  '    V = np.zeros([T,1])\n',
  '    m = np.zeros([T,1])\n',
  '    h = np.zeros([T,1])\n',
  '    n = np.zeros([T,1])\n',
  '\n',
  '    V[0]=-70.0\n',
  '    m[0]=0.05\n',
  '    h[0]=0.54\n',
  '    n[0]=0.34\n',
  '\n',
  '    for i in range(0,T-1):\n',
  '        V[i+1] = V[i] + dt*(gNa0*m[i]**3*h[i]*(ENa-(V[i]+65)) + gK0*n[i]**4*(EK-(V[i]+65)) + gL0*(EL-(V[i]+65)) + I0);\n',
  '        m[i+1] = m[i] + dt*(alphaM(V[i])*(1-m[i]) - betaM(V[i])*m[i]);\n',
  '        h[i+1] = h[i] + dt*(alphaH(V[i])*(1-h[i]) - betaH(V[i])*h[i]);\n',
  '        n[i+1] = n[i] + dt*(alphaN(V[i])*(1-n[i]) - betaN(V[i])*n[i]);\n',
  '    return V,m,h,n,t\n'],
 22)
```



<div class="alert alert-block alert-info">

**Q:**  Examine this code.  Can you make sense of it?  Can you identify the
gating variables?  The rate functions?  The equations that define the dynamics?
We'll answer these questions in this in module, but try so on your own first.

</div>

Whenever examining code, it's useful to consider the *inputs* to the code, and the *outputs* produced by the code.  There are two inputs to `HH0`:

- `I0` = the current we inject to the neuron.
- `T0` = the total time of the simulation in [ms].

And there are five outputs:

- `V` = the voltage of neuron.
- `m` = activation variable for Na-current.
- `h` = inactivation variable for Na-current.
- `n` = activation variable for K-current.
- `t` = the time axis of the simulation (useful for plotting).


## Part 2:  At low input current (`I0`), examine the HH dynamics.

  To understand how the HH model works, we'll start by focusing on the
  case when `I0` is small. Let's fix the input current to zero,




{:.input_area}
```python
I0 = 0
```


and let's simulate the model for 100 ms,



{:.input_area}
```python
T0 = 100
```


We've now defined both inputs to the `HH` function, and can execute it, as follows,



{:.input_area}
```python
[V,m,h,n,t]=HH(I0,T0)
```


Notice that the function returns five outputs, which we assign to the variables `V`, `m`, `h`, `n`, and `t`.

<div class="alert alert-block alert-info">

**Q:**  What are the dynamics of the voltage (variable `V`) resulting
from this simulation?<br>
HINT:  Plot `V` vs `t`.

</div>

<div class="alert alert-block alert-info">

**Q:**   What are the dynamics of the gating variables (`m`, `h`, `n`)
resulting from this simulation?<br>
HINT:  Plot them!

</div>

<div class="alert alert-block alert-info">

**Q:**  What are the final values (after the 100 ms of simulation) of
`V`, `m`, `h`, and `n`?

</div>

### Observation for Part 2
At this value of input current (`I0=0`), the model dynamics
approach a "fixed point", whose location we can identify as a point in four dimensional space.


## Part 3:  At high input current (`I0`), examine the HH dynamics of a spike.
  Let's now increase the input current to the HH model and get this model
  to generate repeated spiking activity.  To do so, let's set,




{:.input_area}
```python
I0 = 10
```


We can now simulate this model,



{:.input_area}
```python
[V,m,h,n,t] = HH(I0,T0)
```


<div class="alert alert-block alert-info">
**Q:**  What happens to the dynamics?<br>
HINT:  Plot V vs t.
</div>

  ### Observation for Part 3
  You should have found that, at this value of input current, the model
  **generates repeated spikes**.
  
  Let's now explore how the combined gates
  and dynamics evolve.  To do so, let's start by focusing our plot on a
  single spike.  As a first step, we'll make a new figure with a seperate subfigure to plot
  the voltage,



{:.input_area}
```python
plt.figure()
plt.subplot(211)
```





{:.output .output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1096794e0>
```



This `subplot` command divides the figure into two rows, and one column, and tells Python we'll start in the first row. See Python Help for more details:

`plt.subplot??`

Now, let's plot the voltage, and choose the time axis to focus on a single spike,



{:.input_area}
```python
plt.plot(t,V,'k')
plt.xlim([42, 56])
plt.ylabel('V [mV]');
```


  Okay, we've now plotted the voltage dynamics for a single spike (and
  colored the curve black).  Let's now plot the three gating variables.
  To do so, we'll move to the next subplot,





{:.input_area}
```python
plt.subplot(212);
```


(the next row in the figure).  Within this subplot, let's start by displaying the gating variable `m` over the same x-limits,



{:.input_area}
```python
plt.plot(t,m,'r', label='m')
plt.xlim([42, 56]);
```


  Notice that, in the call to `plot` we included the input `label`. This will be useful when we create a legend ... <br><br>Within this subplot, we can also simultaneously show the gating
  variables `h` and `n`,




{:.input_area}
```python
plt.plot(t,h,'b', label='h')
plt.plot(t,n,'g', label='n');
```


Label the x-axis,



{:.input_area}
```python
plt.xlabel('Time [ms]');
```


Now, let's add a legend to help us keep track of the different curves,



{:.input_area}
```python
plt.legend();
```


<div class="alert alert-block alert-info">
**Q:** Using the figure you created above, describe how the gates swing open and closed during a spike.
</div>

### ASIDE:
Here's a nice plotting trick, to link the x-axes of our two subfigures.  Linking the axes is useful so that, when we zoom or move one subfigure, the other subfigure will match the x-axis.



{:.input_area}
```python
plt.figure()
ax1 = plt.subplot(211);                 # Define axis for 1st subplot,
ax2 = plt.subplot(212, sharex=ax1);     # ... and link axis of 2nd subplot to the 1st.
ax1.plot(t,V,'k')                       # Plot the voltage in the first subplot,
plt.xlim([42, 56]);
ax2.plot(t,m,'r', label='m')            # ... and the gating variables in the other subplot.
ax2.plot(t,h,'b', label='h')
ax2.plot(t,n,'g', label='n');
plt.xlabel('Time [ms]');
plt.legend();
```


Now, in the figure, you may use the pan/zoom tool to adjust the linked subplots.

## Part 4:  At high input current (`I0`), describe the dynamics of the conductances.
  In Part 3, we explored how the three gates `m`, `h`, and `n` evolve
  during a spike.  By combining these terms, we can visualize how the
  *conductances* evolve during a spike.  To do so, let's stick with the
  simulation results we generated in Part 3, and focus our plot on a
  single spike,




{:.input_area}
```python
plt.figure()
ax1=plt.subplot(311)                # Make a subplot,
ax1.plot(t,V,'k')                   #... and plot the voltage,
plt.xlim([42, 56])                  #... focused on a single spike,
plt.ylabel('V [mV]');               #... with y-axis labeled.
```


Now, to plot the conductances, let's define three new variables,



{:.input_area}
```python
gNa0 = 120
gNa  = gNa0*m**3*h                 # Sodium conductance
gK0  = 36
gK   = gK0*n**4                    # Potassium conductance
gL0  = 0.3
gL   = gL0*np.ones(np.shape(gK))   # Leak conductance
```


<div class="alert alert-block alert-info">
**Q:** Where do these terms come from?
</div>

Then, let's plot these conductances,



{:.input_area}
```python
ax2 = plt.subplot(312, sharex=ax1)  #Make a second subplot,
ax2.plot(t,gNa,'m', label='gNa')    #... and plot the sodium conductance,
ax2.plot(t,gK, 'g', label='gK')     #... and plot the potassium conductance,
ax2.plot(t,gL, 'k', label='gL')     #... and plot the leak conductance.
plt.xlim([42, 56])                  #... focused on a single spike,
plt.xlabel('Time [ms]')             #... label the x-axis.
plt.ylabel('mS/cm^2')               #... and label the y-axis.
plt.legend();                       #... make a legend.
```


<div class="alert alert-block alert-info">
**Q:** How do the conductances evolve during a spike?
</div>

## Part 5:  At high input current (`I0`), describe the dynamics of the *currents*.
  In Part 4, we explored how the three conductances (`gNa`, `gK`, `gL`) evolve
  during a spike.  Let's now visualize how the *ionic currents* evolve
  during a spike.  To do so, let's stick with the same settings used in
  Part 4 and examine the same simulation result.  Again, we'll focus our plot
  on a single spike.
  
  
  Now, to plot the *current*, let's define the new variables,





{:.input_area}
```python
gNa0 = 120
ENa  = 115
INa  = gNa0*m**3*h*(ENa-(V+65))    # Sodium current.
gK0  = 36
EK   =-12
IK   = gK0*n**4*(EK-(V+65))        # Potassium current.
gL0  = 0.3
EL   = 10.6;
IL   = gL0*(EL-(V+65))             # Leak current.

ax3=plt.subplot(313, sharex=ax1)   # Make a third subplot,
ax3.plot(t,INa,'m', label='INa')   #... and plot the sodium current,
ax3.plot(t,IK, 'g', label='IK')    #... and plot the potassium current,
ax3.plot(t,IL, 'k', label='IL')    #... and plot the leak current.
plt.xlim([42, 56])                 #... focus on a single spike,
plt.xlabel('Time [ms]')            #... label the x-axis.
plt.ylabel('mA/cm^2')              #... and label the y-axis.
plt.legend();                      #... make a legend.
```


<div class="alert alert-block alert-info">
**Q:** How do the conductances evolve during a spike?
</div>

<div class="alert alert-block alert-info">
**Q:** You may notice a small, transient decrease in the sodium current `INa` near 47 ms. What causes this?
</div>
