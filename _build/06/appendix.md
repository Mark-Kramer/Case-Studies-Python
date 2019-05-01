---
interact_link: content/06/appendix.ipynb
kernel_name: python3
title: 'Multiplication and Convolution in Different Domains'
prev_page:
  url: /06/the-power-spectrum-part-2
  title: 'The Power Spectrum (Part 2)'
next_page:
  url: /08/basic-visualizations-and-descriptive-statistics-of-spike-train-data
  title: 'Basic Visualizations and Descriptive Statistics of Spike Train Data'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

<a id="top"></a> 
# Appendix: Multiplication and Convolution in Different Domains



{:.input_area}
```python
# Import numpy and matplotlib packages
import numpy as np
import matplotlib.pyplot as plt
```


We stated in this chapter the important fact that multiplication in the time domain is equivalent to convolution in the frequency domain. Mathematically, we may express this relation as,

$$
FT[xw] = FT[x]\star FT[w]
\tag{1}
$$

where $x$ and $w$ are two time series, $FT[x]$ is the Fourier transform of $x$, and $X \star Y$ indicates the convolution of $X$ and $Y$, 

$$X \star Y[\beta] = \int_{-\infty}^{\infty}X[b]Y[\beta - b]db.$$

The convolution of two functions (with arguments $b$ in this formula) is itself a function of the same argument (with symbol $\beta$ in this formula). Equation (1) states that the Fourier transform of the element-by-element product of $x$ and $w$ equals the convolution of the Fourier transform of $x$ and the Fourier transform of $w$. We consider here an equivalent, alternative statement: that convolution in the time domain is equivalent to multiplication in the frequency domain. Mathematically,

$$
FT[x \star w] = FT[x]FT[w].
$$

This equation states that the Fourier transform of the convolution of $x$ and $w$ equals the product of the Fourier transform of $x$ and the Fourier transform of $w$. To prove this relation, let’s consider the Fourier transform of the convolution of $x$ and $w$. We use the following expression for the continuous-time Fourier transform,

$$
FT(x \star w[\tau]) = \int_{-\infty}^{\infty}\big(x\star w[\tau]\big)e^{-2 \pi i f \tau}d\tau,
$$

where the notation $[\tau]$ indicates that the convolution $(x\star w)$ is a function of time $\tau$. Now, let's substitute the definition of convolution into this expression and simiplify using an introduction of a second exponential expression,

$$
\begin{align}
FT(x \star w[\tau]) = & 
\int_{-\infty}^{\infty}
\bigg( \int_{-\infty}^\infty x[t]w[\tau-t]dt \bigg)
e^{-2\pi i f \tau} 
d\tau \\
%
= & \int_{-\infty}^{\infty} 
\int_{-\infty}^\infty 
x[t]w[\tau-t]dt\ 
e^{-2\pi i f (\tau - t)} 
e^{-2\pi i f t} 
d\tau \\
%
= & \int_{-\infty}^{\infty} \int_{-\infty}^\infty 
\big( x[t] e^{-2\pi i f t} \big)
\big( w[\tau - t] e^{-2\pi i f (\tau - t)}\big)
dt\ d\tau.\\
\end{align}
$$

Setting $T \equiv \tau - t$, we find

$$
\begin{align}
FT(x \star w[\tau]) = & 
\int_{-\infty}^{\infty} \int_{-\infty}^\infty 
\big( x[t] e^{-2\pi i f t} dt \big)
\big( w[T] e^{-2\pi i f (T)} dT \big) \\
%
= & \bigg( \int_{-\infty}^{\infty}x[t] e^{-2\pi i f t} dt \bigg)
\bigg(\int_{-\infty}^{\infty} w[T] e^{-2\pi i f (T)} dT \bigg) \\
%
= & FT[x]FT[w]
%
\end{align}
$$

and therefore conclude that the Fourier transform of the convolution of $x$ and $w$ equals the element-by-element product of their Fourier transforms.

We may compute a simple example to illustrate this relation:



{:.input_area}
```python
x = np.arange(3, 7)                            # Define a simple signal x,
w = np.array([-1, 0.1, -0.2, 1])               # ... and another simple signal w.
a = np.fft.fft(np.convolve(w, x))              # Take the FT of the convolution,
b = np.fft.fft(np.hstack([w, [0, 0, 0]])) * \
    np.fft.fft(np.hstack([x, [0, 0, 0]]))      # ... and the product of the FTs
    
plt.plot(a.real - b.real, label='real')        # Plot the difference
plt.plot(a.imag - b.imag, label='imaginary')
plt.legend()
plt.title('Difference between Fourier transform of convolution \n and element-wise product of Fourier transforms\n')
plt.show()
```



{:.output .output_png}
![png](../images/06/appendix_7_0.png)



In the first two lines, we define two simple signals; each consists of only four elements, which is enough to illustrate the relation. In the third line, we first compute the convolution of $w$ and $x$, and then the Fourier transform. In the last line, we compute the Fourier transform of each variable, and then their element-by-element product. Notice that we zero-pad both variables before computing the Fourier transform in the last line. We do so to avoid computing circular correlations between the variables (i.e., wrapping around one variable when comparing it to another). Also, we make the lengths of variables $a$ and $b$ the same. Evaluating the statement, we find the difference between $a$ and $b$ to be on the order of $10^{-15}$, which is very close to zero.



{:.input_area}
```python
from IPython.core.display import HTML
HTML('../assets/custom/custom.css')
```





<div markdown="0" class="output output_html">
<style>
.left {
    margin-left: 0px;
}
.math-note {
    color: #3c763d;
    background-color: #dff0d8;
	border-color: #d6e9c6;
	border: 1px solid;
	border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    margin-top: 12px;
}
.python-note {
    color: #8a6d3b;
    background-color: #fcf8e3;
	border-color: #faebcc;
	border: 1px solid;
	border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    margin-top: 12px;
}
.question {
    color: #31708f;
    background-color: #d9edf7;
	border-color: #bce8f1;
	border: 1px solid;
    padding: 12px;
    margin-bottom: 12px;
    margin-top: 12px;
	border-radius: 10px;
}
.output_area img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.inner_cell img {
	width:100%;
	max-width:500px;
}
.thumb {
    position: inherit;
}
.thumb span { 
    width: 200px;
    visibility: hidden;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 5px;
    position: absolute;
    z-index: 2;
    right: 10%;
    transition: 5ms visibility;
}
.thumb img { 
	border:1px solid #000;
	margin:0px;
    background:#fff;
    width: 100%;
	max-width: 300px;
}
.thumb:hover, .thumb:hover span { 
	visibility:visible;
    transition-delay: 500ms;
		
} 
.fig {
    position: inherit;
}   
.fig img { 
	border:1px solid #000;
	margin:0px;
    background:#fff;
	width: 100%;
}
.fig span { 
	visibility: hidden;
    width: 500px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 5px;
    position: absolute;
    z-index: 2;
    right: 10%;
    transition: 5ms visibility;
}
.fig:hover, .fig:hover span { 
	visibility:visible;
    transition-delay: 500ms;
}
</style>

</div>


