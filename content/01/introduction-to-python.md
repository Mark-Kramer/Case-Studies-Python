---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.4.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Python for the practicing neuroscientist

To be frank:  this notebook is rather boring.  Throughout all of the case studies, we will use the software package [Python](https://www.python.org/). The best way to learn new software (and probably most things) is when motivated by a particular problem.  Would you read assembly instructions for furniture you do not plan to own?  Probably not. In other sections we will pursue specific questions driven by neuronal data, and use our desire to understand these data to motivate the development and application of computational methods.  But not in this section.  Here, we focus on basic coding techniques and principles in Python in the abstract, without motivation.  You - poor reader - must trust that these ideas and techniques will eventually be useful.  We begin by dipping our toe into the Python pool, and learning the basic strokes;  the fun and interesting parts in the "real world" of neuronal data happen later.

+++

Let us delay no further.  In the following examples, you are asked to execute code in Python.  If your Python experience is limited, you should actually *do* this, not just read the text below.  If you intend to ignore this advice - and not execute the code in Python - then instead walk to the local coffee shop, get a double espresso, and return to attempt these examples.  This notebook follows in spirit and sometimes in detail notebook 2 of <a href="https://www.elsevier.com/books/matlab-for-neuroscientists/wallisch/978-0-12-383836-0">MATLAB for Neuroscientists</a>, an excellent reference for learning to use MATLAB in neuroscience with many additional examples.  If you have never used Python before, there are many excellent resources online (e.g., the [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)).

+++

## Starting Python

There are two ways to interact with this notebook.  First, you could run it locally on your own computer using <a href="https://jupyter.org/">Jupyter</a>. This is an excellent choice, because you'll be able to read, edit and excute the Python code directly and you can save any changes you make or notes that you want to record.  The second way is to open this notebook in your browser using [Binder](https://mybinder.org/v2/gh/Mark-Kramer/Case-Studies-Python/master), and execute the examples directly in your browser, without installing additional software on your computer. In any case, we encourage you to execute each line of code in this file!

+++

<div class="python-note">
    
Throughout this notebook, we assume that you are running [Python 3](https://www.python.org/download/releases/3.0/). Most of the functions used here are the same in Python 2 and 3. One noteable exception however is division. If you are using Python 2, you will find that the division operator `/` actually computes the *floor* of the division if both operands are integers (i.e., no decimal points). For example, in Python 2, `4/3` equals `1`. While, in Python 3, `4/3` equals `1.333`.

We encourage you to use Python 3 for the sake of compatibility with this notebook, as well as for compatibility with future releases of Python.

</div>

+++

## On-ramp: analysis of neural data in Python
We begin this notebook with an "*on-ramp*" to analysis in Python. The purpose of this on-ramp is to introduce you immediately to some aspects of Python. You may not understand all aspects of the Python language here, but that's not the point. Instead, the purpose of this on-ramp is to  illustrate what *can* be done. Our advice is to simply run the code below and see what happens...

```{code-cell} ipython3
import scipy.io as sio               # Import packages to read data, do analysis, and plot it.
from pylab import *
%matplotlib inline

mat = sio.loadmat('sample_data.mat') # Load the example data set.
t   = mat['t'][0]                    # Get the values associated with the key 't' from the dictorary.
LFP = mat['LFP'][0]                  # Get the values associated with the key 'LFP' from the dictorary

                                     # Print useful information about the data.
print("Sampling frequency is " + str( 1/(t[2]-t[1]))  + ' Hz.')
print("Total duration of recording is " + str(t[-1]) + ' s.')
print("Dimensions of data are " + str(shape(LFP)) + ' data points.')

initial_time_interval = t < 5        # Choose an initial interval of time, from onset to 5 s,
                                     # ... and plot it.
plot(t[initial_time_interval], LFP[initial_time_interval])
xlabel('Time [s]')
ylabel('LFP')
title('Initial interval of LFP data');
```

<div class="question">

**Q:** Try to read the code above. Can you see how it loads data, extracts useful information to print, then selects an interval of data to plot?

**A:** If you've never used Python before, that's an especially difficult question. Please continue on to learn more!

</div>

+++

## Example 1: Python is a calculator

Execute the following commands in Python:

```{code-cell} ipython3
4+9
```

```{code-cell} ipython3
4/3
```

<div class="question">

**Q:**  What does Python return?  Does it make sense?

</div>

+++

## Example 2.  Python can compute complicated quantities.

Enter the following command in Python:

```{code-cell} ipython3
4/10**2
```

<div class="question">

**Q:** Does this answer make sense?

</div>

+++

<div class="question">

**Q:** Can you use parentheses to change the answer?

</div>

+++

##  Example 3.  Python has useful built in functions.

A function is a program that operates on arguments. Standard math functions and variables (and other useful things) can be accessed from the `math` and `pylab` modules. To use the `math` and `pylab` modules, we must first import both:

```{code-cell} ipython3
from pylab import *
import math
```

<div class="python-note">

The `pylab` module bundles commonly used numeric and plotting functions. Another common alternative is to use

	import numpy as np
	import matplotlib.pyplot as plt

In this style, we would need to indicate which module, or *namespace*, contains the function we want to call: `x = np.arange(10)` or `plt.plot(x, y)`.

</div>

You will often begin your data analysis with `import` statements, to load the functionality you need. We can now call functions from math using `numpy.*`.  For example,

```{code-cell} ipython3
sin(2*pi)
```

Above, `sin` is the sine function.  It operates on the argument `2*pi`.  Notice that, once we have imported the `numpy` module, Python knows the value of `pi`. Here's another example function that operates on arguments:

```{code-cell} ipython3
math.atan(2*pi)
```

<div class="question">

**Q:**  What is `math.atan`?

**A:** To answer this, try using Python Help.  To start the Python Help, simply put a `?` at the end of `math.atan` and then run this code block. 

</div>

```{code-cell} ipython3
math.atan?
```

You should see a description of the function pop up at the bottom of the window.

**NOTE:**  Python Help is *extremely* useful.  You can always look there when you have questions about a function, or search the internet for help, i.e., [google it](http://www.google.com).

+++

##  Example 4.  We can use Python to define lists of numbers.

In Python, there are several different data structures that are designed to store more than one element. Here we will focus on the `array` data structure, but if you are curious to know how and when to use other structures, there is a good explanation <a href='http://www.physics.nyu.edu/pine/pymanual/html/chap3/chap3_arrays.html'>here</a>. Let's define an array:

```{code-cell} ipython3
a = array([[1, 2, 3, 4]])
```

##  Example 5.  We can manipulate arrays by scalars.

A scalar is a single number.  Consider,

```{code-cell} ipython3
a = array( [[1, 2, 3, 4]] )
print( a * 3 )
print( 4 * a )
print( a + 1 )
```

<div class="question">
    
**Q:** What do you find?

**A:** Notice that the scalar operates on each element of the array.

</div>

+++

##  Example 6.  We can manipulate arrays with arrays. 

Let's create an array and multiply it by itself,

```{code-cell} ipython3
a = array([1,2,3,4])
a * a
```

<div class="question">

**Q:**  What does this return?

**A:** We see that the operator `*` performs element-by-element multiplication of the values in array `a`.

</div>

+++

## Example 7. More examples of manipulating arrays with arrays.

```{code-cell} ipython3
a = 2
b = array( [[0, 4, 7, 6]] )
c = array( [[1, 5, 6, 8]] )

print( b * c )
print( b / c + a)
print( multiply( b, c ))
```

<div class="question">

**Q:** What operation does `multiply()` perform?

</div>

+++

## Example 8.  We can probe the variables we've defined in Python.

To see a list of the variables you've defined, type `who` or `whos` in a code block by themselves. Notice `whos` provides more information.

```{code-cell} ipython3
who
```

```{code-cell} ipython3
whos
```

## Example 9.  Determine the dimensions of an array

To examine the dimensions of an array, we can ask for the `shape`,

```{code-cell} ipython3
a = array([[1,2,3,4]])
print(a.shape)
print(shape(a))
```

We find that the shape of `a` is `(1,4)` or 1 row and 4 columns. Notice we have two options to execute the `shape` function:

- In `a.shape` we return the attribute `shape` of the variable `a`. 

- In `shape(a)` we apply the function `shape` to the variable `a`.

The result is equivalent.

+++

## Example 10. Sometimes we need to reset the workspace

By doing so, we get rid of all the variables. To do so, type `%reset` and enter `y`

```{code-cell} ipython3
%reset
```

<div class="question">

**Q.** What command could we use to confirm there are no variables in the workspace?

**A.** Consider `who`.

</div>

```{code-cell} ipython3
who
```

<div class="python-note">
    
The `%reset` command is an example of a *magic*. Magics are commands that start with the `%` symbol and use a language other than Python. They are only available in the notebook environment. In fact, the set of magics that is available is specific to the notebook kernel. This means that if you have a Jupyter notebook running a Ruby kernel the magics will be different.

</div>

+++

##  Example 11.  We can define matrices in Python.
A matrix is an array with more than one dimensio.  Consider the following:

```{code-cell} ipython3
from pylab import *  # Have to reimport as we cleared the workspace above!

p = array( [[1,2,3],[4,5,6]] )
```

This creates a matrix with two rows and three columns. Consider,

```{code-cell} ipython3
print( p )
```

<div class="question">

**Q:** Can you see the two rows and three columns?

</div>

We can manipulate matrices like we manipulate vectors.

```{code-cell} ipython3
print( p + 2 )
print( 2 * p )
print( p * p )
```

##  Example 12.  Indexing matrices and vectors.
Matrices and vectors are arrays of numbers, and sometimes we want to
access individual elements or small subsets of these lists.  That's
easy to do in Python.  Consider,

```{code-cell} ipython3
a = array( [1, 2, 3, 4, 5] )
b = array( [6, 7, 8, 9, 10] )
```

 Python indexes from 0 (like C, C++, Java, and unlike MATLAB and Fortran which start at 1). 
 To access the 2nd element of `a` or `b`, type `a[1] / b[1]`.
 We'll be a bit fancier with our printing now to distinguish variables. 
 Calling `str(a)` converts the variable `a` to a string that can be printed easily.
 Adding two strings just concatenates them: `"hi" + " bye" = "hi bye". `

```{code-cell} ipython3
print( "a[1] = " + str(a[1]) )
print( "b[1] = " + str(b[1]) )
```

<div class="question">
    
**Q.**  Do the results make sense?  How would you access the 4th element of each vector?

</div>

+++

  We can combine `a` and `b` to form a matrix with `a` as the first row and `b` as the second.
  Note that we apply the function `array()` to the *list* `[a,b]`, which it converts to a matrix.

```{code-cell} ipython3
c = array([a,b])
print( "c = \n" + str(c) )    # \n is a newline, or carriage return, which makes the printed matrix lineup better 
```

  To learn the size (or shape)
 of `c` we use `shape()`:

```{code-cell} ipython3
print( "shape of c = " + str( shape(c) ) )
```

  The shape of `c` is `[2 5]`.  It has two rows and five columns.  To access
  the individual element in the 1st row and 4th column of `c`, type `c[0,3]`

```{code-cell} ipython3
print( "c[0,3] = " + str( c[0,3] ) )
```

<div class="python-note">
    
We access matrices using 'row, column' notation.  So `c[0,3]` means
print the element in row 0, column 3 of `c`.

</div>

+++

<div class="question">

**Q.**  How would you print all rows in the 2nd column of `c`?

</div>

+++

##  Example 13:  We can find subsets of elements in matrices and vectors.
Often we are interested in only some of the elements of a matrix or vector. For example, we might want to look at the data from a single experimental trial which is stored in a particular row of a matrix. Alternatively, we might want to find out when the values in a time series cross a given boundary. Doing this is simple in Python.

+++

### Slicing
*Slicing* means that we want to look at a specific portion of a vector or matrix, for example, the first row of a matrix. We will continue with the matrix `c` from the previous example. The notation '`:`' means '*all indices*'. To access all columns in the entire first row of `c`, type `c[0,:]`. To access the 2nd thru 4th columns of the first row of `c`, type `c[0,1:4]`.

```{code-cell} ipython3
print( "c[0,:] = " + str( c[0,:] ) )
print( "2nd through 4th columns of the first row are c[0,1:4] = " + str(c[0,1:4]) )
```

  <div class="python-note">
    
  The notation `1:4` means *all integers from 1 up to, but not including 4*, 
  which in this case gives columns 1, 2, and 3. 
  
  </div>

+++

Leaving out the number before the colon tells Python to start at index 0. Leaving out the number after the colon tells Python to continue all the way to the end.

```{code-cell} ipython3
print("c[0, :4] = " + str( c[0,:4]))
print("c[0, 1:] = " + str( c[0,1:]))
```

We can also tell Python how to step through the indices. To access only the even columns of `c`, we can use the following:

```{code-cell} ipython3
c[0,::2]
```

This code tells Python to start at 0, continue to the end, and step by 2. The result should be the values in row 0, columns 0, 2, and 4 of `c`. We could write this explicitly as `c[0,0:5:2]`.

+++

  ### Selecting elements that satisfy a condition
  Sometimes we're interested in locating particular values within a
  matrix or vector.  As an example, let's first define a vector.

```{code-cell} ipython3
a = arange(1,10)    # this creates a vector of increasing values from 1 to 9
a = 2*a 

print( "a = " + str(a) )
```

<div class="question">

**Q.**  Calculate the shape of `a`.  What is the maximum value of `a`? 
*Hint:* Use the `max()` function.

</div>

+++

  Now let's find all values in `a` that exceed 10.

```{code-cell} ipython3
a[a > 10]
```

This is called logical indexing, let's look at what `a>10` returns:

```{code-cell} ipython3
lgIdx = a > 10
lgIdx
```

When we index `a` using this array `lgIdx` we get back only the entries 
in `a` corresponding to `True`, as above:

```{code-cell} ipython3
a[lgIdx]
```

Sometimes we want to know the actual indices in a where `a > 10`.
We can get them using the `nonzero()` array method, which returns the
index of all entries that were `True`, or non-zero.

```{code-cell} ipython3
lgIdx.nonzero()
```

<div class="python-note">
    
The command `nonzero()` can be used as both a *function* and a *method*. A method is called by adding it after the object it is meant to operate on with a period in between ( `lgIdx.nonzero()` ). A function is called with the *argument* explicitly provided inside the parentheses ( `nonzero(lgIdx)` ). Basically, a function and a method do the same thing, but a function needs to be given an argument, while a method assumes that the argument is the object that the method is attached to. Note that if we use `nonzero()` as a function, we need to tell it to look in NumPy for the definition (i.e. add `` at the beginning of the function call). 

</div>

+++

Now we have another way to select the desired elements of `a`:

```{code-cell} ipython3
a[ (a > 10).nonzero() ]
```

We can use these two types of indexing to change subsets of the values of `a`.

```{code-cell} ipython3
print("a = " + str(a))
a[a > 10] = 100
print("a = " + str(a))
```

**Q:** How does `a` change in the first and second print statements?

We can perform these same logical operations for a matrix,

```{code-cell} ipython3
b = array([[1,2,3],[4,5,6],[7,8,9]])
print( "b = \n" + str(b) )
print( " b > 5 is \n" + str(b > 5) )
print(" b[b>5] is an array: " + str(b[b>5]) )
```

<div class="python-note">
    
Notice that the last line collapses the `True` entries to an array, 
ordered by row and then by column. If you've used MATLAB, this is 
the opposite of what it does!

</div>

+++

##  Example 14:  Plotting data in Python.
It's not easy to look at lists of numbers and gain an intuitive
feeling for their behavior, especially when the lists are long.  In
these cases, it's better to visualize the lists of numbers by
plotting them.  Consider

```{code-cell} ipython3
x = linspace(0,10,11)   
print( "x = " + str(x) )

#   The above line constructs a vector that starts at 0, ends at 10, and
#   has 11 entries (takes steps of size 1 from 0 to 10). Let

y = sin(x)
print( "y = " + str(y) )
```

<div class="question">

**Q.** Looking at the values in 'y' printed above, can you tell what's happending?

**A.** Not really ... let's visualize `y` vs `x` instead.

</div>

+++

### matplotlib
To visualize `y` versus `x` let's plot it. To do so, let's first import some basic plotting routines from `matplotlib`, which provides a nice [2D plotting library](https://matplotlib.org/). We'll also tell Python to show  `matplotlib` graphics inline, in this notebook.

Let's start by plotting a simple example for `x` and `y`,

```{code-cell} ipython3
%matplotlib inline
from pylab import *

x = ([1, 2, 3, 4])
y = x
plot(x,y) 
show()          # this is the plotting equivalent of print()
```

<div class="question">

**Q.** Does the plot above make sense for the variables `x` and `y`?

</div>

Now, let's go back to the definitions of `x` and `y` that we started this example with and plot `y` versus `x`.

```{code-cell} ipython3
x = linspace(0,10,11)   
y = sin(x)

plot(x, y)
show()
```

 The plot of `x` versus `y` should look a bit jagged, and not 
 smooth like a sinusoid.  To make the curve smoother,
 let's redefine `x` as,

```{code-cell} ipython3
x = linspace(0,10, 101)
print(x)
```

<div class="question">

**Q.**  Compare this definition of `x` to the definition above.  How do these
two definitions differ?

</div>

+++

<div class="question">
    
**Q.**  What is the size of `x`?  Does this make sense?

</div>

+++

Now let's replot the sine function.

```{code-cell} ipython3
y = sin(x)
plot(x,y,'k')   # the 'k' we've added makes the curve black instead of blue
show()
```

<div class="question">

**Q.** Does this plot make sense, given your knowledge of `x`, `y`, and trigonometry?

</div>

+++

## Example 15: What if we want to compare several functions?

+++

Continuing the example in the previous section, let’s define a second vector

```{code-cell} ipython3
z = cos(x)
```

and plot it:

```{code-cell} ipython3
plot(x,z)
show()
```

We’d now like to compare the two variables `y` and `z`. To do this, let’s plot both vectors on
the same figure, label the axes, and provide a legend,

```{code-cell} ipython3
plot(x,z)                     # plot z vs x.
plot(x,y,'r')                 # plot y vs x in red
show()
```

Notice that we’ve included a third input to the function `plot`. Here the third input tells Python to draw the curve in a particular color: `'r'` for red. There are many options we can use to plot; to see more, check out the documentation for [plot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html).

We can also label the axes, give the figure a title, and provide a legend,

```{code-cell} ipython3
plot(x,z)                     # plot z vs x
plot(x,y,'r')                 # plot y vs x in red
xlabel('x')                   # x-axis label
ylabel('y or z')              # y-axis label
title('y vs x and z vs x')    # title
legend(('y','z'))             # make a legend labeling each line
show()
```

To futher edit this plot, you might decide - for example - that the font size for the labels is too small. We can change the default with:

```{code-cell} ipython3
rcParams.update({'font.size': 12})  
rcParams['axes.labelsize']=14      # make the xlabel/ylabel sizes a bit bigger to match up better

# we can change the default linewidth with
rcParams['lines.linewidth']=2

# let's make a new plot to check 
plot(x,y, label='y')     # sometimes it is easier to name a trace within the plot() call
plot(x,z, label='z')     # notice without a color matplotlib will assign one
xlabel('x')
ylabel('y')
title('y vs x')
legend()
show()
```

##  Example 16:  We can make random numbers in Python.
To generate a single Gaussian random number in Python, use the function in the NumPy `random` module.

```{code-cell} ipython3
print("a Gaussian random number (mean=0, variance=1): " + str( randn() ))

# a uniform random number on [0,1)
print("a uniform random number from [0,1): " + str(rand()))
```

Let's generate a vector of 1000 Gaussian random numbers:

```{code-cell} ipython3
r = randn(1000)
```

... and look at a histogram of the vector:

```{code-cell} ipython3
hist(r)
show()
```

<div class="question">

**Q.** Does this histogram make sense? Is it what you expect for a distribution of Gaussian random variables?

</div>

+++

<div class="python-note">
    
See Python Help (`hist?`) to learn about the function `hist()`.

</div>

+++

##  Example 17:  Repeating commands over and over and over . . . 
  Sometimes we'll want to repeat the same command over and over again.
  For example, what if we want to plot `sin(x + k*pi/4)` where `k` varies from 1 to 5 in
  steps of 1;  how do we do it?  Consider the following:

```{code-cell} ipython3
x = arange(0,10,0.1)  # Define a vector x that ranges from 0 to 9.9 with step 0.1.
k = 1                    # Fix k=1,
y = sin(x + k*pi/4)# ... and define y at this k.

figure()             # Make a new figure,
plot(x,y)            # ... and plot y versus x.

k = 2                    # Let's repeat this, for k=2,
y = sin(x + k*pi/4)# ... and redefine y at this k,
plot(x,y)            # ... and plot it.

k = 3                    # Let's repeat this, for k=3,
y = sin(x + k*pi/4)# ... and redefine y at this k,
plot(x,y)            # ... and plot it.

k = 4                    # Let's repeat this, for k=4,
y = sin(x + k*pi/4)# ... and redefine y at this k,
plot(x,y)            # ... and plot it.

k = 5                    # Let's repeat this, for k=5,
y = sin(x + k*pi/4)# ... and redefine y at this k,
plot(x,y)            # ... and plot it.

show()
```

That's horrible code!  All I did was cut and paste the same thing four times.  As a general rule, if you're repeatedly cutting and pasting in code, what you're doing is inefficient and typically error prone.  There's a much more elegant way to do this, and it involves making a `for` loop.  Consider:

```{code-cell} ipython3
x = arange(0,10,0.1)        #First, define the vector x.
```

Now let's declare a `for` loop where `k` successively takes the values 1, then 2, then 3, ..., up to 5. Note, any code we want to execute as part of the loop must be indented one level. The first line of code that is not indented, in this case `show()` below, executes after the for loop completes

```{code-cell} ipython3
for k in range(1,6):               
    y = sin(x + k*pi/4)      #Define y (note the variable 'k' in sin), also note we have indented here!
    plot(x,y)                  #Plot y versus x
    
# no indentation now, so this code follows the loop
show() 
```

The small section of code above replaces all the cutting-and-pasting.
  Instead of cutting and pasting, we update the definition of `y` with different values of `k` and plot it within this for-loop.
  
<div class="question">

**Q.**  Spend some time studying this for-loop.  Does it make sense?

</div>

+++

<div class="python-note">
    
**Important note:** Python uses **indentation** to define `for` loops.

</div>

+++

##  Example 18:  Defining a new function.
We've spent some time in this notebook writing and executing code. Sometimes we'll need to write our own Python functions.  Let's do that now.

Our function will do something very simple: it will take as input a
vector and return as output the vector elements squared plus an additive
constant.

If have a vector, `v`, and a constant, `b`, we would like to call:

    vsq = my_square_function(v, b)
    
This won't work!  We first need to define `my_square_function`. Let's do so now,

```{code-cell} ipython3
def my_square_function(x, c):
    """Square a vector and add a constant.

    Arguments:
    x -- vector to square
    c -- constant to add to the square of x
    
    Returns:
    x*x + c
    """
    
    return x * x + c    
```

The function begins with the keyword `def` followed by the function name and the inputs in parentheses. Notice that this first line ends with a colon `:`. All of the function components that follow this first line should be **indented one level**. This is just like the `for` loop we applied earlier; the operations performed by the for loop were indented one leve.

+++

<div class="python-note">
    
When defining the function, the code the function executes should be indented one level.

</div>

+++

The text inside triple quotes provides an optional documentation string that describes our function. While optional, including a '*doc string*' is an important part of making your code understandable and reuseable.

The keyword `return` exits the function, and in this case returns the expression `x * x + c`. Note that a return statement with no arguments returns `None`, indicating the absence of a value.

With the function defined, let's now call it. To do so we first define the inputs, and then run the function, as follows:

```{code-cell} ipython3
v = linspace(0.,10.,11)
b = 2.5

# Now let's run the code,
v2 = my_square_function(v, b)
print("v = " + str(v))
print("v*v+2.5 = " + str(v2))
```

To see the doc string that describes our function, type `my_square_function?`

```{code-cell} ipython3
# Let's check that our docstring works
my_square_function?
```

<div class="question">

**Q.** Try to make a function, my_power, so that 
`y = power(x,n)` evaluates $y = x^n$, 
(in Python you can use `x**n` to take the power)

</div>

+++

<div class="question">

**Q.** Try to make a function, my_power, so that 
`y = power(x,n)` evaluates $y = x^n$, 
(in Python you can use `x**n` to take the power)

</div>

+++

## Example 19: Animating figures 
Finally, let's make an animation in Python. To do this we need two additional functions from external modules: `HTML()` and `FuncAnimation()`. `FuncAnimation` is what creates the animated figure, while `HTML` tells the notebook that to interpret the argument as HTML and show the results.

```{code-cell} ipython3
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
```

In English, we set up (or initialize) the figure and then make a function that does all of the updates for each frame of the animation. Finally, we pass the figure, the function, and the frame numbers to `FuncAnimation()` and we have our animation.

Here's an example in which we plot a sinusoid of different heights, and allow the user to adjust the heights with a slider.

```{code-cell} ipython3
x = linspace(0.,2.,1001)                 # Define x from 0 to 2 with 1001 steps.
lines = plot(x, 0. * sin(x*pi))   # Make the first plot, save the curve in "lines"
axis([0, 2, -1, 1])                     # Set the x and y limits in the plot
title("plot number = 0")                # ... and label the plot.

def animate(frame):                         # Define the function to perform the animation.
    lines[0].set_ydata(float(frame) / 100. * sin(x * pi)) # Change the y values at each x location
    title('plot number = ' + str(frame))# Update the title with the new plot number
    
fig = FuncAnimation(gcf(), animate, frames=range(100))
HTML(fig.to_jshtml())
```

## Example 19: Load MATLAB data into Python
For our last example let's load a MATLAB file in the `.mat` format into Python. Before doing so, let's clear all of the variables and functions we have defined. This command is not necessary, but we perform it here so that any new variables we subsequently load are obvious.

```{code-cell} ipython3
%reset
```

Then, let's import the `scipy.io` module, which we'll use to import the `.mat` data,

```{code-cell} ipython3
import scipy.io as sio
```

Now, let's load a data file using the function `loadmat`,

```{code-cell} ipython3
mat = sio.loadmat('sample_data.mat')
type(mat)
```

The variable that holds the loaded data is a dictionary. In Python, a dictionary is like a list, with the elements of the dictionary accessed via “keys”. Let's print these keys:

```{code-cell} ipython3
print(mat.keys())
```

<div class="python-note">
    
Use the `keys()` method to see what variables are contained in `mat`. In other words, run the command `mat.keys()`.

</div>

+++

The two keys of interest to us are `t` and `LFP`. Our collaborator who provided the data tells us that these correspond to a time axis (`t`) and voltage recording (`LFP`), respectively, for her data. Let's define variables to hold the data corresponding to each key,

```{code-cell} ipython3
t = mat['t'][0]                      # Get the values associated with the key 't' from the dictorary.
LFP = mat['LFP'][0]                  # Get the values associated with the key 'LFP' from the dictorary
```

Now, let's plot the LFP data versus the time axis,

```{code-cell} ipython3
from pylab import *
%matplotlib inline

# Choose a subset to plot
t = t[0:500]
LFP = LFP[0:500]

plot(t, LFP)
title('My plot')
xlabel('Time [s]')
ylabel('Voltage [$\mu$ V]') # Wrap latex characters in $..$
show()
```
