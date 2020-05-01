# Case-Studies-Python

### Quick start to reading and running the Python code in your browser:

- Click here to start working in Binder: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Mark-Kramer/Case-Studies-Python/master?filepath=content)
- Select a section, and click on the `.ipynb` file

----
### THIS IS VERY MUCH A WORK IN PROGRESS

This [repository](https://github.com/Mark-Kramer/Case-Studies-Python.git) is a companion to the textbook [Case Studies in Neural Data Analysis](https://mitpress.mit.edu/books/case-studies-neural-data-analysis), by Mark Kramer and Uri Eden.  In the textbook, we use MATLAB to analyze examples of neuronal data.  The material here is very similar, except that we use Python (instead of MATLAB).

Our intended audience is the "practicing neuroscientist" - e.g., the students, researchers, and clinicians collecting neuronal data in the hospital or lab.  The material can get pretty math-heavy, but we've tried to outline the main concepts as directly as possible, with hands-on implementations of all concepts.  We focus on only two main types of data: spike trains and electric fields (such as the local field potential [LFP], or electroencephalogram [EEG]).  If you're interested in other data (e.g., calcium imaging, or BOLD), you may still find the examples indirectly useful (for example, demonstrations of how to compute and interpret a power spectrum of a signal).

There are two ways to interact with these notebooks.  First, you could run it locally in <a href="https://jupyter.org/">Jupyter</a>. This is an excellent choice because you'll be able to read, edit and execute the Python code directly in your browser and you can save any changes you make or notes that you want to record.  The second way is to open this notebook in <a href="https://mybinder.org/v2/gh/Mark-Kramer/Case-Studies-Python.git/master?filepath=content">Binder</a> and interact with the notebooks through a JupyterHub server. Binder provides an easy interface to interact with this material; read about it in [eLife here](https://elifesciences.org/labs/a7d53a88/toward-publishing-reproducible-computation-with-binder).  In any case, we encourage you to execute each line of code in the files!

We assume you have installed Python and can get it running on your computer.  Some useful references to do so include,

<ul>
  <li><a href="https://www.python.org/">Python.org</a></li>
  <li><a href="https://conda.io/docs/user-guide/install/index.html">Conda</a></li>
</ul>

If this is your first time working with Python, using <a href="https://conda.io/docs/user-guide/install/index.html">conda</a> is probably a good choice. Conda is a package and environment manager that makes it really easy to get up and running in Python. In particular, we recommend installing Miniconda - a light version of the software distribution Anaconda - and using conda to add software as needed.

We'd like to thank all of the students, collaborators, and funders who have helped make this possible!
