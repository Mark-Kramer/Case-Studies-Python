# Case-Studies-Python

This repository is a companion to the textbook <a href="https://mitpress.ublish.com/book/case-studies-neural-data-analysis" rel="external" target="_blank">Case Studies in Neural Data Analysis</a>, by Mark Kramer and Uri Eden. That textbook uses MATLAB to analyze examples of neuronal data. The material here is  similar, except that we use Python.

The intended audience is the *practicing neuroscientist* - e.g., the students, researchers, and clinicians collecting neuronal data in the hospital or lab.  The material can get pretty math-heavy, but we've tried to outline the main concepts as directly as possible, with hands-on implementations of all concepts.  We focus on only two main types of data: spike trains and electric fields (such as the local field potential [LFP], or electroencephalogram [EEG]).  If you're interested in other data (e.g., calcium imaging, or BOLD), you may still find the examples indirectly useful (for example, demonstrations of how to compute and interpret a power spectrum of a signal).

This repository was created by Emily Schlafly and Mark Kramer, with important contributions from Dr. Anthea Cheung.

**Thank you to:**

- <a href="https://mitpress.mit.edu" rel="external" target="_blank">MIT Press</a> for publishing the MATLAB version of this material.
- <a href="https://reporter.nih.gov/project-details/9309027" rel="external" target="_blank">NIH NIGMS R25GM114827</a> and <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=1451384" rel="external" target="_blank">NSF DMS #1451384</a> for support.

---

## Quick start to learning Python for neural data analysis:

- Visit the  <a href="https://mark-kramer.github.io/Case-Studies-Python/" rel="external" target="_blank">web-formatted version of the book</a>.
- Watch this <a href="https://youtu.be/Oj9e2bB3BfI"  rel="external" target="_blank">2 minute video</a>.
- Read and interact with the Python code in your web browser.

---

## Slow start to learning Python for neural data analysis:

There are multiple ways to interact with these notebooks.

- **Simple**: Visit the <a href="https://mark-kramer.github.io/Case-Studies-Python/intro.html" rel="external" target="_blank">web-formatted version of the notebooks</a>.

- **Intermediate**:  Open a notebook in <a href="https://mybinder.org/v2/gh/Mark-Kramer/Case-Studies-Python.git/binder?urlpath=lab" rel="external" target="_blank">Binder</a> and interact with the notebooks through a JupyterHub server. Binder provides an easy interface to interact with this material; read about it in <a href="https://elifesciences.org/labs/a7d53a88/toward-publishing-reproducible-computation-with-binder" rel="external" target="_blank">eLife</a>.

- **Advanced**: <a href="https://github.com/Mark-Kramer/Case-Studies-Python/archive/binder.zip" rel="external" target="_blank">Download</a> the notebooks and run them locally (i.e. on your own computer) in <a href="https://jupyter.org/">Jupyter</a>. You'll then be able to read, edit and execute the Python code directly in your browser and you can save any changes you make or notes that you want to record. You will need to [install Python](#install-python) and we recommend that you [configure](#configure-python) a Python environment as well.

---
<a id="install-python"></a>
## Install Python

We assume you have installed Python and can get it running on your computer. Some useful references to do so include,

- <a href="https://www.python.org/" rel="external" target="_blank">Python.org</a>

- <a href="https://docs.conda.io/en/latest/miniconda.html" rel="external" target="_blank">Miniconda</a>

- <a href="https://www.anaconda.com/products/individual" rel="external" target="_blank">Anaconda</a>

If this is your first time working with Python, using <a href="https://www.anaconda.com/products/individual" rel="external" target="_blank">Anaconda</a> is probably a good choice. It provides a simple, graphical interface to start <a href="https://jupyter.org/" rel="external" target="_blank">Jupyter</a>.

--- 

<a id="configure-python"></a>
## Configure Python

If you have never used the terminal before, consider using <a href="https://docs.anaconda.com/anaconda/navigator/" rel="external" target="_blank">Anaconda Navigator</a>, Anaconda's desktop graphical user interface (GUI).

Once you have installed Anaconda or Miniconda, we recommend setting up an <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html" rel="external" target="_blank">environment</a> to run the notebooks. If you downloaded the <a href="https://github.com/Mark-Kramer/Case-Studies-Python/archive/student.zip" rel="external" target="_blank">repository from Github</a>, then you can run the commands below in your terminal to configure your local environment to match the Binder environment. If you have never used the terminal before, consider using <a href="https://docs.anaconda.com/anaconda/navigator/" rel="external" target="_blank">Anaconda Navigator</a>, Anaconda's desktop graphical user interface (GUI). The environment file we use on Binder is located in the `binder` folder.

```
# create environment <case-studies>
conda env create --file environment.yml
conda activate case-studies  # activate environment <case-studies>
make config  # configure jupyter in environment
```

This will ensure that you have all the packages needed to run the notebooks. Note that you can use `make clean` to remove the changes made during `make config`. 

Finally, whenever you are ready to work with the notebooks, activate your environment and start Jupyter:

```
conda activate case-studies  # activate python environment
jupyter notebook  # start jupyter in the current location
```

If you prefer, you can use `jupyter lab` instead of `jupyter notebook`.

---

## Contributions
We very much appreciate your contributions to this material. Contribitions may include:
- Error corrections
- Suggestions
- New material to include (please start from this <a href="https://github.com/Mark-Kramer/Case-Studies-Python/blob/master/template.ipynb" rel="external" target="_blank">template</a>).

There are two ways to suggest a contribution:

- **Simple**: Visit <a href="https://github.com/Mark-Kramer/Case-Studies-Python/" rel="external" target="_blank">Case Studies Python</a>, locate the file to edit, and follow <a href="https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-another-users-repository" rel="external" target="_blank">these instructions</a>.

- **Advanced**: Fork <a href="https://github.com/Mark-Kramer/Case-Studies-Python/" rel="external" target="_blank">Case Studies Python</a> and <a href="https://jarv.is/notes/how-to-pull-request-fork-github/" rel="external" target="_blank">submit a pull request</a>


