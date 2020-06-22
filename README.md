# Case-Studies-Python

## Quick start to learning Python for neural data analysis:

- Visit the <a href="https://mark-kramer.github.io/Case-Studies-Python/intro.html" rel="external">web-formatted version of the book</a>.
- Read and interact with the Python code in your web browser.

## Slow start to learning Python for neural data analysis:

- See [below](#started)

----
This repository is a companion to the textbook <a href="https://mitpress.mit.edu/books/case-studies-neural-data-analysis" rel="external">Case Studies in Neural Data Analysis</a>, by Mark Kramer and Uri Eden. That textbook used MATLAB to analyze examples of neuronal data. The material here is  similar, except that we use Python.

The intended audience is the *practicing neuroscientist* - e.g., the students, researchers, and clinicians collecting neuronal data in the hospital or lab.  The material can get pretty math-heavy, but we've tried to outline the main concepts as directly as possible, with hands-on implementations of all concepts.  We focus on only two main types of data: spike trains and electric fields (such as the local field potential [LFP], or electroencephalogram [EEG]).  If you're interested in other data (e.g., calcium imaging, or BOLD), you may still find the examples indirectly useful (for example, demonstrations of how to compute and interpret a power spectrum of a signal).

This repository was created by Emily Schlafly and Mark Kramer, with important contributions from Dr. Anthea Cheung.

**Thank you to:**

- <a href="https://projectreporter.nih.gov/project_info_description.cfm?aid=9043612&icde=0" rel="external">NIH NIGMS R25GM114827</a> and <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=1451384" rel="external">NSF DMS #1451384</a> for support.
- <a href="https://mitpress.mit.edu/books/case-studies-neural-data-analysis" rel="external">MIT Press</a> for publishing the original version of this material.

---
<a id="started"></a>
## Getting Started

There are multiple ways to interact with these notebooks.

- **Simple**: Visit the <a href="https://mark-kramer.github.io/Case-Studies-Python/intro.html" rel="external">web-formatted version of the notebooks</a>.

- **Intermediate**  Open a notebook in <a href="https://mybinder.org/v2/gh/Mark-Kramer/Case-Studies-Python.git/master">Binder</a> and interact with the notebooks through a JupyterHub server. Binder provides an easy interface to interact with this material; read about it in <a href="https://elifesciences.org/labs/a7d53a88/toward-publishing-reproducible-computation-with-binder" rel="external">eLife</a>.

- **Advanced**: <a href="https://github.com/Mark-Kramer/Case-Studies-Python/archive/student.zip" rel="external">Download</a> the notebooks and run them locally (i.e. on your own computer) in <a href="https://jupyter.org/">Jupyter</a>. You'll then be able to read, edit and execute the Python code directly in your browser and you can save any changes you make or notes that you want to record. You will need to [install Python](#install-python) and we recommend that you [configure](#configure-python) a Python environment as well.

---
<a id="install-python"></a>
## Install Python

We assume you have installed Python and can get it running on your computer.  Some useful references to do so include,

- <a href="https://www.python.org/" rel="external">Python.org</a>

- <a href="https://docs.conda.io/en/latest/miniconda.html" rel="external">Miniconda</a>

- <a href="https://www.anaconda.com/products/individual" rel="external">Anaconda</a>

If this is your first time working with Python, using <a href="https://www.anaconda.com/products/individual" rel="external">Anaconda</a> is probably a good choice. It provides a simple, graphical interface to start <a href="https://jupyter.org/" rel="external">Jupyter</a>.

--- 

<a id="configure-python"></a>
## Configure Python

Once you have installed Anaconda or Miniconda, we recommend setting up an <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html" rel="external">environment</a> to run the notebooks. If you downloaded the <a href="https://github.com/Mark-Kramer/Case-Studies-Python/archive/student.zip" rel="external">*student* branch</a> of the repository, then you should see a file called `environment.yml`. Type the following code into your terminal to create and activate an environment called `case-studies`. 

```
conda create env --file environment.yml
conda activate case-studies
```

This will ensure that you have all the packages needed to run the notebooks. If you run a notebook at this point, you may notice a few cosmetic differences. The code will still work without the next step, but if you want to match the formatting in the published version, you can run the following in your terminal:

```
mkdir -p ~/.jupyter/custom
cp -i _config/_static/custom.* ~/.jupyter/custom/
mkdir -p ~/.ipython/profile_default/
cp -ir _config/startup ~/.ipython/profile_default/
```

---

## Contributions
We very much appreciate your contributions to this material. Contribitions may include:
- Error corrections
- Suggestions
- New material to include

There are two ways to suggest a contribution:

- **Simple**: Visit <a href="https://github.com/Mark-Kramer/Case-Studies-Python/" rel="external">Case Studies Python</a>, locate the file to edit, and follow <a href="https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-another-users-repository" rel="external">these instructions</a>.

- **Advanced**: Fork <a href="https://github.com/Mark-Kramer/Case-Studies-Python/" rel="external">Case Studies Python</a> and <a href="https://jarv.is/notes/how-to-pull-request-fork-github/" rel="external">submit a pull request</a>


