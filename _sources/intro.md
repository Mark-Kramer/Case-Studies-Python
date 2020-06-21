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

- **Advanced**: Run the notebooks locally on your computer in <a href="https://jupyter.org/">Jupyter</a>. You'll then be able to read, edit and execute the Python code directly in your browser and you can save any changes you make or notes that you want to record. To access and download a notebook. You will need to [install Python](#install-python) and we recommend [configure Python](#configure-python).

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

Once you have installed Anaconda or Miniconda, we recommend setting up an <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html" rel="external">environment</a> to run the notebooks. Type the following code into your terminal to create and activate an environment called `case-studies`. 

```
conda create env --file environment.yml
conda activate case-studies
```

This will ensure that you have all the packages needed to run the notebooks. If you run a notebook at this point, you may notice that when you generate plots, they look different. If you want to match the formatting in the published version, you can run the following in your terminal:

```
mkdir -p ~/.jupyter/custom
cp -i _static/custom.* ~/.jupyter/custom/
mkdir -p ~/.ipython/profile_default/
cp -ir _config/startup ~/.ipython/profile_default/
```

Note that if you already have files in these directories, you will be prompted to overwrite them. In this case, you may prefer to append the contents to the end of the existing files. You can do this with the following:

```
for f in $(ls _static/custom.*); do echo $(_static/$f) >> ~/.jupyter/custom/$f; done
for f in $(ls _config/startup); do echo $(_config/startup/$f) >> ~/.ipython/profile_default/startup/$f; done
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


