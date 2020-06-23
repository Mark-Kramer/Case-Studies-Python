#!/bin/bash


if [ -z "$NB_PYTHON_PREFIX" ] ; then NB_PYTHON_PREFIX=$CONDA_PREFIX ; fi
mkdir -p ~/.jupyter/custom
cat _config/_static/custom.css >> ~/.jupyter/custom/custom.css
cat _config/_static/custom.js >> ~/.jupyter/custom/custom.js
cat _config/_static/custom.css >> $NB_PYTHON_PREFIX/share/jupyter/lab/themes/\@jupyterlab/theme-light-extension/index.css

mkdir -p ~/.ipython/profile_default/startup
cp _config/startup/* ~/.ipython/profile_default/startup/
