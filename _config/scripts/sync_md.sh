#!/bin/bash

jupytext --update-metadata '{"jupytext": {"formats": "ipynb,md:myst"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.7.7"}}' --sync $1

