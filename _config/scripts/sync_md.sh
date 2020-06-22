#!/bin/bash

while IFS=["."," "] read -r a b file ext ; do 
    if [ "$file" ] && test -f "$file.md" ; then 
        echo "$file.md"
        jupytext --update-metadata '{"jupytext": {"formats": "ipynb,md:myst"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.7.7"}}' --sync $file.md
    fi
done < _toc.yml

