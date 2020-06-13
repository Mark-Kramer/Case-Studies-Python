#!/bin/bash

cp -r _config/* _book/
cp -r content/* _book/
ls content/*ipynb | while IFS=["/","."] read -r content file ext; do rm _book/$file.md ; done 
jb build _book
tail -n +2 _book/_toc.yml > _book/_toc1.yml
jb build --toc _book/_toc1.yml _book 
cp -r content/imgs _book/_build/html/

