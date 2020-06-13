#!/bin/bash

cp -r _config/* _book/
head -n 1 _book/_toc.yml > _book/_toc0.yml
jb build --toc _book/_toc0.yml _book >> _book/_build/.build_logs
cp -r content/* _book/
ls content/*ipynb | while IFS=["/","."] read -r content file ext; do rm _book/$file.md ; done 
tail -n +2 _book/_toc.yml > _book/_toc1.yml
jb build --toc _book/_toc1.yml _book
cp -r content/imgs _book/_build/html/

