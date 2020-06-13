#!/bin/bash

cp -r _config/* _book/
cp -r content/* _book/
ls content/*ipynb > file_list.txt
while IFS=["/","."] read -r content file ext; do rm _book/$file.md ; done < file_list.txt
jb build _book
tail -n +2 _book/_toc.yml > _book/_toc1.yml
jb build --toc _book/_toc1.yml _book 
rm file_list.txt
cp -r content/imgs _book/_build/html/

