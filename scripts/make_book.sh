#!/bin/bash

cp -r _config/* _book/
cp -r content/* _book/
ls content/*ipynb > file_list.txt
while IFS=["/","."] read -r content file ext; do rm _book/$file.md ; done < file_list.txt
jb build _book
rm file_list.txt
cp -r content/imgs _book/_build/html/

