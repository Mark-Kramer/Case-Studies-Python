#!/bin/bash

cp -r _config/* _book/
cp -r content/* _book/
ls content/*ipynb | while IFS=["/","."] read -r content file ext; do rm _book/$file.md ; done 
jb build _book
if test -f _book/intro.md ; then 
    echo "<meta http-equiv=\"Refresh\" content=\"0; url=intro.html\" />" > _book/_build/html/index.html 
    rm _book/intro.md
fi
cp -r content/imgs _book/_build/html/

