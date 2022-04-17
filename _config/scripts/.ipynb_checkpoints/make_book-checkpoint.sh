#!/bin/bash

mkdir ignore
cp _config/{_toc,_config}.yml .
cp -r _config/_static .
ls *ipynb | while IFS=["/","."] read -r file ext; do mv $file.md ignore/ ; done 
jb build ./
if test -f intro.md ; then 
    echo "<meta http-equiv=\"Refresh\" content=\"0; url=intro.html\" />" > _build/html/index.html 
fi
cp -r imgs _build/html/
mv ignore/*md . && rm -r ignore
rm _toc.yml _config.yml
rm -r _static
