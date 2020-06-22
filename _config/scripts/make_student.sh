#!/bin/bash

mkdir tmp
if test -d student ; then cp -r student/* tmp ; fi
while IFS=[" "] read -r a b fname ; do 
    if [ "$fname" ] && test -f "$fname" ; then 
        cp $fname tmp/ ; 
    fi; 
done < _config/_toc.yml

ln -s $PWD/{matfiles,imgs} tmp/
cp *.py tmp
cp README.md tmp
cp -r _config tmp
mv tmp/_config/Makefile.student tmp/Makefile
rm tmp/_config/_*yml

ghp-import -p -f -l -b student tmp/
rm -r tmp


