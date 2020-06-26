#!/bin/bash

mkdir -p tmp
cp -r binder tmp/
mv tmp/binder/Makefile tmp
while IFS=[" "] read -r a b fname ; do 
    if [ "$fname" ] && test -f "$fname" ; then 
        cp $fname tmp/ ; 
    fi; 
done < _config/_toc.yml

ln -s $PWD/{matfiles,imgs} tmp/
cp *.py tmp
cp README.md tmp
cp _config/{_static,startup}/* tmp/binder

ghp-import -p -f -l -b binder tmp/
rm -r tmp


