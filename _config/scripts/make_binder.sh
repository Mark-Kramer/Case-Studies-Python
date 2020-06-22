#!/bin/bash

mkdir -p tmp/binder
if test -d _binder ; then cp -r _binder/* tmp/binder/ ; fi
while IFS=[" "] read -r a b fname ; do 
    if [ "$fname" ] && test -f "$fname" ; then 
        cp $fname tmp/ ; 
    fi; 
done < _config/_toc.yml

ln -s $PWD/{matfiles,imgs} tmp/
cp *.py tmp
cp README.md tmp
mkdir tmp/_config
cp -r _config/{_static,startup} tmp/_config

ghp-import -p -f -l -b binder tmp/
rm -r tmp


