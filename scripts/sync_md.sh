#!/bin/bash

for f in $(ls content/*.md); do jupytext --sync $f; done

