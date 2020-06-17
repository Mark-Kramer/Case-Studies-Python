#!/bin/bash

ls content/*/*.ipynb > file_list.txt

while IFS=["/","."] read -r build ch file ext; do echo "<meta http-equiv=\"Refresh\" content=\"0; url=$file.html\" />" > $build/$ch/index.html; done < file_list.txt

rm file_list.txt

