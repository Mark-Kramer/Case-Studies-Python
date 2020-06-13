#!/bin/bash

ls _build/*/*.html > file_list.txt

while IFS="/" read -r build ch file; do echo "<meta http-equiv=\"Refresh\" content=\"0; url=$file\" />" > $build/$ch/index.html; done < file_list.txt

rm file_list.txt

