#!/bin/bash
if [ -z "$1" ];then
   echo ""
   echo "No Notebook file given usage:"
   echo "$(basename $0) jupyter_notebook_file.ipynb [output_filename] [browser]"
   echo ""
   exit 257
fi
infile=$1
outfile=$1
if [ "$2" ];then
   outfile=$2
fi

export BROWSER=chromium-browser
jupyter nbconvert $1 --to slides --post serve \
   --SlidesExporter.reveal_theme=reveal \
   --SlidesExporter.reveal_scroll=True \
   --SlidesExporter.reveal_transition=none
