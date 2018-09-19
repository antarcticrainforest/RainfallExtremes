#!/bin/bash
export BROWSER=chromium-browser
jupyter nbconvert $1 --to slides --post serve \
   --SlidesExporter.reveal_theme=default \
   --SlidesExporter.reveal_scroll=True \
   --SlidesExporter.reveal_transition=none
