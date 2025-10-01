#!/bin/bash

python ard_xdsm.py
magick -quality 100 -density 300 -colorspace sRGB ard_xdsm.pdf -flatten ard_xdsm.png

