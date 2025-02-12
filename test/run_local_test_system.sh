#!/bin/bash
if python -c "import interarray" 2>/dev/null ; then
  pytest --cov=ard --cov-report=html test/system
else
  pytest --cov=ard --cov-report=html test/system --cov-config=.coveragerc_no_interarray
fi

rm -rf test/system/layout/problem*_out

#
