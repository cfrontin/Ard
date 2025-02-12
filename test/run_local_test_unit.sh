#!/bin/bash
if python -c "import interarray" 2>/dev/null ; then
  pytest --cov=ard,gpAEP --cov-report=html test/unit
else
  pytest --cov=ard,gpAEP --cov-report=html test/unit --cov-config=.coveragerc_no_interarray
fi

rm -rf test/unit/layout/problem*_out

#
