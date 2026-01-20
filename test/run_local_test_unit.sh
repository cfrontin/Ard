#!/bin/bash
case "$1" in
  ard|flowers)
    CASES_TO_RUN=$1
    ;;
  ""|all)
    CASES_TO_RUN="ard flowers"
    ;;
  *)
    echo "$1 NOT FOUND IN VALID CASES!"
    ;;
esac

for CASE in $CASES_TO_RUN; do
  if python -c "import optiwindnet" 2>/dev/null || [[ "$CASE" != "ard" ]] ; then
    pytest --cov=$CASE --cov-report=html test/$CASE/unit
  else
    pytest --cov=$CASE --cov-report=html test/$CASE/unit --cov-config=.coveragerc_no_optiwindnet
  fi
done
