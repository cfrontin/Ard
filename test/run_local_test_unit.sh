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

IDX_CASE=0
for CASE in $CASES_TO_RUN; do
  FLAGS="--cov-report=html"
  if [[ IDX_CASE -gt 0 ]] ; then
    FLAGS="$FLAGS --cov-append"
  fi
  if [[ "$CASE" == "ard" ]] && ! python -c "import optiwindnet" 2>/dev/null ; then
    FLAGS="$FLAGS --cov-config=.coveragerc_no_optiwindnet"
  fi
  pytest --cov=$CASE ${FLAGS} test/$CASE/unit
  ((IDX_CASE++))
done
