#!/bin/bash

prefix=nonuniform

for log in `ls ${prefix}*.log`; do
    stat=`grep -A 1 '^Optimization' $log | tr '\n' ' ' | sed 's/--*//g'`
    echo "$log $stat"
done
