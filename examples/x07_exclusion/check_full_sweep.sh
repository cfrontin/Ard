#!/bin/bash
echo 'Number of logs should match number of optimization statuses found...'

for dname in thresh_0p*; do
    nlog=`ls $dname/*.log | wc -l`
    nstat=`grep '^Optimization' $dname/*.log | wc -l`
    echo "$dname $nlog $nstat"
done

dname='no_exclusion'
nlog=`ls $dname/*.log | wc -l`
nstat=`grep '^Optimization' $dname/*.log | wc -l`
echo "$dname $nlog $nstat"
