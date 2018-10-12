#!/bin/bash

input_file=$1
output_file=$2
echo $input_file $output_file
train=0

if [ $train -eq 1 ]
then
    python train.py 
fi
python test.py $input_file $output_file