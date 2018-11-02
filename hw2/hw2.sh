#!/bin/bash

train=0

if [ $train -eq 1 ]
then
    python3 generative.py --train --train_x_path $1 --train_y_path $2 --test_x_path $3 --output_path $4
fi
python3 generative.py --test --train_x_path $1 --train_y_path $2 --test_x_path $3 --output_path $4