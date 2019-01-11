train_x_path=$1
train_y_path=$2
test_path=$3
sample_path=$4
output_path=$5

cd ..
wget 'https://www.dropbox.com/s/0hjw4wvh5otixt6/ResNet34_256_1_multi_0.489.h5'
mv ResNet34_256_1_multi_0.489.h5 models/

python3 src/finaltest.py $train_x_path $train_y_path $test_path $sample_path $output_path
cd src
