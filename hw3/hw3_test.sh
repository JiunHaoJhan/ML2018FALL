
test_path=$1
output_path=$2

wget 'https://www.dropbox.com/s/h4fujwy3ceiw6wy/weights-improvement-207-0.7103.hdf5'
wget 'https://www.dropbox.com/s/kfng5stt02susxh/weights-improvement-206-0.7089.hdf5'
wget 'https://www.dropbox.com/s/kf9nf8fc2e4rzfq/weights-improvement-252-0.7061.hdf5'
wget 'https://www.dropbox.com/s/9xmbtj4a3h3q74h/weights-improvement-98-0.7047.hdf5'

python3 CNN.py --test --test_path $test_path --output_path $output_path