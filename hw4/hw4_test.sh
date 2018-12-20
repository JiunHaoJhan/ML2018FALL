test_path=$1
dict_path=$2
output_path=$3

wget 'https://www.dropbox.com/s/1vnrv1mys84s4ie/weights-improvement-003-0.7602.hdf5'
wget 'https://www.dropbox.com/s/zco3mt4tofszax2/weights-improvement-002-0.7603.hdf5'
wget 'https://www.dropbox.com/s/qfwf7f88pqdiah1/weights-improvement-003-0.7641.hdf5'

python3 hw4.py --test --test_path $test_path --dict_path $dict_path --output_path $output_path