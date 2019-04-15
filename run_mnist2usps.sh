set -ex

GPUID=2

# mnist2usps
## train on source

## test on target
CUDA_VISIBLE_DEVICES=${GPUID} python test_model2.py \
trained_models/svhn2mnist_source.pt \
--adapt-setting svhn2mnist \
--tar-root datasets/mnist/train \
--tar-list sourcefiles/mnist_train.txt

for MODEL in revgrad adda wdgrl
do
    CUDA_VISIBLE_DEVICES=${GPUID} python ${MODEL}2.py \
    trained_models/mnist2usps_source.pt \
    --adapt-setting mnist2usps \
    --name ${MODEL} \
    --src-root datasets/mnist/train \
    --src-list sourcefiles/mnist_train.txt \
    --tar-root datasets/usps \
    --tar-list sourcefiles/usps_train.txt
done

# ## test on target
# CUDA_VISIBLE_DEVICES=0 python test_model2.py \
# trained_models/mnist2usps_revgrad.pt \
# --adapt-setting mnist2usps \
# --tar-root datasets/usps \
# --tar-list sourcefiles/usps_train.txt
