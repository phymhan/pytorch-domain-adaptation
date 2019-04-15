set -ex

GPUID=2

# mnist2usps
## train on source

## test on target

for MODEL in wdgrl
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
