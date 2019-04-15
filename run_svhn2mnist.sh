set -ex

GPUID=3

# svhn2mnist
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
    trained_models/svhn2mnist_source.pt \
    --adapt-setting svhn2mnist \
    --name ${MODEL} \
    --src-root datasets/svhn/train \
    --src-list sourcefiles/svhn_train.txt \
    --tar-root datasets/mnist/train \
    --tar-list sourcefiles/mnist_train.txt
done

# ## test on target
# CUDA_VISIBLE_DEVICES=0 python test_model2.py \
# trained_models/svhn2mnist_${MODEL}.pt \
# --adapt-setting svhn2mnist \
# --tar-root datasets/mnist/train \
# --tar-list sourcefiles/mnist_train.txt