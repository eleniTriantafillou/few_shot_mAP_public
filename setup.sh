OMNIGLOT_PATH="/ais/gobi4/eleni/omniglot"
MINI_IMAGENET_PATH="/ais/gobi4/eleni/miniImagenet/images"
OMNIGLOT_SPLITS_PATH="/ais/gobi4/eleni/few_shot_mAP/data/dataset_splits/omniglot"
MINI_IMAGENET_SPLITS_PATH="/ais/gobi4/eleni/few_shot_mAP/data/dataset_splits/mini_imagenet"

mkdir -p data/dataset_splits
ln -s $OMNIGLOT_PATH data/omniglot
ln -s $MINI_IMAGENET_PATH data/mini_imagenet
ln -s $OMNIGLOT_SPLITS_PATH data/dataset_splits/omniglot
ln -s $MINI_IMAGENET_SPLITS_PATH data/dataset_splits/mini_imagenet
