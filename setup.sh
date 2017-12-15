OMNIGLOT_PATH="path_to_omniglot"
MINI_IMAGENET_PATH="path_to_imagenet"
MINI_IMAGENET_SPLITS_PATH="path_to_mini_imagenet_splits"

ln -s $OMNIGLOT_PATH data/omniglot
ln -s $MINI_IMAGENET_PATH data/mini_imagenet
ln -s $MINI_IMAGENET_SPLITS_PATH data/dataset_splits/mini_imagenet
