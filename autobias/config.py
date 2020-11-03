from os.path import join, expanduser, dirname

WEIGHTS_NAME = 'state.pth'
BEST_WEIGHTS_NAME = 'best-state.pth'

DATA_DIR = join(dirname(dirname(__file__)), "data")

MNLI_BIAS_CACHE = join(DATA_DIR, "mnli-bias")
IMAGENET_BIAS_CACHE = join(DATA_DIR, "imagenet-bias")
TRANSFORMER_CACHE_DIR = None
TORCHVISION_CACHE_DIR = join(DATA_DIR, "torch-vision-cache")

GLUE_DATA = join(DATA_DIR, "glue_data")

WORD_VEC_DIR = join(DATA_DIR, "word-vectors")

IMAGENET_ANIMALS = join(DATA_DIR, "imagenet-restricted")

# This needs to be set to train on ImageNet Animals
IMAGENET_HOME = None

DUAL_TOKENIZED_CACHE = join(DATA_DIR, "tokenized-cache")

HANS = join(DATA_DIR, "hans", "heuristics_evaluation_set.txt")

# Set to None to disable caching
IMAGE_LOCAL_CACHE_RESIZED = join(DATA_DIR, "imagenet-resized-cache")
