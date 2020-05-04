import os
import torch


def mkdir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(ROOT_DIR, 'ISBI2012')
CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
TRAIN_LOG_DIR = os.path.join(LOG_DIR, 'train')
VAL_LOG_DIR = os.path.join(LOG_DIR, 'val')

IMGS_DIR = os.path.join(DATA_DIR, 'imgs')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

TRAIN_IMGS_DIR = os.path.join(IMGS_DIR, 'train')
VAL_IMGS_DIR = os.path.join(IMGS_DIR, 'val')
TEST_IMGS_DIR = os.path.join(IMGS_DIR, 'test')

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, 'train')
VAL_LABELS_DIR = os.path.join(LABELS_DIR, 'val')
TEST_LABELS_DIR = os.path.join(LABELS_DIR, 'test')

mkdir(
    CKPT_DIR, LOG_DIR, TRAIN_LOG_DIR, VAL_LOG_DIR,
    IMGS_DIR, LABELS_DIR, TRAIN_IMGS_DIR, VAL_IMGS_DIR, TEST_IMGS_DIR,
    TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR,
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
class Config:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 2
    NUM_EPOCHS = 100
