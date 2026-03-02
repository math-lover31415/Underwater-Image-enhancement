IMAGE_SIZE = (256, 256)
TRAIN_DATA_PATH = "./data/train"
VAL_DATA_PATH = "./data/val"
SAVE_DIR = "./checkpoints"
class SupervisedTrainingParameters:
    BATCH_SIZE = 40
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 20


class UnsupervisedPretrainingParameters:
    BATCH_SIZE = 40
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 3


class KnowledgeTransfer:
    BATCH_SIZE = 40
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5