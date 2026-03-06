IMAGE_SIZE = (256, 256)
TRAIN_DATA_PATH = "./data/LSUI"
TEST_DATA_PATH = "./data/SUIM"
VAL_DATA_PATH = "./data/UIEB"
SAVE_DIR = "./checkpoints"
EARLY_STOPPING = 3
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