IMAGE_SIZE = (256, 256)
TRAIN_DATA_PATH = "./data/LSUI"
VAL_DATA_PATH = "./data/SUIM"
TEST_DATA_PATH = "./data/UIEB"
MODEL_PATH = "checkpoints/weights_schedule.pth"
SAVE_DIR = "./checkpoints"
EARLY_STOPPING = 8

class SupervisedTrainingParameters:
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 35

class UnsupervisedPretrainingParameters:
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 8

class KnowledgeTransfer:
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 10
