# parameters for processing the dataset
PATH_PREFIX= 'data'
DATA_PATH = 'data'
DATA_FILE = 'build_data.xlsx'
NUMBER_SHEET = 4
LIST_SHEET_NAME = ['CTDT_HTTT_2018','Ho_tro_tuyen_sinh','van_ban','Giang_vien']
OUTPUT_FILE = 'build_data.txt'
PROCESSED_PATH = PATH_PREFIX + '/processed'
CPT_PATH = PATH_PREFIX + '/checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 46

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "), 
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 264
DEC_VOCAB = 871
ENC_VOCAB = 263
DEC_VOCAB = 865
