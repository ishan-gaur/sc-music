from pathlib import Path

CELLMAPS = Path("/home/ishang/miniconda3/envs/music/lib/python3.9/site-packages/")
REPO = Path.cwd()
INPUT_FOLDER = REPO / "0.U2OS_ref"
# INPUT_FOLDER = REPO / "0.example"

SC_EMBEDDINGS_FOLDER = REPO / "1.sc_embeddings"

SC_EMBEDDINGS_FILE = SC_EMBEDDINGS_FOLDER / "u2os_embeddings.npy"
SC_LABELS_FILE = SC_EMBEDDINGS_FOLDER / "u2os_labels.csv"

WALK_LENGTH, N_WALKS = 80, 10
P, Q = 2, 1
N_WORKERS = 10

COEMBEDDING_LATENT_DIM = 128

HIERARCHY_NAME, ORG_NAME, PROJ_NAME = None, None, None
CONTAINMENT_THRESHOLD = 0.75
JACCARD_THRESHOLD = 0.9
MIN_DIFF = 1
MIN_SYSTEM_SIZE = 4
PPI_CUTOFFS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                0.007, 0.008, 0.009, 0.01, 0.02, 0.03,
                0.04, 0.05, 0.10]

PASS_NODE2VEC = False
PASS_COEMBEDDING = False
PASS_HIERARCHY = False

SKIP_SC = True
SKIP_SETUP = False
SKIP_NODE2VEC = False
SKIP_COEMBEDDING = False
SKIP_HIERARCHY = False