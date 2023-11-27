from pathlib import Path

CELLMAPS = Path("/home/ishang/miniconda3/envs/music/lib/python3.9/site-packages/")
REPO = Path.cwd()
INPUT_FOLDER = REPO / "0.U2OS_ref"

SC_EMBEDDINGS_FOLDER = REPO / "1.sc_embeddings"

SC_EMBEDDINGS_FILE = SC_EMBEDDINGS_FOLDER / "u2os_embeddings.npy"
SC_LABELS_FILE = SC_EMBEDDINGS_FOLDER / "u2os_labels.csv"