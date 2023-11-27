import os
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from cellmaps_utils import constants # Super useful file to understand what's going on
from cellmaps_ppidownloader import cellmaps_ppidownloadercmd
from cellmaps_ppi_embedding import cellmaps_ppi_embeddingcmd
from config import CELLMAPS, REPO, INPUT_FOLDER
from config import SC_EMBEDDINGS_FOLDER, SC_EMBEDDINGS_FILE, SC_LABELS_FILE

def make_header(x):
    line_delim = "=" * 80
    return f"{line_delim}\n{x}\n{line_delim}"

M, D, Y = time.localtime()[:3]
pid = os.getpid()
print(make_header(f"Runner Process: {pid}"))
OUTPUT_FOLDER = REPO / f"run_{pid}_{M}_{D}_{Y}"
os.mkdir(OUTPUT_FOLDER)

# PPI Bioplex Graph Download
OUTDIR_PPI_DOWNLOAD = REPO / constants.PPI_DOWNLOAD_STEP_DIR
provenance = INPUT_FOLDER / 'provenance.json'
edgelist = INPUT_FOLDER / 'edgelist.tsv'
baitlist = INPUT_FOLDER / 'baitlist.tsv'

ppi_download_args = [
    "dummy_program_name",
    str(OUTDIR_PPI_DOWNLOAD),
    "--edgelist", str(edgelist),
    "--baitlist", str(baitlist),
    "--provenance", str(provenance),
    "--verbose"
]
print(make_header(f"Downloading PPI Data from Bioplex to {str(OUTDIR_PPI_DOWNLOAD)}"))
print(f"With args:\n{ppi_download_args}")
if not OUTDIR_PPI_DOWNLOAD.exists():
    cellmaps_ppidownloadercmd.main(ppi_download_args)
else:
    print("Skipping PPI download because output directory already exists")

# SC Embedding Data
if not SC_EMBEDDINGS_FOLDER.exists():
    raise FileNotFoundError(f"SC Embeddings folder {str(SC_EMBEDDINGS_FOLDER)} does not exist")

u2os_embeddings = np.load(SC_EMBEDDINGS_FILE)
u2os_labels = pd.read_csv(SC_LABELS_FILE)

# Select Genes
print(make_header("Selecting Genes and Filtering PPI and SC Embedding Data"))

u2os_hpa_genes = set(u2os_labels["gene_names"].unique())
u2os_bioplex = pd.read_csv(INPUT_FOLDER / "edgelist.tsv", sep="\t")
u2os_bioplex_genes = set(u2os_bioplex["Symbol1"].unique()) | set(u2os_bioplex["Symbol2"].unique())
genes = u2os_hpa_genes & u2os_bioplex_genes
gene_list_file = OUTPUT_FOLDER / "gene_list.txt"
with open(gene_list_file, "w") as f:
    f.write("\n".join(genes))
print(f"Selected {len(genes)} genes.")

# Filter PPI data for selected genes
# cellmaps_ppi_embeddingcmd.main() seems to depends only on ppi_edgelist.tsv,
# but we will also filter ppi_gene_node_attributes.tsv to be safe
ppi_edgelist_file = OUTDIR_PPI_DOWNLOAD / constants.PPI_EDGELIST_FILE
ppi_gene_node_attributes_file = OUTDIR_PPI_DOWNLOAD / constants.PPI_GENE_NODE_ATTR_FILE
ppi_edgelist = pd.read_csv(ppi_edgelist_file, sep="\t")
ppi_gene_node_attributes = pd.read_csv(ppi_gene_node_attributes_file, sep="\t")
col1, col2 = constants.PPI_EDGELIST_GENEA_COL, constants.PPI_EDGELIST_GENEB_COL
ppi_edgelist = ppi_edgelist[
    ppi_edgelist[col1].isin(genes) & ppi_edgelist[col2].isin(genes)
]
if "name" not in constants.PPI_GENE_NODE_COLS:
    raise ValueError("PPI_GENE_NODE_COLS does not contain 'name' column, cellmaps format may have changed")
ppi_gene_node_attributes = ppi_gene_node_attributes[ppi_gene_node_attributes["name"].isin(genes)]

proc_PPI_DOWNLOAD_FOLDER = OUTPUT_FOLDER / constants.PPI_DOWNLOAD_STEP_DIR
shutil.copytree(OUTDIR_PPI_DOWNLOAD, proc_PPI_DOWNLOAD_FOLDER)
ppi_edgelist.to_csv(proc_PPI_DOWNLOAD_FOLDER / constants.PPI_EDGELIST_FILE, sep="\t", index=False)
ppi_gene_node_attributes.to_csv(proc_PPI_DOWNLOAD_FOLDER / constants.PPI_GENE_NODE_ATTR_FILE, sep="\t", index=False)

# Filter SC Embeddings for selected genes
u2os_embeddings = u2os_embeddings[u2os_labels["gene_names"].isin(genes)]
u2os_labels = u2os_labels[u2os_labels["gene_names"].isin(genes)]

proc_SC_EMBEDDINGS_FOLDER = OUTPUT_FOLDER / "1.sc_embeddings"
os.mkdir(proc_SC_EMBEDDINGS_FOLDER)
np.save(proc_SC_EMBEDDINGS_FOLDER / "u2os_embeddings.npy", u2os_embeddings)
u2os_labels.to_csv(proc_SC_EMBEDDINGS_FOLDER / "u2os_labels.csv", index=False)

# Run PPI Embedding
proc_PPI_EMBEDDINGS_FOLDER = OUTPUT_FOLDER / constants.PPI_EMBEDDING_STEP_DIR
ppi_embedding_args = [
    "dummy_program_name",
    str(proc_PPI_DOWNLOAD_FOLDER),
    "--inputdir", str(proc_PPI_DOWNLOAD_FOLDER),
    "--dimensions", str(len(u2os_embeddings[0])),
    "--verbose"
]
print(make_header(f"Embedding PPI Data"))
print(f"With args:\n{ppi_embedding_args}")
cellmaps_ppi_embeddingcmd.main(ppi_embedding_args)