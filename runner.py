import os
import config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICES

import argparse

import time
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from warnings import warn

from cellmaps_ppidownloader import cellmaps_ppidownloadercmd
from cellmaps_ppi_embedding import cellmaps_ppi_embeddingcmd
from cellmaps_coembedding import cellmaps_coembeddingcmd
from cellmaps_generate_hierarchy import cellmaps_generate_hierarchycmd

from cellmaps_utils import constants # Super useful file to understand what's going on

# ================================
# ========= RUNNER SETUP =========
# ================================

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", help="Checkpoint to resume from, give the folder name")
parser.add_argument("-s", "--stage", help="Stage to resume from, 'coembedding' or 'hierarchy'")
args = parser.parse_args()

if args.stage is not None:
    if args.checkpoint is None:
        raise ValueError("Must specify checkpoint to resume from")
    if args.stage not in ["coembedding", "hierarchy"]:
        raise ValueError("Invalid stage to resume from")

if args.checkpoint is not None:
    print(f"Resuming from checkpoint {args.checkpoint}")
    _, pid, M, D, Y = args.checkpoint.split("_")
    OUTPUT_FOLDER = config.REPO / args.checkpoint
    if not OUTPUT_FOLDER.exists():
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} does not exist")
    if args.stage is None:
        raise ValueError("Must specify stage to resume from")
    elif args.stage == "coembedding":
        print("Resuming from coembedding stage")
        config.SKIP_SETUP = True
        config.SKIP_NODE2VEC = True
    elif args.stage == "hierarchy":
        print("Resuming from hierarchy stage")
        config.SKIP_SETUP = True
        config.SKIP_NODE2VEC = True
        config.SKIP_COEMBEDDING = True
    else:
        raise ValueError("Invalid stage to resume from")

if args.stage is not None and args.checkpoint is None:
    raise ValueError("Must specify checkpoint to resume from")

def make_header(x):
    line_delim = "=" * 80
    return f"{line_delim}\n{x}\n{line_delim}"

def runner_proc_setup():
    M, D, Y = time.localtime()[:3]
    pid = os.getpid()
    print(make_header(f"Runner Process: {pid}"))
    OUTPUT_FOLDER = config.REPO / f"run_{pid}_{M}_{D}_{Y}"
    os.mkdir(OUTPUT_FOLDER)
    return OUTPUT_FOLDER, pid

if not config.SKIP_SETUP:
    OUTPUT_FOLDER, pid = runner_proc_setup()

# ================================================
# ========= Collecting Prerequisite Data =========
# ================================================

# PPI Bioplex Graph Download
OUTDIR_PPI_DOWNLOAD = config.REPO / constants.PPI_DOWNLOAD_STEP_DIR
provenance = config.INPUT_FOLDER / 'provenance.json'
edgelist = config.INPUT_FOLDER / 'edgelist.tsv'
baitlist = config.INPUT_FOLDER / 'baitlist.tsv'

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
if not OUTDIR_PPI_DOWNLOAD.exists() and not config.SKIP_SETUP:
    cellmaps_ppidownloadercmd.main(ppi_download_args)
elif config.SKIP_SETUP and not OUTDIR_PPI_DOWNLOAD.exists():
    raise FileNotFoundError(f"Cannot skip setup because {str(OUTDIR_PPI_DOWNLOAD)} does not exist")
else:
    print("Skipping PPI download because output directory already exists or setup was skipped")

# SC Embedding Data
print(make_header(f"Loading SC Embeddings from {str(config.SC_EMBEDDINGS_FILE)}"))
if not config.SC_EMBEDDINGS_FOLDER.exists():
    raise FileNotFoundError(f"SC Embeddings folder {str(config.SC_EMBEDDINGS_FOLDER)} does not exist")

if not config.SKIP_SETUP and not config.SKIP_SC:
    u2os_embeddings = np.load(config.SC_EMBEDDINGS_FILE)
    u2os_labels = pd.read_csv(config.SC_LABELS_FILE)
elif not config.SKIP_SETUP and config.SKIP_SC:
    u2os_embeddings_file = config.REPO / constants.IMAGE_EMBEDDING_STEP_DIR / constants.IMAGE_EMBEDDING_FILE
    # this file is a tsv in the format of:
    # header (\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15)
    # gene_name\tembedding
    u2os_embeddings_df = pd.read_csv(u2os_embeddings_file, sep="\t")
    u2os_embeddings = u2os_embeddings_df.values
    u2os_labels = u2os_embeddings_df.index.values
    u2os_labels = pd.DataFrame({"gene_names": u2os_labels})
else:
    print("Skipping SC Embedding loading because setup was skipped")

# =======================================================
# ========= Restricting Inputs to Desired Genes =========
# =======================================================

# Select Genes
if not config.SKIP_SETUP:
    print(make_header("Selecting Genes and Filtering PPI and SC Embedding Data"))

    u2os_hpa_genes = set(u2os_labels["gene_names"].unique())
    u2os_bioplex = pd.read_csv(config.INPUT_FOLDER / "edgelist.tsv", sep="\t")
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
    proc_SC_EMBEDDINGS_FOLDER = OUTPUT_FOLDER / "1.sc_embeddings"
    shutil.copytree(config.SC_EMBEDDINGS_FOLDER, proc_SC_EMBEDDINGS_FOLDER)

    indices = []
    df_gene_list = u2os_labels[u2os_labels["gene_names"].isin(genes)].values
    for gene in tqdm(genes, desc="Filtering and Selecting SC Embeddings"):
        gene_indices = np.array(u2os_labels["gene_names"].values == gene).nonzero()
        indices.extend(np.random.choice(gene_indices[0], 1))
    sc_embeddings = u2os_embeddings[indices]
    sc_gene_symbols = u2os_labels["gene_names"].values[indices]

    sc_embedding_dim = len(sc_embeddings[0])
    with open(proc_SC_EMBEDDINGS_FOLDER / constants.IMAGE_EMBEDDING_FILE, "w") as f:
        f.write("gene\t" + "\t".join([str(i) for i in range(sc_embedding_dim)]) + "\n")
        for i in range(len(sc_embeddings)):
            f.write(sc_gene_symbols[i] + "\t" + "\t".join([str(j) for j in sc_embeddings[i]]) + "\n")
else:
    proc_PPI_DOWNLOAD_FOLDER = OUTPUT_FOLDER / constants.PPI_DOWNLOAD_STEP_DIR
    proc_SC_EMBEDDINGS_FOLDER = OUTPUT_FOLDER / "1.sc_embeddings"
    sc_embedding_dim = len(np.load(config.SC_EMBEDDINGS_FILE)[0])
    if not proc_PPI_DOWNLOAD_FOLDER.exists():
        raise FileNotFoundError(f"Cannot skip setup because {str(proc_PPI_DOWNLOAD_FOLDER)} does not exist")
    if not proc_SC_EMBEDDINGS_FOLDER.exists():
        raise FileNotFoundError(f"Cannot skip setup because {str(proc_SC_EMBEDDINGS_FOLDER)} does not exist")
    print("Skipping gene selection and filtering because setup was skipped")


# ================================================
# ========= Run Embedding Generation =============
# ================================================

# Run PPI Embedding
print(make_header(f"Embedding PPI Data"))
proc_PPI_EMBEDDINGS_FOLDER = OUTPUT_FOLDER / constants.PPI_EMBEDDING_STEP_DIR
ppi_embedding_args = [
    "dummy_program_name",
    str(proc_PPI_EMBEDDINGS_FOLDER),
    "--inputdir", str(proc_PPI_DOWNLOAD_FOLDER),
    "--dimensions", str(sc_embedding_dim),
    "--walk_length", str(config.WALK_LENGTH),
    "--num_walks", str(config.N_WALKS),
    "--workers", str(min(config.N_WALKS, config.N_WORKERS)),
    "--p", str(config.P),
    "--q", str(config.Q),
    "--verbose"
]

if not config.SKIP_COEMBEDDING:
    if config.PASS_NODE2VEC:
        ppi_embedding_args.append("--fake_embedder")
        warn("Running in DEV_NODE2VEC mode, will generate fake embedding")

    if not proc_PPI_EMBEDDINGS_FOLDER.exists():
        print(f"With args:\n{ppi_embedding_args}")
        cellmaps_ppi_embeddingcmd.main(ppi_embedding_args)
    else:
        print("Skipping PPI embedding because output directory already exists")
else:
    print("Skipping PPI embedding")

# Run coembedding step
print(make_header(f"Running Coembedding Step"))
proc_COEMBEDDING_FOLDER = OUTPUT_FOLDER / constants.COEMBEDDING_STEP_DIR
coembedding_args = [
    "dummy_program_name",
    str(proc_COEMBEDDING_FOLDER),
    cellmaps_coembeddingcmd.PPI_EMBEDDINGDIR, str(proc_PPI_EMBEDDINGS_FOLDER),
    cellmaps_coembeddingcmd.IMAGE_EMBEDDINGDIR, str(proc_SC_EMBEDDINGS_FOLDER),
    '--latent_dimension', str(config.COEMBEDDING_LATENT_DIM),
    "--verbose"
]
if not config.SKIP_COEMBEDDING:
    if config.PASS_COEMBEDDING:
        coembedding_args.append("--fake_embedding")
        warn("Running in DEV_COEMBEDDING mode, will generate fake embedding")

    print(f"With args:\n{coembedding_args}")
    cellmaps_coembeddingcmd.main(coembedding_args)
else:
    print("Skipping Coembedding")

# ================================================
# ========== Run Hierarchy Generation ============
# ================================================
print(make_header(f"Running Hierarchy Generation"))
proc_HIERARCHY_FOLDER = OUTPUT_FOLDER / constants.HIERARCHY_STEP_DIR
hierarchy_gen_args = [
    "dummy_program_name",
    str(proc_HIERARCHY_FOLDER),
    cellmaps_generate_hierarchycmd.CO_EMBEDDINGDIRS, str(proc_COEMBEDDING_FOLDER),
    # '--containment_threshold', str(config.CONTAINMENT_THRESHOLD),
    # '--jaccard_threshold', str(config.JACCARD_THRESHOLD),
    # '--min_diff', str(config.MIN_DIFF),
    # '--min_system_size', str(config.MIN_SYSTEM_SIZE),
    "--verbose"
]

# hierarchy_gen_args.extend(["--ppi_cutoffs"] + [str(x) for x in config.PPI_CUTOFFS])

# if config.HIERARCHY_NAME is not None:
#     hierarchy_gen_args.extend(["--hierarchy_name", str(config.HIERARCHY_NAME)])
# if config.ORG_NAME is not None:
#     hierarchy_gen_args.extend(["--org_name", str(config.ORG_NAME)])
# if config.PROJ_NAME is not None:
#     hierarchy_gen_args.extend(["--proj_name", str(config.PROJ_NAME)])

if not config.SKIP_HIERARCHY:
    print(f"With args:\n{hierarchy_gen_args}")
    cellmaps_generate_hierarchycmd.main(hierarchy_gen_args)
else:
    print("Skipping Hierarchy Generation")