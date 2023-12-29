import os
import sys
import json
import ndex2
import pandas as pd
from pathlib import Path

if len(sys.argv) != 3:
    raise ValueError("Require two arguments, the .cx file for the hierarchy and the nodetable (as a .tsv)")

def get_file(p, suffix, arg_num):
    file_path = Path(p).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"{str(file_path)} was not found.")
    if file_path.suffix != suffix:
        raise ValueError(f"Expected arg number {arg_num} to have suffix {suffix}")
    return file_path

cx_file = get_file(sys.argv[1], ".cx", 1)
net = ndex2.create_nice_cx_from_file(str(cx_file))

tsv_file = get_file(sys.argv[2], ".tsv", 2)
df = pd.read_csv(tsv_file, sep='\t')
df.set_index('Name', inplace=True)

SRC, TGT, TYP = "source", "target", "type"
edgelist_df = pd.DataFrame(columns=[SRC, TGT, TYP])
sources, targets = set(), set()
for id, edge in net.get_edges():
    s_id = edge.get('s')
    t_id = edge.get('t')
    source = net.nodes[s_id].get('n')
    target = net.nodes[t_id].get('n')
    edgelist_df.loc[len(edgelist_df.index)] = [source, target, "default"]
    sources.add(source)
    targets.add(target)
leafs = targets - sources
for node in leafs:
    genes = df.loc[node, 'CD_MemberList'].split(' ')
    for gene in genes:
        edgelist_df.loc[len(edgelist_df.index)] = [node, gene, "gene"]
output_file = cx_file.parent / (cx_file.stem + "_edgelist.tsv")
print(f"Writing to {output_file}")
edgelist_df.to_csv(output_file, sep="\t", index=False)
