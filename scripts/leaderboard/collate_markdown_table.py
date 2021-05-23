import os
import sys
import pdb
import json
import argparse

import numpy as np

L_table_columns = [
    "model_name",
    "model_url",
    "reported_fid",
    "reproduced_fid",
    "clean_fid",
    "reported_kid (x 10^3)",
    "reproduced_kid (x 10^3)",
    "clean_kid (x 10^3)",
    "dataset_split",
    "num_reference_images",
    "num_generated_images",
    "dataset_name",
    "dataset_res",
    
    "task_name"
]

def get_ref_num(c):
    if c["dataset_name"]=="cifar10" and c["dataset_split"]=="test":
        return 10_000
    elif c["dataset_name"]=="cifar100" and c["dataset_split"]=="test":
        return 10_000
    elif c["dataset_name"]=="lsun_cat" and c["dataset_split"]=="trainfull":
        return 1657264
    elif c["dataset_name"]=="afhq_dog" and c["dataset_split"]=="train":
        return 4739
    elif c["dataset_name"]=="afhq_wild" and c["dataset_split"]=="train":
        return 4738
    elif c["dataset_name"]=="brecahad" and c["dataset_split"]=="train":
        return 1944
    elif c["dataset_name"]=="metfaces" and c["dataset_split"]=="train":
        return 1336

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--use_kid", action="store_true")
    args = parser.parse_args()
    # load the JSON data file
    with open(args.input_json, "r") as f: 
        D = json.load(f)
    with open(args.output_txt, "w") as f: f.write("\n")

    # make the table headers
    row0,row1 = "|", "|"
    for col_name in L_table_columns:
        if "kid" in col_name and not args.use_kid: pass
        else:
            row0  = f"{row0} {col_name}\t|"
            row1  = f"{row1} :---: |" 
    with open(args.output_txt, "a") as f: 
        f.write(row0+"\n")
        f.write(row1+"\n")


    # iterate over all experiment rows
    for exp_id in D.keys():
        row = "|"
        curr_exp = D[exp_id]
        l_rep_fid = [curr_exp[f"legacy_tensorflow_{i}"] for i in range(10)]
        l_clean_fid = [curr_exp[f"clean_{i}"] for i in range(10)]
        if args.use_kid:
            l_rep_kid = [curr_exp[f"KID_legacy_tensorflow_{i}"] for i in range(10)]
            l_clean_kid = [curr_exp[f"KID_clean_{i}"] for i in range(10)]
        for col_name in L_table_columns:
            if col_name == "reproduced_fid":
                row = f"{row} {np.mean(l_rep_fid):.2f} \u00B1 {np.std(l_rep_fid):.2f}\t|"
            elif col_name == "clean_fid":
                row = f"{row} {np.mean(l_clean_fid):.2f} \u00B1 {np.std(l_clean_fid):.2f}\t|"
            elif "reproduced_kid" in col_name:
                if args.use_kid:
                    row = f"{row} {np.mean(l_rep_kid)*1000:.2f} \u00B1 {np.std(l_rep_kid)*1000:.2f}\t|"
                else: pass
            elif "clean_kid" in col_name:
                if args.use_kid:
                    row = f"{row} {np.mean(l_clean_kid)*1000:.2f} \u00B1 {np.std(l_clean_kid)*1000:.2f}\t|"
                else: pass
            elif "reported_kid" in col_name:
                if args.use_kid: row  = f"{row} {curr_exp['reported_kid']}\t|"
                else: pass
            elif col_name == "num_reference_images":
                row  = f"{row} {get_ref_num(curr_exp)}\t|"
            elif col_name == "model_url":
                row  = f"{row} [ckpt]({curr_exp[col_name]})\t|"
            else:
                row  = f"{row} {curr_exp[col_name]}\t|"
        with open(args.output_txt, "a") as f: 
            f.write(row+"\n")
