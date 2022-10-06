import argparse
import json


def get_options():

    description = "Filters out the synonymous and upstream snps from the .vcf file"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--infile", help=".vcf file including all dataste ")
    parser.add_argument(
        "--annot",
        help="annotation of markers in marker:annotation dictionary in .json format",
    )
    parser.add_argument("--out", help=".vcf file including only nsv")

    return parser.parse_args()


options = get_options()


def all2nsv(infile, annot, out):
    """

    Input: .vcf file including all variants and .json file including dictionary of maker:sv/nsv data
    output: .vcf file including only NSVs (non syonymous variants)"""
    ###@1)importing the data and libraries

    vcf_all = infile
    # getting the syn_nonsyn data for each snp
    with open(annot) as f:
        cf2syn_nonsyn = json.load(f)

    unwanted = ["synonymous_variant", "upstream_gene_variant"]
    vcf_nsv = open(out, "w")
    nsv_count = 0
    with open(vcf_all, "r") as file:
        line = file.readline()
        while line[0] == "#":
            vcf_nsv.write(line)
            line = file.readline()
        while line:
            marker_name = line.split("\t")[2].replace(
                "Pseudomonas_aeruginosa_P749_PES", "PA749"
            )
            if cf2syn_nonsyn[marker_name] not in unwanted:
                nsv_count += 1
                vcf_nsv.write(line)
            line = file.readline()
    print("{} number of NSVs in the data".format(nsv_count))
    vcf_nsv.close()


all2nsv(options.infile, options.annot, options.out)
