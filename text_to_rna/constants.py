import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

DEPLOYMENT = False
PSEUDO_COUNT = 1
STD_EPS = 1
STD_SCALE = True

TOTAL_GENES = 44592
STUDY_BLACKLIST = {
    "TCGA-BLCA",
    "TCGA-KIRP",
    "TCGA-UVM",
    "TCGA-UCEC",
    "TCGA-PAAD",
    "TCGA-CESC",
    "TCGA-SARC",
    "TCGA-BRCA",
    "TCGA-THYM",
    "TCGA-COAD",
    "TCGA-STAD",
    "TCGA-SKCM",
    "TCGA-HNSC",
    "TCGA-READ",
    "TCGA-DLBC",
    "TCGA-ACC",
    "TCGA-PCPG",
    "GTEx",
    "TCGA-GBM",
    "TCGA-LUSC",
    "TCGA-PRAD",
    "TCGA-TGCT",
    "TCGA-ESCA",
    "TCGA-LIHC",
    "TCGA-MESO",
    "TCGA-LAML",
    "TCGA-UCS",
    "TCGA-OV",
    "TCGA-LUAD",
    "TCGA-CHOL",
    "TCGA-KIRC",
    "TCGA-THCA",
    "TCGA-LGG",
    "TCGA-KICH",
}

MODALITIES = {
    "czi": {
        "n_features": 37_125,
        "n_targets": 37_125,
    },
    "perturbseq": {
        "n_features": 37_125,
        "n_targets": 37_125,
    },
    "bulk": {
        "n_features": 44_592,
        "n_targets": 44_592,
    },
}
