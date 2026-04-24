
Data Storage and Access
Annotated Data Files
The annotated data files from the dataset are stored in a subfolder /data (added to .gitignore)

Access Example:

dataset = anndata.read_zarr('data/PUB_BLCA_Mariathasan_EGAS00001002556.adata.zarr/')

Note on AnnData datasets writing/saving:

Avoid interruption of anndata writing – it corrupts anndata datasets.

Avoid saving with duplicated columns in .obs – after expected ValueError, .obs will be erased from the saving path, even if it was written previously. 

Related task and notebook with tests: here.


Nomenclature for features in AnnData:
1. obs: DataFrame to store clinical and other necessary per-sample annotations and some calculations

relevant clinical annotations

Kassandra_* prefix – Deconvolution for solid tumors (with SPP1, CXCL9, myCAF, iCAF) 

MFP_* prefix – Classifier probabilities for MFP model (solid tumors)


1. obsm: Dictionary to store all projection data based on gene expression

all possible embeddings calculated by the foundation RNA models, key format is 

obsm/FM_<name_of_the_model>_embeddings

Raw_FGES – Unscaled signatures, respective signature names stored in .uns["fges_names"]

Add nomenclature for other types as needed. Update this paragraph.

1. layers: Field to store gene expression transformations and imputations. Matrix shape should be the same as X. Other TPM expression modifications (different scaling, transformations, etc)

1. uns: Field for storing versions of specific packages/models used in analysis and column names for some objects in .obsm
Example structure:

{
    "fges_names": raw_ssgsea.columns.to_list() # FGES signature names for values stored in .obsm,
}


1. Data Types and Encoding Standards:
Use embedding-type for fields:

Numeric: numeric-scalar

Categories: categorical

Object (almost only Sample_ID): string

Data types within DataFrames:

int, float16, float32, object, categories

No float64! Avoid object type!