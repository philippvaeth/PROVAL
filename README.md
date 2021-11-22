# Code repository "Comparison of Protein Sequence Embeddings to Classify Molecular Functions"
1. Install requirements:
    * Conda environment: 
        * conda env create -f environment.yml
    * Smith-Waterman SSW Library:
        * Download the software from https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library.
        * cd src
        * make
    * ProtVec:
        * pip install biovec
    * biLSTM:
        (Google Colab: https://colab.research.google.com/drive/1awz63tkC8u2RHF25n6BElcq5C_XS5wSt?usp=sharing)
        or use the provided "bilstm_vecs_proj.p" vectors
    * ESM-1b: 
        * pip install fair-esm
    * CPCProt:
        * git clone https://github.com/amyxlu/CPCProt.git
        * cd CPCProt
        * pip install -e .
    * Create directories: 
        * mkdir dataset_metrics eigenspectrum temp tsne vecs
2. Run dataset_metrics.py for the data set plots (Figure 3.1)
3. Run embeddings.py to obtain the vectors 
4. Run semantics.py for the classification results (Table 4.2)
5. Run clustering.py for the clustering results (Figure 4.1)
6. Run eigenspectrum_plot for the information theory results (Figure 4.2)

# Extension to other embedding algorithms
1. Load pretrained model
2. Add function to embedding_utils.py, which takes the train and test sequences as lists of Bio sequences (see read_fasta() in utils.py) and returns the vectors in a dictionary of the form id(String):vector(NumPy array)
3. Add approach to embedding list (embeddings.py, line 17)
4. Add embedding function call to the if/elif statements in the simiar form
5. Run embeddings.py and the respective comparison scripts


FTP Repo of Data https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2021_02/ 
