# Code repository for the Master thesis "Comparison of Protein Sequence Embeddings to Classify Molecular Functions" by Philipp Väth
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
        (Google Colab: )
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
