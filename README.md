# ProtSeqEmb
Comparison of Protein Sequence Embeddings to Classify Molecular Functions
1. Install requirements.txt:
    * ProtVec:
        * pip install biovec
    * biLSTM:
        (Google Colab)
    * ESM-1b: 
        * pip install fair-esm
    * CPCProt:
        * git clone https://github.com/amyxlu/CPCProt.git
        * cd CPCProt
        * pip install -e .
2. Run dataset_metrics.py for the data set plots (Figure 3.1)
3. Run embeddings.py to obtain the vectors 
4. Run semantics.py for the classification results (Table 4.2)
5. Run clustering.py for the clustering results (Figure 4.1)
6. Run eigenspectrum_plot for the information theory results (Figure 4.2)
