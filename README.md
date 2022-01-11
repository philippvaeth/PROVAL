# Code repository "Comparison of Protein Sequence Embeddings to Classify Molecular Functions"
<!-- 1. Install requirements:
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
        * mkdir -p dataset_metrics eigenspectrum temp tsne vecs
2. Run dataset_metrics.py for the data set plots (Figure 3.1)
3. Run embeddings.py to obtain the vectors 
4. Run semantics.py for the classification results (Table 4.2)
5. Run clustering.py for the clustering results (Figure 4.1)
6. Run eigenspectrum_plot for the information theory results (Figure 4.2) -->
## PROVAL Setup
1. (optional) Use virtual environment, e.g.:  
`conda create --name proval`  
`conda activate proval`
2. Install (framework) requirements:  
`pip install -r requirements.txt`
<!-- 3. Create directories 
`mkdir -p dataset_metrics eigenspectrum temp tsne vecs` -->
---

## Extension to Other Embedding Algorithms
<details>
  <summary>Integration into embedding.py</summary>

  1. Load pretrained model
  2. Add function to embedding_utils.py, which takes the train and test sequences as lists of Bio sequences (see read_fasta() in utils.py) and returns the vectors in a dictionary of the form id(String):vector(NumPy array)
  3. Add approach to embedding list (embeddings.py, line 17)
  4. Add embedding function call to the if/elif statements in the similar form
  5. Run embeddings.py and the respective comparison scripts
</details>
or 
<details>
  <summary>Custom integration through vector file</summary>

  1. Load the train and test sequences as lists of Bio sequences (see read_fasta() in utils.py)
  2. Use custom embedding to predict the embedding vector for each sequence in the dictionary format id(String):vector(NumPy array).
  3. Truncate the vectors to d=100 if necessary, compare embeddings.py
  4. Save as pickle '.p' file, compare embeddings.py
</details>   

---

## Full Reproducibility of the Paper Results
_Note, the extraction of the vectors and the results might not be fully deterministic and small deviations might be possible._
<details>
  <summary>Data set (optional)</summary>

  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>

<details>
  <summary>Embedding methods</summary>
  
  

  1. Install full requirements  
  `pip install -r requirements_full.txt`  
  2. Clone and setup the Smith-Waterman alignment script  
  `git clone https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library.git`  
  `cd Complete-Striped-Smith-Waterman-Library/src`  
  `make`
  3. Run embeddings.py to obtain the vectors  
</details>

<details>
  <summary>Figures</summary>

   * Run dataset_metrics.py for optional data set plots
   * Run semantics.py for the classification results (Table 3)
   * Run visualization.py for the visualization results (Figure 7)
   * Run eigenspectrum_plot.py for the information theory results (Figure 8)
</details>
FTP Repo of Data https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2021_02/ 

---