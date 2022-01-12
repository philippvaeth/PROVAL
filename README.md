# Code repository "Comparison of Protein Sequence Embeddings to Classify Molecular Functions"
## PROVAL Setup
1. (optional) Use virtual environment, e.g.:  
`conda create --name proval`  
`conda activate proval`  
`conda install pip`  
2. Install (framework) requirements:  
`pip install -r requirements.txt`
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


  Steps to reproduce the _test.fasta_ and _train.fasta_ files in the _data_ folder:
  1. Download the full SwissProt data set (release 02/2021):  
  https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2021_02/ 
  2. Select the sequence IDs, the sequence strings and the molecular function information ('GO:xxxxxx' terms)
  3. Discard all sequences with more than one molecular function (to reduce the complexity of the experiments)
  4. Select 1000 random sequences for each of the most frequent 15 molecular functions (=15,000 sequences)
  5. Randomly split the sequences in training and test sets (70:30)
  6. Save the sequences in the _.fasta_ format, compare the _test.fasta_ and _train.fasta_ files in the _data_ folder:
     >\<Sequence ID\> \[\<GO-ID\>\]  
     \<Sequence\>  
     \<Sequence ID\> \[\<GO-ID\>\]  
     \<Sequence\>  
     ...
    
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
---