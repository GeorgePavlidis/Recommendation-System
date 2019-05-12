# Recommendation-System


Τhis is an experimental implementation of Recommendation System whitch create a random NxM matrix witch representing the rating of Μ items by Ν users. This score is distributed normally from 1 to 5 and the matrix's density is X%.
The system  makes recommendations based on users and based on objects using Knn algorithm with several different similarity functions and compare the results.
 The similarity functions:
 - Jaccard similarity.
 - cosine similarity.
 - Pearson similarity.

## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Numpy, Scipy, Sklearn and matplotlib.

```bash
pip install numpy
pip intall scipy
pip install -U scikit-learn
pip install matplotlib
```

## Configuration File
   - T : the number of iterations of execution
   - Ν : The number of items
   - Μ : The number of users
   - Χ : matrix's density
   - Κ : the number of nearest neighbors


## Files
   - Recommendation_System.py
      the main programm
   - expirements.py
      a script whitch create configuration files
   - Graphs.py
      create a graphical representation of results 
   - run
       a linux script whitch execute expirements.py to create 7 examples, after that execute Recommendation_System.py for all this examples and finnally create graphs for different value of K and X.
       

 ## Execution  
 Recommendation_System.py
 ```bash
 python Recommendation_System.py <configuration-file> (optionally) <dir>
 ```
 <dir>: The direction where to save the result
 
 expirements.py
 ```bash
  python expirements.py <T> <N> <M> <X> <K>
 ``` 
 Graphs.py (example: python Graphs.py expirement1 "k=3 k=5 k=10")
 ```bash
  python Graphs.py <div> <categories>
 ```

