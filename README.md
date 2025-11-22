# Bank Market Analysis Project
On this project we are predicting whether clients will subscribe to a term deposit using the Bank Marketing dataset. A logistic regression model was developed, incorporating all available predictor variables after appropriate preprocessing. The model was evaluated using shuffled cross-validation with an emphasis on the F1 score balance precision and recall. The analysis was conducted using Python and key libraries such as NumPy, pandas, and scikit-learn, with all code documented for reproducibility.
Our final classifier performed fairly well on an unseen test data set, achieving an accuracy of 0.844, f1-score of 0.551, and roc-auc score of 0.91. This indicates that the model is reasonably effective at identifying clients who will subscribe to a term deposit, although there is room for improvement, particularly in recall. Further refinements could involve exploring additional features, tuning hyperparameters, or experimenting with alternative modeling techniques to enhance predictive performance.
# List of author
- Charlene Chin
- Daniel Yorke 
- Jackson Lu
- Mohammed Ibrahim
# Usage
First time running the project,
run the following from the root of this repository:

``` bash
conda-lock install --name term-deposit-predictor conda-lock.yml
```
To run the analysis notebook, launch JupyterLab with the following command:

``` bash  
conda activate term-deposit-predictor
jupyter lab
```
Then open `term-deposit-predictor-analysis.ipynb` in JupyterLab.  

# Dependencies 
  - `python>=3.10`
  - `pandas==2.1.4`
  - `ucimlrepo`
  - `jupyterlab`
  - `nb_conda_kernels`
  - Python and packages listed in [`environment.yml`](environment.yml)
# Licenses
MIT License
This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
# References
- [UCI Machine Learning Repository: Bank Marketing Dataset.](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
- DSCI 571 lecture notes.
- DSCI 573 lecture notes.

# Acknowledgements
We would like to thank the creators of the Bank Marketing dataset for making this valuable resource available to the research community. Their work has significantly contributed to advancements in predictive modeling and data analysis within the banking sector.

