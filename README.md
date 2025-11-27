# Bank Market Analysis Project
On this project we are predicting whether clients will subscribe to a term deposit using the Bank Marketing dataset. A logistic regression model was developed, incorporating all available predictor variables after appropriate preprocessing. The model was evaluated using shuffled cross-validation with an emphasis on the F1 score balance precision and recall. The analysis was conducted using Python and key libraries such as NumPy, pandas, and scikit-learn, with all code documented for reproducibility.

Our final classifier performed fairly well on an unseen test data set, achieving an accuracy of 0.844, f1-score of 0.551, and roc-auc score of 0.91. This indicates that the model is reasonably effective at identifying clients who will subscribe to a term deposit, although there is room for improvement, particularly in recall. Further refinements could involve exploring additional features, tuning hyperparameters, or experimenting with alternative modeling techniques to enhance predictive performance.
# List of author
- Charlene Chin
- Daniel Yorke 
- Jackson Lu
- Mohammed Ibrahim

# Report

The final report can be found
[here](https://jluover9000.github.io/proj-522/).

# Usage
1. First time running the project,
run the following from the root of this repository:

``` bash
docker compose up
```
2. In the terminal, look for a URL that starts with `http://127.0.0.1:8888/lab?token=...`. Copy and paste that URL into your browser.

3. To run the analysis,
open `docs/term-deposit-predictor-analysis.ipynb` in Jupyter Lab you just launched
and under the "Kernel" menu click "Restart Kernel and Run All Cells...".

### Clean up

To shut down the container and clean up the resources, 
type `Cntrl` + `C` in the terminal
where you launched the container, and then type `docker compose rm`

# Dependencies 
  - `python>=3.10`
  - `pandas==2.1.4`
  - `ucimlrepo`
  - `jupyterlab`
  - `nb_conda_kernels`
  - Python and packages listed in [`environment.yml`](environment.yml)
# Licenses
- MIT License
- This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
# References
- [UCI Machine Learning Repository: Bank Marketing Dataset.](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
- [scikit-learn documentation]( https://scikit-learn.org/stable/)
- [Bera, Suman, Deeparnab Chakrabarty, Nicolas Flores, and Maryam Negahbani. 2019. “Fair Algorithms for Clustering.” ](<https://www.semanticscholar.org/paper/Fair-Algorithms-for-Clustering-Bera-Chakrabarty/34a46c62cb3a7809db4ed7d0c1a651f538b9fe87>)

- [Ziko, Imtiaz, Eric Granger, Jing Yuan, and Ismail Ayed. 2019. “Clustering with Fairness Constraints: A Flexible and Scalable Approach.” ](<https://www.semanticscholar.org/paper/Clustering-with-Fairness-Constraints%3A-A-Flexible-Ziko-Granger/d56841fe68f2a913583a40edf541efeaed0a7e5b>)

- [Lamy, Alexandre, Ziyuan Zhong, Aditya Menon, and Nakul Verma. 2019. “Noise-Tolerant Fair Classification.” ](<https://www.semanticscholar.org/paper/Noise-tolerant-fair-classification-Lamy-Zhong/c4ac496bf57410638260196a25d8ae3366ea03c7>)

- [Iosifidis, Vasileios, and Eirini Ntoutsi. 2019. “AdaFair: Cumulative Fairness Adaptive Boosting.” ](<https://www.semanticscholar.org/paper/AdaFair%3A-Cumulative-Fairness-Adaptive-Boosting-Iosifidis-Ntoutsi/18fe4800f3c85f315d79063d6b0fe38c7610ad45>)

- [Vaz, Afonso, Rafael Izbicki, and Rafael Stern. 2018. “Quantification under Prior Probability Shift: The Ratio Estimator and Its Extensions.” ](<https://www.semanticscholar.org/paper/Quantification-under-prior-probability-shift%3A-the-Vaz-Izbicki/50adf7b8fd1274149a195ef4a7b4ab9f84b3dd13>)

- [Zhu, Zining, Jekaterina Novikova, and Frank Rudzicz. 2018. “Semi-supervised Classification by Reaching Consensus among Modalities.” ](<https://www.semanticscholar.org/paper/Semi-supervised-classification-by-reaching-among-Zhu-Novikova/072956b72ddc23f276b18da0c9a6ccc5ed5067e8>)

- [Yoon, Jinsung, William R. Zame, and Mihaela van der Schaar. 2017. “ToPs: Ensemble Learning with Trees of Predictors.”](<https://www.semanticscholar.org/paper/ToPs%3A-Ensemble-Learning-With-Trees-of-Predictors-Yoon-Zame/05268691d4bf6b84e71ae421a3af0ab27cd3d8f1>)

- [Ross, Stéphane, Paul Mineiro, and John Langford. 2014. “Normalized Online Learning.”](<https://www.semanticscholar.org/paper/Normalized-Online-Learning-Ross-Mineiro/1d127af1174a3f0f36e9181348eaa731d3cca67b>)
- DSCI 571 lecture notes.
- DSCI 573 lecture notes.

# Acknowledgements
We would like to thank the creators of the Bank Marketing dataset for making this valuable resource available to the research community. Their work has significantly contributed to advancements in predictive modeling and data analysis within the banking sector.

