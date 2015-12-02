# Testing WEKA ML on BUPA alcoholism dataset

Testing `WEKA` classifiers using Scala on the BUPA liver disorders from excessive alcohol consumption dataset

An explanation of the BUPA dataset is [here](http://sci2s.ugr.es/keel/dataset.php?cod=55)

The BUPA dataset in the UC Irvine Machine Learning Repository is [here](https://archive.ics.uci.edu/ml/datasets/Liver+Disorders)

This dataset is also available in `R` (at least) in the `Kernel Distance Weighted Discrimination` package explained
[here](https://cran.r-project.org/web/packages/kerndwd/kerndwd.pdf)

          install.packages("kerndwd")
          require(kerndwd)
          data(BUPA)
          # head(BUPA)

# WIP

This project is a *work in progress*. The implementation is *incomplete* and
subject to change. The documentation can be inaccurate.

The liver is a very complex organ, with different cells, lot of structures (lobes, etc). Hence, in this BUPA dataset, the influence of one covariate (attribute) on another may exist or may not exist. E.g., the covariate `sgpt` in the dataset (`alamine aminotransferase`) may influence or not another covariate, `gammagt` (`gamma-glutamyl transpeptidase`). This influence in ML may be represented either as one single complex model, or, as structurally the liver is a complex organ with different components, can be split in different partitions with less covariates or less instances, and then each partition be independently analyzed. Ie., structurally the liver may have one symptom (covariate) without showing another. The issue is whether there is any Bayes influence among the covariates. Based on this hypothesis that the liver is composed of different semi-independent sub-structures (or more auto-correlated within themselves than with other sub-structures), then a `random forest` could be a proper simplification of this whole task, which each tree a smaller task anlyzing the hipothesis of that auto-correlated sub-structure in the liver.

Also, a `time-series` could also be of use for analyzing the progression of the relation among the different covariates in the liver disorders arising from excessive alcohol consumption, but the BUPA dataset does not offer such a times-series. Also, in respect of maintaining a time-series, see the financial cost associated with the tests in this dataset (in the [`./costs/`](http://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/costs/)  subdirectory in the UC Irvine Machine Learning Repository for the BUPA samples).

# Required Libraries

The `WEKA` jar file needs to be in the `CLASSPATH` when calling Scala.

(To obtain WEKA, go to its [website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html).)

