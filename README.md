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

The liver is a very complex organ, with different cells and many structures (lobes, etc). Hence, in this BUPA dataset, the influence of one covariate (attribute) on another may exist or may not exist. E.g., the covariate `sgpt` in the dataset (`alamine aminotransferase`) may influence or not another covariate, `gammagt` (`gamma-glutamyl transpeptidase`). This influence in ML may be represented either as one single complex model with all the covariates and instances, or, as structurally the liver is a complex organ with different components, can be split in different partitions with less covariates or less instances, and then each partition be independently analyzed. Ie., structurally the liver may have one symptom (covariate) without showing another.

It possibly could also be said that those sub-structures in the liver are functionally more auto-correlated within themselves (in respect to some values of these covariates) than with other substructures in the liver up to some extent in some range of values of the covariates, and, from an evolutionary standpoint, it seems possible that the liver has more auto-correlated internal sub-structures than other sub-structures in these ranges of values, because possibly it could be less costly or risky to remain stable in an evolution in these circumstances (this natural evolution in the physiology by stable, or auto-correlated, subchanges in the morphology is debatable though).

The issue is whether there exists any hidden Bayes influence among the covariates. Based on the above tentative hypothesis that the liver is composed of different semi-independent sub-structures (more auto-correlated within themselves than with other sub-structures), then a `random forest` could be a proper simplification of this whole task, which doesn't ignore the possible Bayes influence among some values of the covariariates, and also, with each tree being a smaller task analyzing the hipothesis of that auto-correlated sub-structure in the liver.

Also, a `time-series` could also be of use for analyzing the progression of the auto-correlation among the different covariates in the liver disorders arising from excessive alcohol consumption. Such a time-series can also give an idea of natural kernels where alcoholism appears in sub-structures in the liver. But the BUPA dataset does not offer such a times-series. Also, in respect of maintaining a time-series, see the financial cost associated with the tests in this dataset (in the [`./costs/`](http://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/costs/)  subdirectory in the UC Irvine Machine Learning Repository for the BUPA samples).

# Required Libraries

The `WEKA` jar file needs to be in the `CLASSPATH` when calling Scala.

(To obtain WEKA, go to its [website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html).)

