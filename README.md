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

# Required Libraries

The `WEKA` jar file needs to be in the `CLASSPATH` when calling Scala.

(To obtain WEKA, go to its [website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html).)

