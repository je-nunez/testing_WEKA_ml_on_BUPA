# Testing WEKA ML on BUPA alcoholism dataset

Testing `WEKA` classifiers using Scala on the BUPA liver disorders from excessive alcohol consumption dataset

    [Explanation of the BUPA dataset](http://sci2s.ugr.es/keel/dataset.php?cod=55)

    [UCI ML repository with the BUPA dataset](https://archive.ics.uci.edu/ml/datasets/Liver+Disorders)

This dataset is also available in `R` (at least) in the `Kernel Distance Weighted Discrimination` package:

    https://cran.r-project.org/web/packages/kerndwd/kerndwd.pdf

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

