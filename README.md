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

For other use of these biological markers as statistical covariates in the analysis, see, for example in the literature, this [article] (http://www.currentpsychiatry.com/index.php?id=22661&tx_ttnews[tt_news]=173698):

David R. Spiegel, MD, Neetu Dhadwal, MD, Frances Gill, MD  <br />
**"I'm sober, Doctor, really": Best biomarkers for underreported alcohol use**  <br />
*Current Psychiatry, Vol. 7, No. 9 / September 2008*


# WIP

This project is a *work in progress*. The implementation is *incomplete* and
subject to change. The documentation can be inaccurate.

The liver is a very complex organ, with different cells and many structures (lobes, etc). Hence, in this BUPA dataset, the influence of one covariate (attribute) on another may exist or may not exist. E.g., the covariate `sgpt` in the dataset (`alamine aminotransferase`) may influence or not another covariate, `gammagt` (`gamma-glutamyl transpeptidase`). This influence in ML may be represented either as one single complex model with all the covariates and instances, or, as structurally the liver is a complex organ with different components, can be split in different partitions with less covariates or less instances, and then each partition be independently analyzed. Ie., structurally the liver may have one symptom (covariate) without showing another.

It possibly could also be said that those sub-structures in the liver are functionally more auto-correlated within themselves (in respect to some values of these covariates) than with other substructures in the liver up to some extent in some range of values of the covariates, and, from an evolutionary standpoint, it seems possible that the liver has more auto-correlated internal sub-structures than other sub-structures in these ranges of values, because possibly it could be less costly or risky to remain stable in an evolution in these circumstances (this natural evolution in the physiology by stable, or auto-correlated, subchanges in the morphology is debatable though).

The issue is whether there exists any hidden Bayes influence among the covariates. Based on the above tentative hypothesis that the liver is composed of different semi-independent sub-structures (more auto-correlated within themselves than with other sub-structures), then a `random forest` could be a proper simplification of this whole task, which doesn't ignore the possible Bayes influence among some values of the covariariates, and also, with each tree being a smaller task analyzing the hipothesis of that auto-correlated sub-structure in the liver.

Also, a `time-series` could also be of use for analyzing the progression of the auto-correlation among the different covariates in the liver disorders arising from excessive alcohol consumption. Such a time-series can also give an idea of natural kernels where alcoholism appears in sub-structures in the liver. But the BUPA dataset does not offer such a times-series. Also, in respect of maintaining a time-series, see the financial cost associated with the tests in this dataset (in the [`./costs/`](http://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/costs/)  subdirectory in the UC Irvine Machine Learning Repository for the BUPA samples).

For example, this possible, hidden Bayes influence among some ranges of values of some of the covariates can be seen [here] (http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3866949/):

Botros, Mona, and Kenneth A Sikaris.  <br />
**The De Ritis Ratio: The Test of Time.**  <br />
*The Clinical Biochemist Reviews 34.3 (2013): 117–130*

*De Ritis described the ratio between the serum levels of aspartate transaminase (AST) and alanine transaminase (ALT) almost 50 years ago.* (Sidenote: These are two of the covariates given in the BUPA datasets, represented here as the `sgot` and `sgpt` attributes, respectively.) *While initially described as a characteristic of acute viral hepatitis where ALT was usually higher than AST, other authors have subsequently found it useful in alcoholic hepatitis, where AST is usually higher than ALT. These interpretations are far too simplistic however as acute viral hepatitis can have AST greater than ALT, and this can be a sign of fulminant disease, while alcoholic hepatitis can have ALT greater than AST when several days have elapsed since alcohol exposure. The ratio therefore represents the time course and aggressiveness of disease that would be predicted from the relatively short half-life of AST (18 h) compared to ALT (36 h). In chronic viral illnesses such as chronic viral hepatitis and chronic alcoholism as well as non-alcoholic fatty liver disease, an elevated AST/ALT ratio is predictive of long terms complications including fibrosis and cirrhosis... Ideally laboratories should be using pyridoxal phosphate supplemented assays in alcoholic, elderly and cancer patients who may be pyridoxine deplete. Ideally all laboratories reporting abnormal ALT should also report AST and calculate the De Ritis ratio because it provides useful diagnostic and prognostic information.*

# Required Libraries

The `WEKA` jar file needs to be in the `CLASSPATH` when calling Scala.

(To obtain WEKA, go to its [website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html).)

