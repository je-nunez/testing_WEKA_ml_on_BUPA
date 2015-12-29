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

For other use of these biological markers as statistical covariates in the analysis, see, for example in the specialized literature, this [article] (http://www.currentpsychiatry.com/index.php?id=22661&tx_ttnews[tt_news]=173698):

David R. Spiegel, MD, Neetu Dhadwal, MD, Frances Gill, MD  <br />
[**"I'm sober, Doctor, really": Best biomarkers for underreported alcohol use**](http://www.currentpsychiatry.com/index.php?id=22661&tx_ttnews[tt_news]=173698)  <br />
*Current Psychiatry, Vol. 7, No. 9 / September 2008*

**Disclaimer**: the present project is not related, nor supported, nor endorsed by the references given in the present document, who should not be contacted in relation to this project.

# WIP

This project is a *work in progress*. The implementation is *incomplete* and
subject to change. The documentation can be inaccurate.

# Description

The liver is a very complex organ, with different cells and many structures (lobes, etc). Hence, in this BUPA dataset, the influence of one covariate (attribute) on another may exist or may not exist. E.g., the covariate `sgpt` in the dataset (`alamine aminotransferase`) may influence or not another covariate, `gammagt` (`gamma-glutamyl transpeptidase`). This influence in ML may be represented either as one single complex model with all the covariates and instances, or, as structurally the liver is a complex organ with different components, can be split in different partitions with less covariates or less instances, and then each partition be independently analyzed. Ie., structurally the liver may have one symptom (covariate) without showing another.

It possibly could also be said that those sub-structures in the liver are functionally more auto-correlated within themselves (in respect to some values of these covariates) than with other substructures in the liver up to some extent in some range of values of the covariates, and, from an evolutionary standpoint, it seems possible that the liver has more auto-correlated internal sub-structures than other sub-structures in these ranges of values, because possibly it could be less costly or risky to remain stable in an evolution in these circumstances (this natural evolution in the physiology by stable, or auto-correlated, subchanges in the morphology is debatable though).

The issue is whether there exists any hidden Bayesian influence among the covariates or other confounding variable(s) intermediate among them. Based on the above tentative hypothesis that the liver is composed of different semi-independent sub-structures (more auto-correlated within themselves than with other sub-structures), then a `random forest` could be a proper simplification of this whole task, which doesn't ignore the possible Bayesian influence among some values of the covariariates, and also, with each tree being a smaller task analyzing the hipothesis of that auto-correlated sub-structure in the liver.

Also, a `time-series` could also be of use for analyzing the progression of the auto-correlation among the different covariates in the liver disorders arising from excessive alcohol consumption and also, to hint on intermediate confounding features. Such a time-series can also give an idea of natural kernels where alcoholism appears in sub-structures in the liver. But the BUPA dataset does not offer such a times-series. Also, in respect of maintaining a time-series, see the financial cost associated with the tests in this dataset (in the [`./costs/`](http://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/costs/)  subdirectory in the UC Irvine Machine Learning Repository for the BUPA samples).

For example, this possible, hidden Bayesian influence or intermediate confounding features among some ranges of values of some of the covariates can be seen [here] (http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3866949/):

Botros, Mona, and Kenneth A Sikaris.  <br />
[**The De Ritis Ratio: The Test of Time.**](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3866949/)  <br />
*The Clinical Biochemist Reviews 34.3 (2013): 117–130*

This is a cite from the article above with a **sidenote** of us in relation to the covariates in the BUPA dataset:

*De Ritis described the ratio between the serum levels of aspartate transaminase (AST) and alanine transaminase (ALT) almost 50 years ago.* (**Sidenote**: These are two of the covariates given in the BUPA datasets, represented by the `sgot` and `sgpt` biomarker features for the `AST` and `ALT` enzimes, respectively.) *While initially described as a characteristic of acute viral hepatitis where ALT was usually higher than AST, other authors have subsequently found it useful in alcoholic hepatitis, where AST is usually higher than ALT. These interpretations are far too simplistic however as acute viral hepatitis can have AST greater than ALT, and this can be a sign of fulminant disease, while alcoholic hepatitis can have ALT greater than AST when several days have elapsed since alcohol exposure. The ratio therefore represents the time course and aggressiveness of disease that would be predicted from the relatively short half-life of AST (18 h) compared to ALT (36 h). In chronic viral illnesses such as chronic viral hepatitis and chronic alcoholism as well as non-alcoholic fatty liver disease, an elevated AST/ALT ratio is predictive of long terms complications including fibrosis and cirrhosis... Ideally laboratories should be using pyridoxal phosphate supplemented assays in alcoholic, elderly and cancer patients who may be pyridoxine deplete. Ideally all laboratories reporting abnormal ALT should also report AST and calculate the De Ritis ratio because it provides useful diagnostic and prognostic information.*

# BUPA liver disorders from excessive alcohol consumption classified by WEKA's Random Forest

This is an example of how most instances of the BUPA dataset are seen by the WEKA's Random Forest. (This Scala program takes one instance of the BUPA dataset and removes it.) The description of the labels (the first five of them are the biomarkers in the liver, the sixth is the number of drinks daily, and the seventh is internally calcalated by the program, the De Ritis Ratio, before training the WEKA classifier and evaluating it on the test set) are:

      1. mcv:      mean corpuscular volume
      2. alkphos:  alkaline phosphotase
      3. sgpt:     alamine aminotransferase (ALT)
      4. sgot:     aspartate aminotransferase (AST)
      5. gammagt:  gamma-glutamyl transpeptidase
      6. drinks:   number of half-pint equivalents of alcoholic beverages drunk per day
      7. De_Ritis_Ratio:   the De Ritis ratio from the specialized literature.
                           (The De Ritis ratio is internally calculated, it is not
                            used in the original BUPA dataset in the UCI archive)
      8. selector: whether this indivual suffers from alcoholism (1 = No/2 = Yes)

Using WEKA classifier: **weka.classifiers.trees.RandomForest** with options: `-I 120 -K 0 -S 1 -print -num-slots 4` (these parameters for the WEKA Random Forest classifier are similar in idea to the Random Forest in Apache Mahout: see some Mahout examples here [https://mahout.apache.org/users/classification/partial-implementation.html](https://mahout.apache.org/users/classification/partial-implementation.html) and in [https://mahout.apache.org/users/classification/breiman-example.html](https://mahout.apache.org/users/classification/breiman-example.html)).

     ...
     De_Ritis_Ratio < 0.95
     |   gammagt < 35.5
     |   |   gammagt < 14.5
     |   |   |   De_Ritis_Ratio < 0.91 : 1 (30/0)
     |   |   |   De_Ritis_Ratio >= 0.91
     |   |   |   |   sgpt < 21 : 1 (2/0)
     |   |   |   |   sgpt >= 21 : 2 (1/0)
     ...

It seems that some `times-series` would be nice in order to analyze the evolution of the relationship among the liver biomarkers and alcoholism, because some trees generated by the WEKA Random Forest don't need the `drinks` feature, as if the effect of the previous state in time of the liver influences these samples so that `drink` is no longer meaningful in the present state. A time-series can help track this evolution up to this point where `drinks` is no longer influential on the result of alcoholism and to hint on other confounding variables during the period this state remains. For example (although, again, from the 346 instances in the BUPA dataset, the program removes one at random for its test set):

     gammagt < 20.5
     |   sgpt < 21.5
     |   |   sgot < 6.5 : 1 (2/0)
     |   |   sgot >= 6.5
     |   |   |   alkphos < 60.5
     |   |   |   |   sgot < 14.5
     |   |   |   |   |   mcv < 93 : 2 (5/0)
     |   |   |   |   |   mcv >= 93 : 1 (2/0)
     |   |   |   |   sgot >= 14.5 : 2 (28/0)  <---------------------

So, in order for this `time-series` of the evolution of the liver under alcoholism (this time-series is not given in the BUPA dataset), this program also tries to find and print all statistical inferences found in the random forest where the state of the sample individuals is healthy that it resists some amounts of `drinks` -so they are tolerated and don't have an effect on his/her liver, ie., they are very far from abuse-, or the opposed extreme case, that the biomakers of the liver given in BUPA are so moved that the liver has symptoms of alcoholism without `drinks` influencing this state, and, in these extreme cases, to find what are the boundary values of these biomarkers in which this unfortunate situation occurs. Ie., these trees are pure or extreme cases of the state of the liver, very healthy or very ill, where the `drinks` attribute was analyzed in each step of the classification, but finally was found to have no remarkable relevance for being in the kernel for these extreme cases of health or illness. One example is the tree above this paragraph, and also this tree below, where `drinks` neither affected the inference for this other subset of samples in BUPA:

     sgpt < 21.5
     |   gammagt < 20.5
     |   |   gammagt < 14.5
     |   |   |   alkphos < 70.5
     |   |   |   |   sgot < 14.5
     |   |   |   |   |   sgpt < 9.5 : 2 (2/0)
     |   |   |   |   |   sgpt >= 9.5 : 1 (1/0)
     |   |   |   |   sgot >= 14.5 : 2 (30/0)

These trees form a boundary of values of the biomarkers for the liver between its very healthy state and its very ill state, where `drinks` no longer influences. E.g., in the two subtrees above without `drinks`, it is seen that the value of the biomarker `sgot >= 14.5` (`AST >= 14.5`) is very influential, as well as `sgpt < 21.5` (`ALT < 21.5`), so they could form a risky contour of samples where `drinks` no longer acts as a catalyst in the state of the liver, *according to the samples given in the BUPA dataset*. (Of course, other liver biomarkers also appear in these extreme BUPA trees where `drinks` no longer influences the inference, like `alkphos < 60.5` or `alkphos < 70.5` and `gammagt < 20.5` or `gammagt < 14.5`, so they must be taken as a set of descriptive conditions: but this report that the program gives is not intended as an exact mathematical line of contour dividing health and illness, but as a hint region for further medical research on the characteristics of a time-series **before** it got closer to these inflexion regions in the biomarkers where the liver falls in a state where `drinks` is no longer relevant, or a time-series on how the liver can improve **after** falling into these inflexion values -e.g., to make these biomarkers return again to normal, healthy regions.)

Not all these extreme regions require a set of descriptive conditions, but some are very simple. E.g., WEKA also reports,

     sgot >= 46 : 2 (13/0)

a one-level subtree (a leaf under the root), with no other biomarker required for its inference but only: `sgot >= 46` ( `AST >= 46`), according to the Random Forest -the measuring units are according to the BUPA dataset. Hence, the Random Forest with options `-I 120 -K 0 -S 1 -print -num-slots 4` infers from the samples in the BUPA dataset that most individuals who have `sgot >= 46` ( `AST >= 46`), have liver problems, and all other biomarkers of the liver do not give more valuable statistical information for this extreme region, not only the `drinks` feature that we were trying to prune. To verify this simple result against the BUPA dataset directly to see how representative of the total population it is, 14 samples (4.05% of the total) have `sgot >= 46` ( `AST >= 46`), and which, as the Random Forest said, also have `selector == 2`, ie., the liver has symptoms of suffering from alcoholism:

mcv|alkphos|sgpt|**sgot**|gammagt|drinks|**selector**
|--:|--:|--:|--:|--:|--:|--:|
91|72|155|**68**|82|0.5|**2**
87|76|22|**55**|9|4.0|**2**
90|96|34|**49**|169|4.0|**2**
91|74|87|**50**|67|6.0|**2**
93|84|58|**47**|62|7.0|**2**
92|95|85|**48**|200|8.0|**2**
91|62|59|**47**|60|8.0|**2**
95|80|50|**64**|55|10.0|**2**
98|74|148|**75**|159|0.5|**2**
85|58|83|**49**|51|3.0|**2**
94|117|77|**56**|52|4.0|**2**
91|86|52|**47**|52|4.0|**2**
94|43|154|**82**|121|4.0|**2**
102|82|34|**78**|203|7.0|**2**

The only sample in the BUPA dataset which excepts the statistical majority inference of `sgot >= 46 : 2` ( `AST >= 46 : 2`) is:

mcv|alkphos|sgpt|**sgot**|gammagt|drinks|**selector**
|--:|--:|--:|--:|--:|--:|--:|
98|66|103|**57**|114|6.0|**1**

Related to this, the program is able to report the subset of training samples which statistically support an inference a random tree gives. This may be useful to hint about other confounding biomarkers in the state of these observations. It is the `--dump` command line option, which prints, after each leaf of the tree, the subset of training samples under it (recall that one random sample is removed from the BUPA dataset to be used as the test set, so this will not appear). For example:

     mcv >= 85.5
     |   gammagt >= 20.5
     |   |   sgpt < 18.5 : 2 (20/0)
                            ( mcv >= 85.5 ) and ( gammagt >= 20.5 ) and ( sgpt < 18.5 )
                            mcv,alkphos,sgpt,sgot,gammagt,drinks,De_Ritis_Ratio,selector
                            94,48,11,23,43,0.5,2.090909,2
                            92,61,18,13,81,3,0.722222,2
                            89,90,15,17,25,4,1.133333,2
                            89,76,14,21,24,4,1.5,2
                            87,64,16,20,24,5,1.25,2
                            90,63,12,26,21,6,2.166667,2
                            90,79,18,15,24,0.5,0.833333,2
                            101,65,18,21,22,0.5,1.166667,2
                            86,58,16,23,23,0.5,1.4375,2
                            93,87,18,17,26,2,0.944444,1  <---------
                            91,44,18,18,23,2,1,2
                            93,45,11,14,21,4,1.272727,2
                            91,63,17,17,46,4,1,2
                            88,46,15,33,55,4,2.2,2
                            99,42,14,21,49,5,1.5,2
                            93,43,11,16,54,6,1.454545,1  <---------
                            86,109,16,22,28,6,1.375,2
                            97,80,17,20,53,8,1.176471,2

The majority of the instances in this region above have `selector` with value `2`, except those pointed out which have value `1`.

Note: We could also have told WEKA to ignore the `drinks` attribute **before** building the Random Forest classifier, but in this case, all the trees would be without the `drinks` attribute, and we want those inferences where `drinks` do not influence the result, but `drinks` was nevertheless analyzed and potentially could have influenced each step of the inference. Ie., we do want to analyze the feature `drinks` in the inference, but to report those extreme trees (cases) of state of the liver where `drinks` no longer acts as a catalyst in them.

# Required Libraries

The `WEKA` jar file needs to be in the `CLASSPATH` when calling Scala. It has been tested with WEKA version 3.7.13.

(To obtain WEKA, go to its [website](http://www.cs.waikato.ac.nz/ml/weka/downloading.html).)

