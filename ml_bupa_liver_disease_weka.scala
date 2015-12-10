#!/usr/bin/env scala -deprecation


// You need to have your CLASSPATH variable including the
//      weka.jar
// file. Eg.
//      export CLASSPATH="$CLASSPATH:/path/to/weka.jar"
// The WEKA library is Copyright (C) since 1999 by the University of Waikato,
//                     Hamilton, New Zealand


import scala.util.Random
import java.io.File

import weka.core.converters.CSVLoader
import weka.core.Utils.splitOptions
import weka.classifiers.Evaluation
import weka.core.{Instance, Instances}

import weka.classifiers.AbstractClassifier
// import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.trees.RandomForest



/** does a WEKA classification on a CSV text file containing the BUPA
  * liver disorders from excessive alcohol consumption dataset.
  *
  * @author Jose Emilio Nunez-Mayans
  * @version 0.0.2
  * @since 0.0.1
  */

object WekaClassifierOnBupaAlcoholism {


  /** returns the WEKA instances of the BUPA dataset in a CSV text file.
    *
    * @param csvFileNameBUPA the CSV text file with the BUPA dataset
    * @return a WEKA Instances object with the BUPA samples
    */

  def loadBupaDataSet(csvFileNameBUPA: String): Instances = {

    // load the BUPA liver disorders instances from a local CSV file.

    val wekaCvsLoader = new CSVLoader()

    System.err.println("Reading training set...")

    // the last feature -the "selector"- has numeric values in the BUPA
    // dataset but is really a nominal feature
    wekaCvsLoader.setOptions(splitOptions("-N last"))

    wekaCvsLoader.setSource(new File(csvFileNameBUPA))

    val instancesBUPA = wekaCvsLoader.getDataSet()
    System.err.println("Read done.")

    // set as the objective dependent variable of the classifier the last
    // feature in the dataset
    instancesBUPA.setClassIndex(instancesBUPA.numAttributes() - 1)

    instancesBUPA
  }


  /** returns the WEKA instances with random instances of the BUPA dataset to
    * be used as test set.
    *
    * @param fromTrainingSet the WEKA instances containing the BUPA samples.
    * @return a WEKA Instances object with the test set. These are also
    *         removed from the original `fromTrainingSet`.
    */

  def randomTestSet(fromTrainingSet: Instances): Instances = {

    // Get a random number between 0 and the size of the Training Set
    val sizeTrainingData = fromTrainingSet.numInstances()
    val rnd = new Random()
    val randomPos = rnd.nextInt(sizeTrainingData)
    val testInstance = fromTrainingSet.instance(randomPos)
    val testInstances = new Instances(fromTrainingSet, 1)
    testInstances.add(testInstance)
    fromTrainingSet.delete(randomPos)

    testInstances
  }


  /** creates the WEKA classifier and initializes its parameters.
    *
    * @param forData WEKA Instances for which the classifier will be created,
    *                so the initial parametrization of the classifier could
    *                some characteristics of these WEKA Instances, like the
    *                number of attributes, etc. NOTE: this method does not
    *                call `buildClassifier(forData)` on the newly created
    *                classifier.
    * @param wekaOptions the WEKA options to set in the created classifier.
    *                    This method initializes the classifier with some
    *                    reasonable values for the parameters, and then
    *                    applies at the last the requested WEKA options string.
    * @return the new WEKA classifier.
    */

  def createWekaClassifier(forData: Instances,
                           wekaOptions: Option[String] = None):
    AbstractClassifier = {

      // create the WEKA classifier object
      // val randomForest = new RandomForest()
      val randomForest = new MyCustomRandomForestOpenBagOfTrees()

      // Assign default values to the parameters of the classifier.

      // How many trees initially: suppose one per attribute, and 20 horizontal
      // clusters with similar instances

      val expectedHorizontalClusters = 20   // 20 horizontal clusters in BUPA instances
      randomForest.setNumTrees((forData.numAttributes() - 1) *
                                 expectedHorizontalClusters)

      // the maximum depth of the tree: 0 == unlimited
      randomForest.setMaxDepth(0)
      // do calculate the estimates of all errors in the classification
      randomForest.setDontCalculateOutOfBagError(false)
      // try to be a robust classifier, so don't break ties randomly
      randomForest.setBreakTiesRandomly(false)
      // number of threads
      randomForest.setNumExecutionSlots(4)

      randomForest.setPrintTrees(true)

      // finally, set the explicit options requested for the classifier
      wekaOptions match {
         case Some(options) => randomForest.setOptions(splitOptions(options))
         case None => None
      }

      // return the new classifier
      randomForest
    }


  /** It runs the main program, building the WEKA classifier from the BUPA
    * training data and applying it on the BUPA test data, and printing the
    * results.
    *
    * @param args ignored so far.
    */

  def main(args: Array[String]) {

    val trainingData = loadBupaDataSet("bupa_liver_disorders.csv")

    val testInstances = randomTestSet(trainingData)

    val eval = new Evaluation(trainingData)

    val wekaClassifier = createWekaClassifier(trainingData)
    println("DEBUG: using WEKA classifier: '" +
              wekaClassifier.getClass().getName() +
              "' with options: " +
              wekaClassifier.getOptions().mkString(" "))

    wekaClassifier.buildClassifier(trainingData)
    /*
    System.err.println("DEBUG: detailed info about the classification: " +
                         wekaClassifier.toString())
     */
    printInferencesWithoutDrinks(wekaClassifier)

    val s = testInstances.toString()
    println("DEBUG: Random instance(s) to be inferred by the classifier:\n" +
            s + "\n")
    eval.evaluateModel(wekaClassifier, testInstances)

    println(eval.toSummaryString("\nResults\n======\n", false))
  }


  /** receives the wekaClassifier whose classifier has already been built for
    * the BUPA dataset of influence of alcoholism on the liver, and report to
    * the standard-output only those statistical inferences which don't have
    * the attribute "drinks" in it. Ie., we are interested in those extreme
    * cases of the statistical inferences found in the BUPA dataset where the
    * state of individual is healthy that it resists some amounts of "drinks"
    * -so they don't have an effect on his/her liver-, or the opposed extreme
    * case, that the biomakers of the liver are so chaotic that the liver has
    * symptoms of alcoholism without "drinks" influencing this state, and
    * which are the boundary values of these biomarkers in which this
    * unfortunate situation occurs (see README of this repository)
    *
    * @param wekaClassifier an instance of MyCustomRandomForestOpenBagOfTrees
    *                       on which the classifier has already been built.
    */

  def printInferencesWithoutDrinks(wekaClassifier: AbstractClassifier) {

    // The format of the print-out of the trees in a random forest in WEKA
    // looks like the lines (for the BUPA datase of influence of alcoholism
    // on the liver):
    //
    //    ...
    //    |   |   |   drinks >= 5.5
    //    |   |   |   |   sgot < 45
    //    |   |   |   |   |   drinks < 13.5
    //
    // where each "|" or (comparitive-expression) is a level in the tree.
    // What we want is to prune those subtrees which has "drinks" in it, because
    // we want to see WEKA's inferences on the BUPA alcoholism dataset where
    // the inference is not affected by the "drinks", ie., very healthy cases
    // where "drinks" don't affect the liver, or very sick cases where the
    // liver is so affected by alcoholism that that the number of "drinks" no
    // longer has any effect on its biomarkers.
    // We could also have told WEKA to ignore the "drinks" attribute _before_
    // building the Random Forest classifier, but in this case, all the trees
    // would be without the "drinks" attribute, and we want those inferences
    // where "drinks" do not influence the result, but it was analyzed and
    // could have influenced the inference.

    // we'll do this task only for the Random Forest classifier we use
    val randomForest = wekaClassifier.asInstanceOf[MyCustomRandomForestOpenBagOfTrees]

    val attributeToPrune = "drinks"   // this is the feature to prune

    for ( strReprTree <- randomForest.getTrees() ) {
      // first version of the code, this needs to be fixed: we need to implement
      // a stack automata which scans every line of the tree and see if this
      // line has a "drinks" token and in what position (tree-level):
      //     if it does, then to skip all the following lines belonging to
      //                      this same subtree
      // and
      //     if this line doesn't have "drinks", then push it in the stack, and
      //                                              continue parsing this subtree
      for ( lineTreeLevel <- strReprTree.split("\n") ) {
        // we need to check this new line (tree-level) whether it has or not
        // the "drinks" attribute in it (we are interested only in those
        // WEKA statistical inferences where "drinks" was discarded.
        println(lineTreeLevel)
      }
    }
  }


  /** Inherits in Scala from a weka.classifiers.trees.RandomForest class in Java
    * in order to access the protected member "m_bagger", which has the different
    * trees the Weka Random Forest classifier has inferred. If we don't inherit
    * from weka.classifiers.trees.RandomForest, then we wouldn't have access to
    * its "m_bagger" field member.
    *
    * This new class only adds a new method, getTrees(), to WEKA's RandomForest.
    */

  class MyCustomRandomForestOpenBagOfTrees
    extends RandomForest() {

      /** returns the Array with the String representation of each tree in the
        * RandomForest, or an Array with an empty string if the RandomForest
        * classifier hasn't been built yet.
        *
        * @return an Array[String]
        */

      def getTrees(): Array[String] = {
        if (m_bagger == null)             // m_bagger can be null in Java
          // m_bagger is a bag classifier with all the different random trees
          // the WEKA classifier generated, and is a protected field member in
          // RandomForest
          Array("")
        else
          // m_bagger is an object of the class weka.classifiers.meta.Bagging,
          // but this class doesn't give access to its protected
          // "m_classifiersCache":
          //    protected java.util.List<weka.classifiers.Classifier> m_classifiersCache;
          // To access this protected "m_classifiersCache" in "m_bagger", we
          // split the String representation "m_bagger" gives from its protected
          // "m_classifiersCache"
          m_bagger.toString().split("^RandomTree$")
      }
    }

}

