#!/usr/bin/env scala -deprecation


// You need to have your CLASSPATH variable including the
//      weka.jar
// file. Eg.
//      export CLASSPATH="$CLASSPATH:/path/to/weka.jar"
// The WEKA library is Copyright (C) since 1999 by the University of Waikato,
//                     Hamilton, New Zealand


import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import java.io.File

import weka.core.converters.CSVLoader
import weka.core.Utils.splitOptions
import weka.classifiers.Evaluation
import weka.core.{Instance, Instances}
import weka.filters.unsupervised.attribute.AddExpression
import weka.filters.unsupervised.attribute.Reorder
import weka.filters.MultiFilter
import weka.filters.Filter

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


  /** adds the De Ritis Ratio calculated attribute to the instances read from
    * the BUPA dataset.
    *
    * @param originalInstances the original instances read from the BUPA dataset
    * @return a WEKA Instances object with the same instances and their De Ritis Ratio
    */

  def addRitisRatio(originalInstances: Instances): Instances = {
    // The De Ritis Ratio is defined as:
    //       = AST / ALT
    //
    // In the BUPA instances, AST is the "sgot" attribute, and ALT the "sgpt" attrib

    // newData = new Instances(data)
    val filterDeRitisRatio = new AddExpression()
    // filterDeRitisRatio.setIndex(originalInstances.numAttributes() - 2)
    filterDeRitisRatio.setName("De_Ritis_Ratio")

    // set the expression to find the De Ritis Ratio. First we need to find
    // positions of "sgot" (AST) attrib and "sgpt" (ALT) attrib in the structure
    val existingAttribNames = getAllAttribNames(originalInstances)
    val idx_ast_sgot = existingAttribNames.indexOf("sgot") + 1
    val idx_alt_sgpt = existingAttribNames.indexOf("sgpt") + 1

    val wekaAttribExpression = f"a$idx_ast_sgot/a$idx_alt_sgpt"
    // System.err.println(f"DEBUG: De Ritis Liver Ratio: $wekaAttribExpression")
    filterDeRitisRatio.setExpression(wekaAttribExpression)
    filterDeRitisRatio.setInputFormat(originalInstances)

    val instancesWithDeRitisRatio = Filter.useFilter(originalInstances, filterDeRitisRatio)

    // swap (reorder) the position of the last and second-to-last attrib columns,
    // so that the new calculated De Ritis Ratio is not the last column, but
    // the second-to-last, and the last attribute is the label, the Liver
    // disorder classification

    val filterReorder = new Reorder()
    var attribPositions = Array.tabulate(existingAttribNames.size + 1)((i) => i)
    attribPositions(attribPositions.size - 1) = attribPositions.size - 2
    attribPositions(attribPositions.size - 2) = attribPositions.size - 1

    filterReorder.setAttributeIndicesArray(attribPositions)
    filterReorder.setInputFormat(instancesWithDeRitisRatio)
    filterReorder.setDebug(true)

    val reOrderedAttribInstances = Filter.useFilter(instancesWithDeRitisRatio, filterReorder)

    // set the last column (the Liver disorder) as the classification label
    reOrderedAttribInstances.setClassIndex(reOrderedAttribInstances.numAttributes() - 1)

    val newRelationName = "De Ritis Ratio in BUPA Liver Disorders Data Set"
    reOrderedAttribInstances.setRelationName(newRelationName)

    reOrderedAttribInstances
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
    *                know some characteristics of these WEKA Instances, like
    *                the number of attributes, etc. NOTE: this method does not
    *                call `buildClassifier(forData)` on the newly created
    *                classifier.
    * @param wekaOptions the WEKA options to set in the created classifier.
    *                    This method initializes the classifier with some
    *                    reasonable values for the parameters, according to
    *                    the set of (BUPA) instances provided, and then
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


  /** Returns all the attribute names of a given WEKA data set of instances.
    *
    * @param dataSet the dataSet for which to get all the names of its attributes
    */

  def getAllAttribNames(dataSet: Instances): Array[String] = {

    val attribNames = new ArrayBuffer[String]()

    for (idx <- 0 until dataSet.numAttributes()) {
      attribNames += dataSet.attribute(idx).name
    }
    attribNames.toArray
  }


  /** It runs the main program, building the WEKA classifier from the BUPA
    * training data and applying it on the BUPA test data, and printing the
    * results.
    *
    * @param args ignored so far.
    */

  def main(args: Array[String]) {

    val trainingDataOrig = loadBupaDataSet("bupa_liver_disorders.csv")

    val trainingData = addRitisRatio(trainingDataOrig)

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
    val attribNamesBupa = getAllAttribNames(trainingData)

    printInferencesWithoutDrinks(wekaClassifier, attribNamesBupa)

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
    * case, that the biomakers of the liver are so moved that the liver has
    * symptoms of alcoholism without "drinks" influencing this state, and, in
    * these extreme cases, to find what are the boundary values of these
    * biomarkers in which this unfortunate situation occurs (see README of
    * this repository). Ie., these trees are pure or extreme cases of the
    * state of the liver, very healthy or very ill, where the "drinks"
    * attribute was analyzed in each step of the classification, but finally
    * was found to have no remarkable relevance for being in the kernel for
    * these extreme cases of health or illness.
    *
    * We could also have told WEKA to ignore the "drinks" attribute _before_
    * building the Random Forest classifier, but in this case, all the trees
    * would be without the "drinks" attribute, and we want those inferences
    * where "drinks" do not influence the result, but "drinks" was analyzed
    * and could have influenced each branch of the inference.
    *
    * @param wekaClassifier an instance of MyCustomRandomForestOpenBagOfTrees
    *                       on which the classifier has already been built.
    *
    * @param allAttribNames the array of all attribute names of the instances
    *                       classified by "wekaClassifier". (In this case, it
    *                       is the array of all BUPA attribute names.)
    */

  def printInferencesWithoutDrinks(wekaClassifier: AbstractClassifier,
                                   allAttribNames: Array[String]) {

    // The format of the print-out of the trees in a random forest in WEKA
    // is in tree-preoder (the root of the tree appears in the first line,
    // and lower subtrees appear indented in the following lines) and looks
    // like (for the BUPA datase of influence of alcoholism on the liver):
    //
    //    ... (print-out of WEKA random tree on BUPA dataset)...
    //    |   |   |   drinks >= 5.5
    //    |   |   |   |   sgot < 45
    //    |   |   |   |   |   drinks < 13.5
    //
    // where each "|" or (comparitive-expression) creates a new branch in the
    // tree. We are going to parse (or filter) this tree transversal.
    //
    // What we want is to prune those subtrees which have "drinks" in it,
    // because we want to see WEKA's inferences on the BUPA alcoholism dataset
    // where the inference is not affected by the "drinks", ie., very healthy
    // cases where "drinks" hasn't affected the liver, or very sick cases where
    // the liver is so affected by alcoholism that that the number of "drinks"
    // no longer has any effect on its biomarkers.

    // we'll do this task only for the Random Forest classifier we use
    val randomForest = wekaClassifier.asInstanceOf[MyCustomRandomForestOpenBagOfTrees]

    val rootSubtreeToPrune = "drinks"     // prune these subtrees under "drinks"

    val treesRandomForest = randomForest.getTrees()
    for ( (strReprTree, treeIdx) <- treesRandomForest.zipWithIndex ) {
      // first version of the code, this needs to be fixed: we need to implement
      // a stack automata which scans every line of the tree and see if this
      // line has a "drinks" token and in what position (tree-level):
      //     if it does, then to skip all the following lines belonging to
      //                      this same subtree
      // and
      //     if this line doesn't have "drinks", then push it in the stack, and
      //                                              continue parsing this subtree

      var previousDrinksLevel = -1   // what is the current, highest subtree
                                     // that is under the influence of "drinks"
      var thisStringWasARandomTree = false
      var thisRandomTreeHasBeenPrinted = false

      for ( lineTreeLevel <- strReprTree.split("\n") ) {
        // we need to check this new line (tree-level) whether it has or not
        // the "drinks" attribute in it (we are interested only in those
        // WEKA statistical inferences where "drinks" was discarded.

        val branchedAttrib = "\\b[A-Za-z_][A-Za-z0-9_]*\\b".r findFirstMatchIn lineTreeLevel

        if (branchedAttrib.isDefined) {    // there was a reg-exp match
           val levelAttribToken = branchedAttrib.get
           if ( levelAttribToken.matched == rootSubtreeToPrune ) {
             // The attribute in this level of the inference tree is "drinks".
             // We need to ignore this line and record at what tree level this
             // "drinks" has been inferred, so all its subtrees are ignored
             // (pruned), as being under this node of "drinks"

             thisStringWasARandomTree = true    // this set of lines was a Random Tree
             // at what character index in the line this "drinks" started
             val currPosDrink = levelAttribToken.start
             // See if there was a previous "drinks" seen in a higher tree node
             // than this one, or to a lower level, ie., if we were already under
             // a "drinks" subtree
             if ( previousDrinksLevel == -1 ) {
               previousDrinksLevel = currPosDrink  // we weren't in a "drinks" subtree
             } else if ( currPosDrink < previousDrinksLevel ) {
               // we previously were under a "drinks" subtree, but at a level
               // farthest from the root of the tree because "currPosDrink" is
               // less than the old one "previousDrinksLevel".
               // This means a new subtree has been found at "currPosDrink"
               previousDrinksLevel = currPosDrink
             }
           } else if (allAttribNames.indexOf(levelAttribToken.matched) != -1) {
             // it is not "drinks" but another attribute in the BUPA dataset
             thisStringWasARandomTree = true    // this set of lines was a Random Tree

             val currPosAttrib = levelAttribToken.start
             // This current attribute is under a drink subtree if:
             //   previousDrinksLevel != 1 and previousDrinksLevel < currPosAttrib
             if ( previousDrinksLevel != -1 &&
                  currPosAttrib <= previousDrinksLevel )
                // this "currPosAttrib" is in the same or higher level subtree
                // than the old "previousDrinksLevel", so it finishes that
                // previous "drinks" statistical inference subtree
                previousDrinksLevel = -1    // cleared the subtree indicator
             }
             if ( previousDrinksLevel == -1 ) { // we aren't currently under a
                       // statistical inference subtree related with "drinks"
               println(lineTreeLevel)
               thisRandomTreeHasBeenPrinted = true
             }
           } else {
             // some other statistical summary line given by WEKA about this tree
             println(lineTreeLevel)
           }
        }

      if (thisStringWasARandomTree) {
        if (thisRandomTreeHasBeenPrinted)
          println(f"---- Finished reporting RandomTree $treeIdx with 'drinks' subtrees pruned")
        else
          println(f"---- Skipped reporting RandomTree $treeIdx")
      }
    }
  }


  /** Inherits in Scala from the weka.classifiers.trees.RandomForest class in Java
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
          // the WEKA classifier generated
          Array("")
        else {
          // m_bagger is an object of the class weka.classifiers.meta.Bagging,
          // but this class doesn't give access to its protected
          // "m_classifiersCache":
          //    protected java.util.List<weka.classifiers.Classifier> m_classifiersCache;
          // To access this protected "m_classifiersCache" member in "m_bagger",
          // we split the String representation that "m_bagger" gives from its
          // protected "m_classifiersCache"

          m_bagger.toString().split("(?sm)^RandomTree$")
        }
      }
    }

}

