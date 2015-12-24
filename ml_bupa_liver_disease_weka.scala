#!/usr/bin/env scala -deprecation


// You need to have your CLASSPATH variable including the
//      weka.jar
// file. Eg.
//      export CLASSPATH="$CLASSPATH:/path/to/weka.jar"
// The WEKA library is Copyright (C) since 1999 by the University of Waikato,
//                     Hamilton, New Zealand


import scala.util.Random
import scala.collection.mutable.{ArrayBuffer, Stack}
import scala.util.matching._
import java.io.File

import weka.core.converters.CSVLoader
import weka.core.Utils.splitOptions
import weka.classifiers.Evaluation
import weka.core.{Instance, Instances}
import weka.filters.unsupervised.attribute.AddExpression
import weka.filters.unsupervised.attribute.Reorder
import weka.filters.MultiFilter
import weka.filters.unsupervised.instance.SubsetByExpression
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
    * @param args the command-line arguments. Only "--dump" accepted so far,
    *             which is a request to print the BUPA subset of samples
    *             which statistically support, or hint, about each leaf in
    *             the random tree -printed after each leaf is printed.
    */

  def main(args: Array[String]) {

    // whether to dump the leaves of the random trees determined by the
    // classification
    var dumpLeavesTree = false

    if (args.size >= 1 && args(0) == "--dump") dumpLeavesTree = true

    // read the BUPA CSV dataset, calculate the De Ritis Ratio, and find
    // the test set
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

    val instancesToDump = if (dumpLeavesTree) Some(trainingData) else None

    printInferencesWithoutDrinks(wekaClassifier, attribNamesBupa, Array("drinks"),
                                 instancesToDump)

    val s = testInstances.toString()
    println("DEBUG: Random instance(s) to be inferred by the classifier:\n" +
            s + "\n")
    eval.evaluateModel(wekaClassifier, testInstances)

    println(eval.toSummaryString("\nResults\n======\n", false))
  }

  /** truncates the stack at the desired length
    *
    * @param stack the stack to truncate
    *
    * @param desiredLength the desired length to leave the stack at
    */

  def truncateStackAtLength[T](stack: Stack[T], desiredLength: Int): Stack[T]
    = {
        // println(f"DEBUG: before truncation at length $desiredLength: " + stack.mkString(" -- "))
        if (stack.length > desiredLength) {
          if (desiredLength <= 0) {
            new Stack[T]
          } else {
            // drops all the elements in stack that are in excess of desiredLength
            stack.drop(stack.length - desiredLength)
          }
        } else {
          stack
        }
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
    *
    * @param attribNamesToPrune the array of attribute names whose sub-trees
    *                       to prune from the report in standard-output of the
    *                       trees. (In this case, it is an array with some
    *                       BUPA attribute names to be pruned from the report,
    *                       e.g., "drinks" to prune its subtrees where it was
    *                       needed for the inference.)
    *
    * @param instances the instances that were used to build this classifier.
    *                  This parameter is optional, and if it is given, then it
    *                  means to report the subset of instances under the leaves
    *                  of the trees as they are printed out.
    */

  def printInferencesWithoutDrinks(wekaClassifier: AbstractClassifier,
                                   allAttribNames: Array[String],
                                   attribNamesToPrune: Array[String],
                                   instances: Option[Instances]) {

    // we'll do this task only for the Random Forest classifier we use
    val randomForest = wekaClassifier.asInstanceOf[MyCustomRandomForestOpenBagOfTrees]

    val treesRandomForest = randomForest.getTrees()
    for ( (strReprTree, treeIdx) <- treesRandomForest.zipWithIndex ) {

      // process this WEKA random tree
      val reportResult =
        printATreePruningSomeAttribs(strReprTree, allAttribNames, attribNamesToPrune, instances)

      if (reportResult == 1) {
        println(f"---- Skipped reporting RandomTree $treeIdx")
      } else if (reportResult == 2) {
        val subtreesPruned = attribNamesToPrune.mkString(", ")
        println(f"---- Finished reporting RandomTree $treeIdx pruning trees needing any of the attribs: $subtreesPruned")
      }

    }
  }


  /** Does the internal work for "printInferencesWithoutDrinks()". This
    * method receives the WEKA representation of a random tree on the BUPA
    * dataset of influence of alcoholism on the liver, and reports to
    * the standard-output only those statistical inferences which don't have
    * the attribute "drinks" in it as "printInferencesWithoutDrinks()" was
    * requested to do. In general form, this method only prints those
    * inferences which don't have any of the attributes in the parameter
    * "attribNamesToPrune".
    *
    * @param strReprRandomTree a multi-line string with the representation
    *                          given by WEKA of a random tree.
    *
    * @param allAttribNames the array of all attribute names of the instances
    *                       classified by this random tree. (In this case, it
    *                       is the array of all BUPA attribute names.)
    *
    * @param attribNamesToPrune the array of attribute names whose sub-trees
    *                       to prune from the report in standard-output of the
    *                       trees. (In this case, it is an array with some
    *                       BUPA attribute names to be pruned from the report,
    *                       e.g., "drinks" to prune its subtrees where it was
    *                       needed for the inference.)
    *
    * @param instances the instances that were used to build this classifier.
    *                  This parameter is optional, and if it is given, then it
    *                  means to report the subset of instances under the leaves
    *                  of the trees as they are printed out.
    *
    * @return an integer with the status of the report:
    *                  value 0 if no tree was parsed -ie., this string contained
    *                  WEKA statistics lines only, it did did not contain any
    *                  attribute from the instances with the "|"-delimited format,
    *                  etc-;
    *                  value 1, if a tree was parsed but no leaf was printed
    *                  because all were pruned (ie., no complete inference was
    *                  found that did not contain any of the attributes in
    *                  "attribNamesToPrune");
    *                  value 2, if at least one leaf of this random tree was
    *                  printed (ie., there was at least one complete inference
    *                  from the root to a leaf which was clean of all attributes
    *                  in "attribNamesToPrune")
    */

  def printATreePruningSomeAttribs(strReprRandomTree: String,
                                   allAttribNames: Array[String],
                                   attribNamesToPrune: Array[String],
                                   instances: Option[Instances]): Int =
    {

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
      // where each "|" or (comparative-expression) creates a new branch in the
      // tree. The relationship between the starts of the tokens (attribute
      // names) in the line string and the level of the tree they are at in
      // WEKA, is, if both are 0-based -ie., the root of the tree is at level 0:
      //       index-start-token-in-string = 4 * its-level-in-tree
      // We are going to parse (or filter) this tree transversal.
      //
      // What we want is to prune those subtrees which have "drinks" in it (or
      // in general, to prune all subtrees under any attribute in the array
      // "attribNamesToPrune" -we'll talk about "drinks" in this respect)
      // because we want to see WEKA's inferences on the BUPA alcoholism dataset
      // where the inference is not affected by the "drinks", ie., very healthy
      // cases where "drinks" hasn't affected the liver, or very sick cases where
      // the liver is so affected by alcoholism that that the number of "drinks"
      // no longer has any effect on its biomarkers.

      var previousDrinksLevel = -1   // what is the string position of the current,
                                     // highest subtree that is under the influence
                                     // of "drinks" (we keep track only of the highest
                                     // current subtree that we are pruning)
      var logicalConditionsStack = new Stack[String]()  // the logical conditions
                                                // that has been collected in the
                                                // traversal of the WEKA tree so far
      var thisStringWasARandomTree = false
      var aLeafInTheTreeHasBeenPrinted = false

      for ( lineTreeLevel <- strReprRandomTree.split("\n") ) {
        // we need to check this new line (tree-level) whether it has or not
        // the "drinks" attribute in it (we are interested only in those WEKA
        // statistical inferences where "drinks" was not necessary for them.

        val branchedAttrib = "\\b[A-Za-z_][A-Za-z0-9_]*\\b".r findFirstMatchIn lineTreeLevel

        if (branchedAttrib.isDefined) {    // there was a reg-exp match
           val levelAttribToken = branchedAttrib.get
           // at what character index in the line this attribute starts
           val currPosAttrib = levelAttribToken.start

           if ( attribNamesToPrune.indexOf(levelAttribToken.matched) != -1 ) {
             // The attribute in this level of the inference tree is "drinks"
             // or another in "attribNamesToPrune" whose subtree should also be
             // pruned from the report.
             // We need to ignore this line and record at what tree level this
             // "drinks" has been inferred, so all its subtrees are ignored
             // (pruned), as being under this node of "drinks"

             thisStringWasARandomTree = true    // this set of lines was a Random Tree

             // See if there was a previous "drinks" seen in a higher tree node
             // than this one, or to a lower level, ie., if we were already under
             // a "drinks" subtree
             var thisLineStartsANewSubtreeToPrune = false
             if ( previousDrinksLevel == -1 ) {
               thisLineStartsANewSubtreeToPrune = true  // we weren't in a "drinks"
               previousDrinksLevel = currPosAttrib      // subtree: now we found
                                                        // one and start pruning it
             } else if ( currPosAttrib < previousDrinksLevel ) {
               // we previously were under a "drinks" subtree, but at a level
               // farthest from the root of the tree because "currPosAttrib" is
               // less than the old one "previousDrinksLevel".
               // This means a new subtree to prune has been found at "currPosAttrib"

               thisLineStartsANewSubtreeToPrune = true
               previousDrinksLevel = currPosAttrib  // update "previousDrinksLevel" to
                                                    // this subtree

               if (instances.isDefined) {
                 // We need to keep track of the logicalConditionsStack.
                 // The relationship between the beginning of the token,
                 // "currPosAttrib", and the level in the WEKA random tree,
                 // since both are 0-indexed, is:
                 //     currPosAttrib == ( 4 * levelInTree )
                 // println(f"sync stack because of start of pruning at currPosAttrib = $currPosAttrib")
                 val levelStack = currPosAttrib/4 - 1   // -1 because we want
                                                        // anything at this the
                                                        // current level to be
                                                        // pruned in the stack
                 logicalConditionsStack =
                   truncateStackAtLength(logicalConditionsStack, levelStack)
               }
             }
             if (thisLineStartsANewSubtreeToPrune) {
               println(lineTreeLevel.substring(0, currPosAttrib) +
                       "[ ... pruning this subtree because '" +
                       levelAttribToken.matched +
                       "' is here at its root ...]")
             }
           } else if (allAttribNames.indexOf(levelAttribToken.matched) != -1) {
             // it is not "drinks" but another attribute in the BUPA dataset
             thisStringWasARandomTree = true    // this set of lines was a Random Tree

             // This current attribute is under a drink subtree if:
             //   previousDrinksLevel != 1 and previousDrinksLevel < currPosAttrib
             if ( previousDrinksLevel != -1 &&
                  currPosAttrib <= previousDrinksLevel ) {
               // this "currPosAttrib" is in the same or higher level subtree
               // than the old "previousDrinksLevel", so it finishes the pruning
               // set by that previous "drinks" statistical inference subtree
               previousDrinksLevel = -1    // cleared the subtree indicator,
                                           // so we stop pruning and start
                                           // transversing the tree again
             }
             if ( previousDrinksLevel == -1 ) { // we aren't currently under a
                        // statistical inference subtree related with "drinks"

               println(lineTreeLevel)

               // Try to see if the line we have just printed was a branch internal
               // node in the tree, or a leaf
               val wekaTreeLeafPatt = """ : [1-9][0-9]* \([1-9][0-9]*/[0-9]*\)""".r
               val wekaTreeLeaf = wekaTreeLeafPatt findFirstMatchIn lineTreeLevel

               if (wekaTreeLeaf.isDefined) {
                 aLeafInTheTreeHasBeenPrinted = true
               }

               if (instances.isDefined) {
                 // if the instances parameter was given to this method, we need
                 // to print as well the subset of instances that fall under
                 // a certain WEKA leaf in the Random Tree. In order to find this
                 // subset, we need to keep track of the conditions stack as we
                 // transverse the WEKA random tree string in pre-order

                 // resync stack if necessary
                 val levelStack = currPosAttrib/4    // both levelStack and currPosAttrib are 0-based
                 // println(f"resync stack during visiting node at index $currPosAttrib, tree-level $levelStack")
                 logicalConditionsStack =
                   truncateStackAtLength(logicalConditionsStack, levelStack)

                 // Try to extract the logical condition expressed in this line.
                 // This depends if this line was a leaf or a branch in the WEKA
                 // random tree

                 if ( ! wekaTreeLeaf.isDefined ) {
                   // this is a branch in the WEKA tree: the logical condition in
                   // this line is from the attribute token till the end of line
                   val conditionInThisBranch = lineTreeLevel.substring(currPosAttrib)
                   // push this logical condition to the top of the condition stack,
                   // enclosed by parentheses, "( ... )"
                   logicalConditionsStack.push(f"( $conditionInThisBranch )")
                 } else {
                   // this is a leaf in the WEKA random tree: the logical condition
                   // in this line is from the attribute token till the leaf
                   // specification suffix starts.
                   // (Since this is a leaf, as an optimization we don't push this
                   //  leaf condition into the "logicalConditionsStack", for it would
                   //  immediately be pop-ed from it, for this is a leaf.)
                   val indexLeafSpecInLine = wekaTreeLeaf.get.start
                   val conditionInThisLeaf = lineTreeLevel.substring(currPosAttrib,
                                                                     indexLeafSpecInLine)
                   var wekaSubsetExpresssion = ""
                   if (logicalConditionsStack.length == 0) {
                     wekaSubsetExpresssion = f"( $conditionInThisLeaf )"
                   } else {
                     // get a reverse copy of the stack in order to get the logical
                     // conjunction "AND ..." of its conditions in human-friendly
                     // root-down-to-leaf format, not in leaf-up-to-root (polish)
                     // machine format. (This order is for reporting to the user,
                     // WEKA's SubsetByExpression filter is fine with both orders.)

                     val logicalConditionsArray = logicalConditionsStack.reverse
                     wekaSubsetExpresssion = logicalConditionsArray.mkString(" and ") +
                                               " and ( " + conditionInThisLeaf + " )"
                   }

                   reportInstancesWhichSupportThisInference(instances.get,
                                                            wekaSubsetExpresssion,
                                                            allAttribNames,
                                                            indexLeafSpecInLine + 4,
                                                            attribNamesToPrune)
                 }
               }
             }
           }
         } else {
             // some other statistical summary line given by WEKA about this tree
             println(lineTreeLevel)
         }
       }

       if (! thisStringWasARandomTree) {
         // this "strReprRandomTree" passed by the caller, the parser determined that
         // it contained totally some WEKA statitics, but no random tree (ie., this
         // string representation did not contain any attribute from the instances
         // with the "|"-delimited format, etc)

         return 0

       } else {
         // this "strReprRandomTree" contained a WEKA random tree: was at least one
         // leaf (complete inference) printed by this parser, or no leaf was
         // printed because all inferences contained attributes to prune

         return if (aLeafInTheTreeHasBeenPrinted) 2 else 1
       }
    }



  /** prints to standard-output the subset of samples which support a
    * statistical inference found by WEKA in a tree of a ramdom forest.
    * The print-out is in CSV format, with a header line and an
    * indentation prefix in the report to alineate it with the leaf in
    * the tree that is reporting the subset of samples about
    *
    * @param universeSamples a WEKA instances with the universe set of
    *                        instances from which the random-forest was
    *                        trained (built).
    *
    * @param inferenceConditions a string with the inference logical
    *                            conditions, which is a logical conjunction
    *                            "... AND ..." of the conditions on the
    *                            attribute-names in the random tree from
    *                            its root till this inferred leaf under
    *                            which we are reporting
    *
    * @param allAttribNames the array of all attribute names in the
    *                       universeSamples
    *
    * @param indentation    the indentation to leave to the left of this
    *                       print-out in order to alineate it under its
    *                       leaf in the random tree
    *
    * @param attribNamesNotExpected this is an array with the attribute
    *                               names which are not expected in the
    *                               logical "inferenceConditions", as to
    *                               verify that these attributes are not
    *                               in this logical condition representing
    *                               this inference
    */

  def reportInstancesWhichSupportThisInference(universeSamples: Instances,
                                               inferenceConditions: String,
                                               allAttribNames: Array[String],
                                               indentation: Int,
                                               attribNamesNotExpected: Array[String] ) {

    // println(f"DEBUG: received WEKA conditions for this subset: $inferenceConditions")

    // we don't need to do this, since we expect that no attribute name in
    // "attribNamesNotExpected": it is just to ensure this state, although this method
    // can work ignoring this "attribNamesNotExpected", that's a logic imposed by a
    // client of this method
    for ( attribPruned <- attribNamesNotExpected ) {
      val regExpAttribPruned = """\\b""" + Regex.quote(attribPruned) + """\\b""" r
      val firstPosition = regExpAttribPruned findFirstMatchIn inferenceConditions
      // attribPruned can't be in the "inferenceConditions", so:
      assert(! firstPosition.isDefined)
    }

    // substitute all attribute names in "inferenceConditions" by their corresponding
    // "ATT<idx#>" tag that WEKA's SubsetByExpression instance filter expects
    var wekaInferenceConditions = inferenceConditions
    for ( (attribName, attribIdx) <- allAttribNames.zipWithIndex ) {
      val regExpAttrib = ("\\b" + Regex.quote(attribName) + "\\b")
      val wekaAttrIdx = attribIdx + 1
      wekaInferenceConditions =
        wekaInferenceConditions.replaceAll(regExpAttrib, f"ATT$wekaAttrIdx")
    }
    // println(f"DEBUG: replaced attribute names by ATT# indexes: $wekaInferenceConditions")

    // find the WEKA SubsetByExpression of the samples which support this inference by
    // random tree
    val samplesFilter = new SubsetByExpression()
    samplesFilter.setExpression(wekaInferenceConditions)
    samplesFilter.setInputFormat(universeSamples)
    val samplesWhichSupportThisInference = Filter.useFilter(universeSamples, samplesFilter)

    // report these subset of samples which support this inference by the random tree
    val arffSections = samplesWhichSupportThisInference.toString.split("(?sm)^@data$")
    val preffixIndent = " " * indentation
    // write the logical condition in the random tree of this statistical inference
    println(preffixIndent + inferenceConditions)
    // write the CSV header line for this subset of samples, indented
    print(preffixIndent)
    for ( (attribName, attribIdx) <- allAttribNames.zipWithIndex ) {
      print(if (attribIdx > 0) f",$attribName" else attribName)
    }
    println()
    for ( csvLine <- arffSections(1).split("\n") if (! csvLine.isEmpty) ) {
      println(preffixIndent + csvLine)
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

