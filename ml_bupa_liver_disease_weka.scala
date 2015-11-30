#!/usr/bin/env scala -deprecation

// You need to have your CLASSPATH variable including the
//      weka.jar 
// file. Eg.
//      export CLASSPATH="$CLASSPATH:/path/to/weka.jar"


import scala.util.Random
import java.io.File

import weka.core.converters.CSVLoader
import weka.core.Utils.splitOptions
import weka.classifiers.Evaluation
import weka.core.{Instance, Instances}

import weka.classifiers.functions.MultilayerPerceptron


object WekaClassifierOnBupaAlcoholism {
  def main(args: Array[String]) {

    // load the BUPA liver disorders instances (local CSV file)
    // This dataset is available in R (at least) in the
    // Kernel Distance Weighted Discrimination package:
    // https://cran.r-project.org/web/packages/kerndwd/kerndwd.pdf
    //      install.packages("kerndwd")
    //      require(kerndwd)
    //      data(BUPA)

    var wekaCvsLoader = new CSVLoader()

    println("Reading training set...")
    wekaCvsLoader.setSource(new File("bupa_liver_disorders.csv"))
    val trainingData: Instances = wekaCvsLoader.getDataSet()
    println("Read done.")

    // set as the objective variable of the classifier the last attribute
    trainingData.setClassIndex(trainingData.numAttributes() - 1)

    val sizeTrainingData = trainingData.numInstances()
    val rnd = new Random()
    val randomPos = rnd.nextInt(sizeTrainingData)
    val testInstance = trainingData.instance(randomPos)
    val testInstances = new Instances(trainingData, 1)
    testInstances.add(testInstance)
    trainingData.delete(randomPos)
    val eval = new Evaluation(trainingData)

    val multiLayerPerceptr = new MultilayerPerceptron()
    // multiLayerPerceptr.setOptions(splitOptions(optionsString))
    // don't do a normalization of the covariates
    multiLayerPerceptr.setNormalizeAttributes(false)
    multiLayerPerceptr.setNominalToBinaryFilter(false) 
    multiLayerPerceptr.setValidationSetSize(40)   // 40%
    multiLayerPerceptr.setHiddenLayers("t,t,t,t")
    val epochs = 10000
    multiLayerPerceptr.setTrainingTime(epochs)
    multiLayerPerceptr.setMomentum(0.1)
    multiLayerPerceptr.setAutoBuild(true)
    multiLayerPerceptr.setGUI(true)
    multiLayerPerceptr.buildClassifier(trainingData)

    val s = testInstances.toString()
    println("DEBUG: Random instance(s) to be inferred by the classifier:\n" +
            s + "\n")
    eval.evaluateModel(multiLayerPerceptr, testInstances)
    System.out.println(eval.toSummaryString("\nResults\n======\n", false))
  }
}
