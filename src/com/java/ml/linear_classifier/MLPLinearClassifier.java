package com.java.ml.linear_classifier;

import java.io.File;
import java.io.IOException;

import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem.Evaluation;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MLPLinearClassifier {
	public static void main(String[] args) throws IOException, InterruptedException {
		String baseDir = "/Users/guilh/MLPLinearClassifier/data/training_data.csv";
		int seed = 123;
		double learningRate = 0.01;
		int batchSize = 50;
		int numEpochs = 30;
		int numInputs = 2;
		int numOutputs = 2;
		int numHiddenNodes = 20;
		File trainingData = new File(baseDir);
		File trainingDataEval = new File("/Users/guilh/MLPLinearClassifier/data/training_data_eval.csv");

		// Training the data
		// Configuring how the data is going to be loaded
		// Loading the training data
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(trainingData));
		DataSetIterator trainIterator = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);
		
		// Loading the test-evaluation data
		RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(trainingDataEval));
		DataSetIterator testIterator = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);
		
		// Building the network
		// Specifying the two layers (inputs) and the outputs
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.maxNumLineSearchIterations(1)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				//.learningRate(learningRate)
				//.updater(Updater.NESTEROVS)
				//.momentum(0.9)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numInputs)
						.nOut(numOutputs)
						.weightInit(WeightInit.XAVIER)
							.activation(Activation.RELU)
							.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.weightInit(WeightInit.XAVIER)
							.activation(Activation.RELU)
							.weightInit(WeightInit.XAVIER)
							.nIn(numHiddenNodes)
							.nOut(numOutputs)
							.build()
				)
				//.pretrain(false)
				.backpropType(BackpropType.Standard)
				.build();
				
		System.out.println(conf.toJson());
		
		// Building the model
		MultiLayerNetwork model =  new MultiLayerNetwork(conf);
			model.init();
			model.setListeners(new ScoreIterationListener(10));
			
			for (int n = 0; n < numEpochs; n++) {
				model.fit(trainIterator);
			}
			
			// Evaluating the model
			System.out.println("Evaluating model..........");
			org.nd4j.evaluation.classification.Evaluation evaluation = new org.nd4j.evaluation.classification.Evaluation(numOutputs);
			while (testIterator.hasNext()) {
				DataSet t = testIterator.next();
				INDArray features = t.getFeatures();
				INDArray labels = t.getLabels();
				INDArray predicted = model.output(features, false);
				evaluation.eval(labels, predicted);
			}
			System.out.println(evaluation.stats());
	}
}
