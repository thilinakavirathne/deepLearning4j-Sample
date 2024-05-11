package com.java.ml.mnist_image_pipeline;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.log4j.spi.LoggerFactory;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

public class MnistImagePipelineSave {
	private static Logger log = org.slf4j.LoggerFactory.getLogger(MnistImagePipelineSave.class);
	
	public static void main(String[] args) throws IOException {
		// The images are 20 x 28 grayscale
		// Grayscale implies single channel
		int height = 28;
		int width = 28;
		int channels = 1;
		int rngseed = 123;
		Random randNumGen = new Random(rngseed);
		int batchsize = 120;
		int outputNum = 10;
		int numEpochs = 15;
		
		// Defining the file paths
		File trainData = new File("/Users/guilh/MLPLinearClassifier/data/mnist_png/training");
		File testData = new File("/Users/guilh/MLPLinearClassifier/data/mnist_png/testing");

		// Defining the FileSplit(PATH, ALLOWED_FORMATS, random)
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

		// Extracting the parent path at the image label
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
		
		/*
		 * Initializing the record reader
		 * Adding a listener, to extract the name
		 */
		recordReader.initialize(train);
		//recordReader.setListeners(new LogRecordListener());
		
		// DataSet Iterator
		
		DataSetIterator dsIter = new RecordReaderDataSetIterator(recordReader, batchsize, 1, outputNum);
		
		// Scaling pixel values to 0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		scaler.fit(dsIter);
		dsIter.setPreProcessor(scaler);
		
		// Building our Neural Network
		log.info("BUILDING MODEL..............");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rngseed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				//.iterations(1)
				//.learningRate(0.006)
				//.updater(Updater.NESTEROVS)
				//.momentum(0.9)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(height * width)
						.nOut(100)
						.activation(Activation.RELU)
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nIn(100)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.build())
				//.pretrain(false).backdrop(true)
				.setInputType(InputType.convolutional(height, width, channels))
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		log.info("TRAINING MODEL..............");
		for (int i = 0; i < numEpochs; i++) {
			model.fit(dsIter);
		}
		
		log.info("EVALUATING MODEL..............");
		recordReader.reset();
		
		recordReader.initialize(test);
		DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchsize, 1, outputNum);
		scaler.fit(testIter);
		testIter.setPreProcessor(scaler);
		
		// Creating Eval object with 10 possible classes
		Evaluation eval = new Evaluation(outputNum);
		
		while (testIter.hasNext()) {
			DataSet next = testIter.next();
			INDArray output = model.output(next.getFeaturesMaskArray());
			eval.eval(next.getLabels(), output);
		}
		log.info(eval.stats());
	}
}
