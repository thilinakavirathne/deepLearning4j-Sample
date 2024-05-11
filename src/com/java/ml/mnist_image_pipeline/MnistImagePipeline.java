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
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

public class MnistImagePipeline {
	private static Logger log = org.slf4j.LoggerFactory.getLogger(MnistImagePipeline.class);
	
	public static void main(String[] args) throws IOException {
		int height = 28;
		int width = 28;
		int channels = 1;
		int rngseed = 123;
		Random randNumGen = new Random(rngseed);
		int batchsize = 1;
		int outputNum = 10;
		
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
		recordReader.setListeners(new LogRecordListener());
		
		// DataSet Iterator
		
		DataSetIterator dsIter = new RecordReaderDataSetIterator(recordReader, batchsize, 1, outputNum);
		
		// Scaling pixel values to 0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		scaler.fit(dsIter);
		dsIter.setPreProcessor(scaler);
		
		for (int i = 0; i <= 3; i++) {
			DataSet ds = dsIter.next();
			System.out.println(ds);
			System.out.println(dsIter.getLabels());
		}
	}
}
