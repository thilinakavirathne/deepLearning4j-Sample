package com.java.ml.storm_reports_record_reader;

import java.util.Date;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.inmemory.InMemoryRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

public class StormReportsRecordReader {

	public static void main(String[] args) {
		int numOfLinesToSkip = 0;
		String delimiter = ",";

		/*
		 * Specifying the root directory
		 */

		String baseDir = "/Users/guilh/MLPLinearClassifier/data/weather_report/";
		String fileName = "reports.csv";
		String inputPath = baseDir + fileName;
		String timeStamp = String.valueOf(new Date().getTime());
		String outputPath = baseDir + "reports_processed_" + timeStamp;

		/*
		 * Data set table columns are datetime, severity, location, country, state,
		 * latitude, longitude, comment, type
		 */

		org.datavec.api.transform.schema.Schema inputDataSchema = new org.datavec.api.transform.schema.Schema.Builder()
				.addColumnsString("datetime", "severity", "location", "country", "state").addColumnDouble("latitude")
				.addColumnDouble("longitude").addColumnsString("comment")
				.addColumnCategorical("type", "TOR", "WIND", "HAIL").build();

		/*
		 * Defining a transform process to extract lat and lon and also transform type
		 * from one of the three strings to either 0, 1 or 2
		 */

		TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
				.removeColumns("datetime", "severity", "location", "country", "state", "comment")
				.categoricalToInteger("type").build();

		/*
		 * Stepping through and printing the before and after Schema
		 */

		int numActions = tp.getActionList().size();
		for (int i = 0; i < numActions; i++) {
			System.out.println("\n\n=======================");
			System.out.println("--- Schema after step " + i + " (" + tp.getActionList().get(i) + ")--");
			System.out.println(tp.getSchemaAfterStep(i));
		}

		SparkConf sparkConf = new SparkConf();
		sparkConf.setMaster("local[*]");
		sparkConf.setAppName("Storm Reports Record Reader Transform");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		/*
		 * Getting our data into a Spark RDD and transforming that Spark RDD using our
		 * transform process
		 */

		// Reading the data file
		JavaRDD<String> lines = sc.textFile(inputPath);
		// Converting it to Writable
		JavaRDD<List<Writable>> stormReports = lines.map(new StringToWritablesFunction(new InMemoryRecordReader(null)));
		// Running our transform process
		JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(stormReports, tp);
		// Converting the Writable back to String for export
		JavaRDD<String> toSave = processed.map(new WritablesToStringFunction(","));
		
		toSave.saveAsTextFile(outputPath);
		sc.close();
	}
}
