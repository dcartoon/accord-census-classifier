using System;
using System.Data;
using System.IO;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Math;
using Accord.Statistics.Filters;
using AccordCensusClassifier.DataConversion;

namespace AccordCensusClassifier
{
	/// <summary>
	/// Example use of the Accord.Net C4.5 decision tree algorithm.  This draws primarily from the sample included
	/// in Accord.Net and the intro comment in C45Learning.cs
	/// 
	/// This code is designed to work with data from: http://archive.ics.uci.edu/ml/datasets/Census+Income
	/// that has also been run through a provided R script to convert the various strings(e.g., Farming-fishing)
	/// </summary>
	internal class Program
	{
		private static void Main(string[] args)
		{
			if (args.Length < 2)
			{
				Console.Out.WriteLine("Usage: AccordCensusClassifier.exe [training data] [test data]");
			}

			string pathToTrainingData = args[0];
			string pathToTestData = args[1];
			
			string[] columnNames = new string[]
				{
					"age", "workclass", "fnlwgt", "education", "education-num",
					"marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
					"hours-per-week", "native-country", "income"
				};

			string[] inputColumnNames = columnNames.Submatrix(0, 13);
			string outputColumn = columnNames[14];

			Console.Out.WriteLine("{0} - Loading data", DateTime.Now);

			//The codification step needs to know about all possible values, or you will get KeyNotFoundExceptions
			//when operating on new data
			DataTable allData = new DataTable("All");

			DataTable trainingData = new DataTable("Training");
			DataTable testData = new DataTable("Test");

			foreach (string columnName in inputColumnNames)
			{
				allData.Columns.Add(columnName);
				trainingData.Columns.Add(columnName);
				testData.Columns.Add(columnName);
			}

			allData.Columns.Add(outputColumn);
			trainingData.Columns.Add(outputColumn);
			testData.Columns.Add(outputColumn);

			String[] lines = File.ReadAllLines(pathToTrainingData);
			String[] testLines = File.ReadAllLines(pathToTestData);

			//This is optional but makes playing around with large training files a little quicker than manually
			//shortening the training files
			lines = lines.Submatrix(0, 10000);

			lines.ToList().ForEach(l => trainingData.Rows.Add(l.Split(',')));
			testLines.ToList().ForEach(l => testData.Rows.Add(l.Split(',')));
			lines.Concat(testLines).ToList().ForEach(l => allData.Rows.Add(l.Split(',')));

			Codification codebook = new Codification(allData); //Build the codifier using allData
			DataTable symbols = codebook.Apply(trainingData);
			DataTable testSymbols = codebook.Apply(testData);

			Console.Out.WriteLine("{0} - Done loading data", DateTime.Now);
			
			double[][] inputs = symbols.ToArray(inputColumnNames);
			double[][] testInputs = testSymbols.ToArray(inputColumnNames);

			int[] outputs = symbols.ToArray<int>(outputColumn);
			int[] testOutputs = testSymbols.ToArray<int>(outputColumn);
			
			// Create the Decision tree
			DecisionVariable[] attributes = DecisionVariable.FromCodebook(codebook, inputColumnNames);
			DecisionTree tree = new DecisionTree(attributes, 2);

			// Creates a new instance of the C4.5 learning algorithm
			C45Learning c45 = new C45Learning(tree);

			// Learn the decision tree
			double trainingError = c45.Run(inputs, outputs);
			Console.Out.WriteLine("{0} - Training error: {1}", DateTime.Now, trainingError);

			double testError = c45.ComputeError(testInputs, testOutputs);
			Console.Out.WriteLine("{0} - Test error: {1}", DateTime.Now, testError);
			
			//Console.Out.WriteLine("Building expression tree");
			//var expression = tree.ToExpression();
			//Console.Out.WriteLine("Done building expression tree");
			//var func = expression.Compile();

			//int i = 0;
			//Console.Out.WriteLine("Evaluating row[{0}]: {1}, From compiled tree: {2}", i, tree.Compute(inputs[i]),
			//	func(inputs[i]));
		}

		/// <summary>
		/// Turns out this doesn't work(get degenerate tree exceptions).  Most likely, this is because
		/// the discrete values really need to be mapped in a specific way, and this is what the 
		/// Codification class does
		/// </summary>
		private static void followingExample()
		{
			const string pathToTrainingData = "";
			
			string[] columnNames = new string[15]
				{
					"age", "workclass", "fnlwgt", "education", "education-num",
					"marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
					"hours-per-week", "native-country", "income"
				};

			string[] inputColumnNames = columnNames.Submatrix(0, 13);
			string outputColumn = columnNames[14];

			Console.Out.WriteLine("Reading data");

			DataTable trainingData = DelimitedTextReader.ReadFile(pathToTrainingData, ",", columnNames);

			// Perform classification
			C45Learning c45;
			string[] colNames;

			double[,] sourceMatrix = trainingData.ToMatrix<double>(out colNames);
			Console.Out.WriteLine("Matrix size: {0} x {1}", sourceMatrix.GetLength(0), sourceMatrix.GetLength(1));

			// Get only the input vector values
			double[][] inputs = sourceMatrix.Submatrix(null, 0, 13).ToArray();

			int[] outputs = sourceMatrix.GetColumn(14).ToInt32();

			DecisionVariable[] attributes =
				{
					new DecisionVariable("age", DecisionAttributeKind.Continuous),
					new DecisionVariable("workclass", DecisionAttributeKind.Discrete),
					new DecisionVariable("fnlwgt", DecisionAttributeKind.Continuous),
					new DecisionVariable("education", DecisionAttributeKind.Discrete),
					new DecisionVariable("education-num", DecisionAttributeKind.Discrete),
					new DecisionVariable("marital-status", DecisionAttributeKind.Discrete),
					new DecisionVariable("occupation", DecisionAttributeKind.Discrete),
					new DecisionVariable("relationship", DecisionAttributeKind.Discrete),
					new DecisionVariable("race", DecisionAttributeKind.Discrete),
					new DecisionVariable("sex", DecisionAttributeKind.Discrete),
					new DecisionVariable("capital-gain", DecisionAttributeKind.Continuous),
					new DecisionVariable("capital-loss", DecisionAttributeKind.Continuous),
					new DecisionVariable("hours-per-week", DecisionAttributeKind.Continuous),
					new DecisionVariable("native-country", DecisionAttributeKind.Discrete)
				};

			// Create the Decision tree
			DecisionTree tree = new DecisionTree(attributes, 2);

			// Creates a new instance of the C4.5 learning algorithm
			c45 = new C45Learning(tree);

			// Learn the decision tree
			double error = c45.Run(inputs, outputs);
			Console.Out.WriteLine("Training error: {0}", error);
		}
	}
}
