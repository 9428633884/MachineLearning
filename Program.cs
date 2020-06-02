using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace MachineLearning
{
    class FeedbackTrainingData
    {
        [Column(ordinal: "0", name: "Label")]
        public bool IsGood { get; set; }

        [Column(ordinal: "1")]
        public string FeedbackText { get; set; }
    }

    class FeedbackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }

    class Program
    {
        static List<FeedbackTrainingData> trainingData = new List<FeedbackTrainingData>() ;

        static List<FeedbackTrainingData> testData = new List<FeedbackTrainingData>();

        static void LoadTestData()
        {
            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "good",
                IsGood = true
            });

            testData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad",
                IsGood = false
            });
        }

        static void LoadTrainingData()
        {
            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "this is good",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "this is horrible",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "this is very average",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad horrible",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "well ok ok",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "shitty terrible",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "soooo nice",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "cool nice",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "sweet and nice",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "nice and good",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "very good",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "quiet average",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "god horrible",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "average and ok",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad and hell",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "this is nice but better can be done",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "bad bad",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "till now it looks nice",
                IsGood = true
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "worst",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "shit",
                IsGood = false
            });

            trainingData.Add(new FeedbackTrainingData()
            {
                FeedbackText = "oh this is shit",
                IsGood = false
            });
        }
        static void Main(string[] args)
        {
            // load the training data
            LoadTrainingData();

            var mlContext = new MLContext();

            IDataView dataView = mlContext.CreateStreamingDataView<FeedbackTrainingData>(trainingData);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedbackText", "Features").Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves:50, numTrees: 50, minDatapointsInLeaves: 1));

            var model = pipeline.Fit(dataView);

            LoadTestData();

            IDataView dataView1 = mlContext.CreateStreamingDataView<FeedbackTrainingData>(testData);

            var predictions = model.Transform(dataView1);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine("Feedback Prediction Model Accuracy : " + metrics.Accuracy);

            string strContinue = "y";

            while (strContinue == "y")
            {
                Console.WriteLine("\nEnter Feedback String:");

                string feedbackString = Console.ReadLine().ToString();

                var predictionFunction = model.MakePredictionFunction<FeedbackTrainingData, FeedbackPrediction>(mlContext);

                var feedbackInput = new FeedbackTrainingData();

                feedbackInput.FeedbackText = feedbackString;

                var feedbackPredicted = predictionFunction.Predict(feedbackInput);

                if(feedbackPredicted.IsGood)
                    Console.WriteLine("+ve feedback");
                else
                    Console.WriteLine("-ve feedback");

                Console.WriteLine("\nContinue ? ( y or n )");

                strContinue = Console.ReadLine().ToString();
            }
        }
    }
}
