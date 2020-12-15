#!/usr/bin/env wolframscript
fileNum = 100;
testRatio = .2;
testNum = Floor[fileNum testRatio];
Print["fileNum: ", fileNum, ", testRatio: ", testRatio];
SetDirectory[FileNameJoin[{ParentDirectory[], "dataset"}]];
Print["Directory: ", Directory[]];

labels = FileNames[];
enc := PeriodogramArray[AudioChannelMix[AudioResample[#, 8000], "Mono"], 1024]&
Print["Loading dataset..."];
data = Thread[(enc[Import[#]]&) /@ RandomSample[FileNames[RegularExpression[".*\\.wav$"], #], fileNum]-> #]& /@ labels;
{test, train} = RandomSample /@ Flatten /@ Transpose[TakeDrop[#, testNum]& /@ data];

Print["Running classifier..."];
myClassify[model_, time_, progress_] :=
Module[{c}, 
c = Classify[train, Method->model, TimeGoal->time, PerformanceGoal->"Quality", TrainingProgressReporting -> progress];
Print[model, " Accuracy: ", ClassifierMeasurements[c, test, "Accuracy" ]];
];

myClassify[#, 120, None]& /@ {"NeuralNetwork", "SupportVectorMachine", "RandomForest", "LogisticRegression"};
