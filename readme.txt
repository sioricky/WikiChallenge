Run "loaddata.npy" to load "training.tsv" and generate training data in numpy format.
Run "preprocess_test.npy"  to generate test_user,test_feature and test_target for time sequential features
Run "preprocess_train.npy"  to generate train_user,train_feature and train_target for time sequential features
Run "cross validation.npy"  to do crossvalidation based on validation dataset using time sequential features
Run "prediction.npy" to do the prediction based on time sequential features
Run "log_data.npy" to transfer features and targets onto logarithm space
Run "prediction_log.npy" to do the prediction in logarithm space based on time sequential features
Run "case1_dataoutput.npy" and "feature_selection.npy" to generate the abstraction feature
Run "case1_predictor" to do the cross validation using validation dataset
Run "case2_datamine.npy" to do the prediction based on abstraction feature using "validation.tsv" and "validation_solution.tsv"
Run "case3_datamine.npy" to do the prediction based on abstraction features using "training.tsv","validation.tsv" and "validation_solution.tsv"
Run "combine_predictor" to do the prediction based on combination features