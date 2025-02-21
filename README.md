1. Run the flask application
2. method test generate eda - perform the exploratory data analysis to understand clean and prepare the data before building models
3. Using stratified data split to choose the dataset for the model training with the split of training of 80% , validation 10% and testing 10%
4. Model training - performs the model Training using transformer based model i.e, Bert
5. As part of the model training saving the Finetuned model for the research grid classification task is stored in the savedmodel ( note saved model tensor is huge to save in the git , please refer the feb20_rgrid jupyter notebook)
6. Predict function - loads the saved model and performs the prediction
7. Bert model has the limitation of handling tokens more than 512 tokens , I have solved the problem with sliding window approach for bert tokenization and performed the model training if the word token length is more than 512

Note: please refer the jupyter notebook - Feb20_rgrid to refer the complete ml development implementation.



 
Snapshot of model training with loss function.

 [477/477 03:12, Epoch 3/3]
Epoch	Training Loss	Validation Loss	Accuracy	Precision	Recall	F1
1	No log	0.312258	0.900709	0.915511	0.900709	0.902135
2	No log	0.351780	0.921986	0.932934	0.921986	0.922009
3	No log	0.266698	0.950355	0.953959	0.950355	0.950886
 [160/477 00:52 < 01:45, 3.01 it/s, Epoch 1/3]
Epoch	Training Loss	Validation Loss

 [18/18 02:31]
TrainOutput(global_step=477, training_loss=0.1570867062614649, metrics={'train_runtime': 193.1519, 'train_samples_per_second': 19.663, 'train_steps_per_second': 2.47, 'total_flos': 1332430273437696.0, 'train_loss': 0.1570867062614649, 'epoch': 3.0})

EVALUATION METRICS

 [18/18 00:01]
Evaluation Metrics: {'eval_loss': 0.2666977345943451, 'eval_accuracy': 0.950354609929078, 'eval_precision': 0.953959115561118, 'eval_recall': 0.950354609929078, 'eval_f1': 0.9508855506195933, 'eval_runtime': 2.0465, 'eval_samples_per_second': 68.898, 'eval_steps_per_second': 8.795, 'epoch': 3.0}

PREDICTION METRICS
Final Model Performance:
Accuracy: 0.9290
Precision: 0.9312
Recall: 0.9290
F1: 0.9292








Back up for my reference
 Epoch	Training Loss	Validation Loss	Accuracy	Precision	Recall	F1
1	No log	0.362238	0.872340	0.890221	0.872340	0.871036
2	No log	0.190488	0.929078	0.933560	0.929078	0.929106
3	No log	0.199945	0.929078	0.934021	0.929078	0.928807
TrainOutput(global_step=477, training_loss=0.3630702180682488, metrics={'train_runtime': 219.4375, 'train_samples_per_second': 17.308, 'train_steps_per_second': 2.174, 'total_flos': 999322705078272.0, 'train_loss': 0.3630702180682488, 'epoch': 3.0})

Evaluation Metrics: {'eval_loss': 0.19994480907917023, 'eval_accuracy': 0.9290780141843972, 'eval_precision': 0.9340210315804808, 'eval_recall': 0.9290780141843972, 'eval_f1': 0.9288065713283824, 'eval_runtime': 1.9433, 'eval_samples_per_second': 72.559, 'eval_steps_per_second': 9.263, 'epoch': 3.0}
