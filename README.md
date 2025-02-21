1. Run the flask application
2. method test generate eda - perform the exploratory data analysis to understand clean and prepare the data before building models
3. Model training - performs the model Training using transformer based model i.e, Bert
4. As part of the model training saving the Finetuned model for the research grid classification task is stored in the savedmodel ( note saved model tensor is huge to save in the git , please refer the feb20_rgrid jupyter notebook)
5. Predict function - loads the saved model and performs the prediction
6. Bert model has the limitation of handling tokens more than 512 tokens , I have solved the problem with sliding window approach for bert tokenization and performed the model training if the word token length is more than 512

Note: please refer the jupyter notebook - Feb20_rgrid to refer the complete ml development implementation.


Snapshot of model training with loss function.

  Epoch	Training Loss	Validation Loss	Accuracy	Precision	Recall	F1
1	No log	0.362238	0.872340	0.890221	0.872340	0.871036
2	No log	0.190488	0.929078	0.933560	0.929078	0.929106
3	No log	0.199945	0.929078	0.934021	0.929078	0.928807
TrainOutput(global_step=477, training_loss=0.3630702180682488, metrics={'train_runtime': 219.4375, 'train_samples_per_second': 17.308, 'train_steps_per_second': 2.174, 'total_flos': 999322705078272.0, 'train_loss': 0.3630702180682488, 'epoch': 3.0})

Evaluation Metrics: {'eval_loss': 0.19994480907917023, 'eval_accuracy': 0.9290780141843972, 'eval_precision': 0.9340210315804808, 'eval_recall': 0.9290780141843972, 'eval_f1': 0.9288065713283824, 'eval_runtime': 1.9433, 'eval_samples_per_second': 72.559, 'eval_steps_per_second': 9.263, 'epoch': 3.0}
