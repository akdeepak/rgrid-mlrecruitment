1. Run the flask application
2. method test generate eda - perform the exploratory data analysis to understand clean and prepare the data before building models
3. Model training - performs the model Training using transformer based model i.e, Bert
4. As part of the model training saving the Finetuned model for the research grid classification task is stored in the savedmodel ( note saved model tensor is huge to save in the git , please refer the feb20_rgrid jupyter notebook)
5. Predict function - loads the saved model and performs the prediction
6. Bert model has the limitation of handling tokens more than 512 tokens , I have solved the problem with sliding window approach for bert tokenization and performed the model training if the word token length is more than 512

Note: please refer the jupyter notebook - Feb20_rgrid to refer the complete ml development implementation.
