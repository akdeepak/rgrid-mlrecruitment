# Deepak understanding ## 
# 1. Data understanding, Data exploration techniques EDA (Exploratory Data Analysis)
# 2. report generation with text length and token length
# 3. Check for the duplicate values and check whether the duplicates class labels are same, If the class labels are
# different for the duplicates then there would be label noise or label inconsistency.
# 4. identify the train test validation for each class labels to train the model.
# 5. custom Bert model to fine tune the model to identify the specific model.

# import pandas as pd 
# import sweetviz as sv

# df = pd.read_csv('data/trials.csv')

# # Generate the report
# report = sv.analyze(df)
# # To display the report in a Jupyter Notebook:
# report.show_notebook()
# # To save the report as an HTML file:
# report.show_html('rgrid_sweetviz.html')


import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import BertTokenizer


import os
os.environ['http_proxy'] ='http://proxy.novartis.net:2011'
os.environ['https_proxy'] ='http://proxy.novartis.net:2011'
os.environ['HTTP_PROXY'] ='http://proxy.novartis.net:2011'
os.environ['HTTPS_PROXY'] ='http://proxy.novartis.net:2011'

from nltk.tokenize import word_tokenize
# nltk.download('punkt')
df = pd.read_csv('data/trials.csv')
df["text_length"] = df["description"].apply(len)
df["word_count"] = df["description"].apply(lambda x: len(x.split()))
df['tokens'] = df['description'].apply(word_tokenize)

# Convert labels to numerical format
label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label"] = df["label"].map(label_mapping)

profile = ProfileReport(df, title="Profiling Report")

profile.to_file("your_report.html")

# duplicates = df[df.duplicated(subset=["description"], keep=False)]

# # Display duplicates with the corresponding labels
# print(duplicates[["description", "label"]])

# duplicate_counts = df[df.duplicated(subset=["description"], keep=False)].groupby(["description", "label"]).size().reset_index(name='count')

# # Display the counts
# print(duplicate_counts)



# Splitting dataset (70-15-15) while preserving category distribution
train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

# Check the distribution after splitting
# print(train["label"].value_counts(normalize=True))
# print(val["label"].value_counts(normalize=True))
# print(test["label"].value_counts(normalize=True))
# print(len(train))
# print(len(val))
# print(len(test))
# #verify the dataset is split correctly 

# train_category_counts = train["label"].value_counts()
# print(train_category_counts)
# val_category_counts = val["label"].value_counts()
# print(val_category_counts)
# test_category_counts = test["label"].value_counts()
# print(test_category_counts)

# Tokenize texts
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize texts
train_encodings = tokenizer(train_encodings, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_encodings, truncation=True, padding=True, max_length=512)