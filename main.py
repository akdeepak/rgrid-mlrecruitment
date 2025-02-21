from flask import Flask, jsonify, request
from typing import Literal
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from ydata_profiling import ProfileReport
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

class DataManager:
    df = pd.read_csv("data/trials.csv")  # Class-level DataFrame    return item

    @classmethod
    def get_data(cls):
        return cls.df


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the input text with padding, truncation, and max_length
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        # Flatten the tensors to remove unnecessary batch dimension (batch_size=1)
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        return encoding

# class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]
label_mapping = {i: label for i, label in enumerate([
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
])}

def preprocess_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


## TODO move to utils 

def preprocess_text(text):
    """
    Perform text preprocessing using NLTK.

    Steps:
    1. Lowercasing
    2. Tokenization
    3. Removing Punctuation & Special Characters
    4. Stopword Removal
    5. Lemmatization

    Returns:
    - Cleaned text as a string
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Convert tokens back to text
    cleaned_text = " ".join(tokens)
    return cleaned_text

def generate_eda_report(file_name: str) -> str:
    """
    Reads a CSV file, performs exploratory data analysis (EDA),
    and generates an HTML report.

    :param file_name: Path to the input CSV file.
    :return: Path of the saved HTML report.
    """
    # Define output file
    output_html = "trials_eda.html"

    # Read CSV file
    print(file_name)
    df = pd.read_csv(file_name)  
    print(df)
    # Add features
    df["text_length"] = df["description"].apply(len)
    df["word_count"] = df["description"].apply(lambda x: len(x.split()))
    df["tokens"] = df["description"].apply(lambda x: len(word_tokenize(x)))

    # Apply preprocessing
    df["cleaned_description"] = df["description"].apply(preprocess_text)
    df["cleaned_tokens"] = df["cleaned_description"].apply(lambda x: len(word_tokenize(x)))

    # Generate profile report
    profile = ProfileReport(df, title="EDA Report", explorative=True)

    # Save the profile report to an HTML file
    profile.to_file(output_html)
    
    print(f"EDA report generated: {output_html}")
    return output_html


def predict_bkp(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """

    # Reload the trained model from the checkpoint
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    trainer = Trainer(model=model, tokenizer=tokenizer)

    print(description)
    test_text = description
    test_encoding = tokenizer(test_text, truncation=True, padding=True, return_tensors="pt")

    # prediction_dataset = TextDataset(description, tokenizer)
     # Make predictions
    predictions, label_ids, metrics = trainer.predict(prediction_dataset)

    # Convert logits to predicted class labels (by taking argmax)
    predicted_labels = np.argmax(predictions)
    # Print the predicted labels
    print(f"Predicted Labels: {predicted_labels}")
    # Convert index back to class label
    predicted_class = label_mapping.get(predicted_labels, "Unknown")
    print(f"Predicted class: {predicted_class}")   
    return predicted_class
    # raise NotImplementedError()


# class TextDataset(torch.utils.data.Dataset):
#     """
#     Custom PyTorch dataset for tokenized text data.
#     """
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels.iloc[idx])
#         return item

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: accuracy, precision, recall, and F1-score.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)  # Get predicted labels

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_bert_model(file_path: str) -> str:
    """
    Train a BERT model using a given dataset file.

    :param file_path: Path to the CSV dataset file.
    :return: Path where the trained model is saved.
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print("loading the dataset for training the bert model")

    # Encode class labels into numerical values
    label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].map(label_mapping)
    print(label_mapping)
    # Split dataset into train, validation, and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["description"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=42
    )
    print(len(train_texts))
    print(len(test_texts))
    print(len(val_texts))
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize text
    def tokenize_texts(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=512)

    train_encodings = tokenize_texts(train_texts)
    val_encodings = tokenize_texts(val_texts)
    test_encodings = tokenize_texts(test_texts)

    # Create datasets
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    # Load BERT model with the correct number of labels
    num_labels = len(label_mapping)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Training started...")
    trainer.train()

    # Save the trained model
    model_path = "saved_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Training complete. Model saved at {model_path}")
    return model_path


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route('/generate_eda', methods=['POST'])
def generate_eda():
    """
    API endpoint to generate an EDA report from a CSV file.
    Expects a JSON request with 'file_path' as input.
    """
    try:
        # Parse request data
        print("I am in generate_eda")
        data = request.get_json(force=True)
        print(data)
        file_path = data.get("file_path")
        # Validate input
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "Invalid or missing file path"}), 400
        # Generate EDA report
        report_path = generate_eda_report(file_path)
        return jsonify({
            "message": "EDA report generated successfully",
            "report_path": report_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    """
    API endpoint to train the BERT model.
    Expects a JSON request with 'file_path' as input.
    """
    try:
        # Parse request data
        print("I am in train the model")
        data = request.get_json()
        file_path = data.get("file_path")

        # Validate input
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "Invalid or missing file path"}), 400

        # Train the model
        model_path = train_bert_model(file_path)

        return jsonify({
            "message": "Model training completed successfully",
            "model_path": model_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Define the TextDataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item
def predict(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """

    # Reload the trained model from the checkpoint
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    trainer = Trainer(model=model, tokenizer=tokenizer)
    print(" work work work ")
    print(description)
    text = description
    encoding = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Create a dataset without labels
    dataset = TextDataset(encoding)  # No labels during prediction
    
    # Predict using Trainer
    predictions, label_ids, metrics = trainer.predict(dataset)
    print(predictions)
    # prediction_dataset = TextDataset(description, tokenizer)
     # Make predictions
    # predictions, label_ids, metrics = trainer.predict(prediction_dataset)

    # Convert logits to predicted class labels (by taking argmax)
    predicted_labels = np.argmax(predictions)
    # Print the predicted labels
    print(f"Predicted Labels: {predicted_labels}")
    # Convert index back to class label
    predicted_class = label_mapping.get(predicted_labels, "Unknown")
    print(f"Predicted class: {predicted_class}")   
    return predicted_class
    # raise NotImplementedError()

@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)

    prediction = predict(data["description"])

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)



### remove this code later

def eda():
    df = pd.read_csv("data/trials.csv")  # Ensure it has 'description' and 'label' columns
    df["text_length"] = df["description"].apply(len)
    df["word_count"] = df["description"].apply(lambda x: len(x.split()))
    df['tokens'] = df['description'].apply(lambda x: len(word_tokenize(x)))
    # Apply preprocessing
    df["cleaned_description"] = df["description"].apply(preprocess_text)
    df['cleaned_tokens'] = df['cleaned_description'].apply(lambda x: len(word_tokenize(x)))

    profile = ProfileReport(df, title="Research Grid EDA data exploration", explorative=True)
    # Save the profile report to an HTML file
    profile.to_file("trials_eda.html")


