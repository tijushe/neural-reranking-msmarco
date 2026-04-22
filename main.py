import re

import evaluate
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Set seed for reproducibility
set_seed(42)

def extract_sentence(text):
    pattern = r'^\[\d+\]:\s*'

    return re.sub(pattern, '', text)

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def base_preprocessing(sentence):
    # Remove [nr]
    sentence = extract_sentence(sentence)
    # Remove HTML tags
    sentence = remove_html_tags(sentence)
    # Normalize whitespace
    sentence = re.sub(r'\s+', ' ', sentence)
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    # print(sentence)
    return sentence

def aggregate_ratings_for_article(df):

    all_rows = []
    
    # For each row, we'll pivot the 20 numeric columns and 20 sentence columns into long form
    for idx, row in df.iterrows():
        article_id = row['id_article']
        for sentence_i in range(20):
            sent_col = f's{sentence_i}'
            bias_col = str(sentence_i)  # because columns are named '0','1','2'...
            
            raw_text = row[sent_col]
            raw_label = row[bias_col]
            
            # If it's NaN or empty, skip
            if pd.isnull(raw_text) or pd.isnull(raw_label):
                continue
            
            sentence_text = extract_sentence(str(raw_text))
            rating = float(raw_label)
            
            # Add to a temporary list
            all_rows.append({
                'id_article': article_id,
                'sentence_idx': sentence_i,
                'text': sentence_text,
                'rating': rating
            })

    # Create a new DataFrame in long format
    long_df = pd.DataFrame(all_rows)

    def majority_vote(arr):
        counts = arr.value_counts()
        return int(counts.index[0])  # label with highest count
    
    # Group by id_article and sentence_idx
    grouped = (long_df
               .groupby(['id_article', 'sentence_idx'], as_index=False)
               .agg({'text': 'first',  # pick the first text (they should be identical if the data is consistent)
                     'rating': majority_vote}))  # average rating

    return grouped

# Function to load and preprocess the dataset for traditional models
def preprocess_traditional_data(data):
    
    # Preview the data
    print("Dataset Preview:")
    print(data.head())
    
    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Handle missing values by filling them with empty strings
    data.fillna('', inplace=True)
    
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Prepare the data for modeling
    sentences = []
    labels = []
    
    # Iterate over each row in the dataset
    for index, row in data.iterrows():
        sentence = base_preprocessing(row["text"])
        bias = row["rating"]

        if sentence and bias:
            # Preprocess the sentence
            tokens = nltk.word_tokenize(sentence.lower())
            tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
            tokens = [stemmer.stem(word) for word in tokens]
            preprocessed_sentence = ' '.join(tokens)

            sentences.append(preprocessed_sentence)
            labels.append(1 if bias >= 3 else 0)

    # Create a DataFrame with sentences and labels
    df = pd.DataFrame({'preprocessed_text': sentences, 'label': labels})
    
    # Remove any empty sentences
    df = df[df['preprocessed_text'].str.strip() != '']
    
    print("\nProcessed DataFrame:")
    print(df.head())
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    
    return df

# Function to create train, validation, and test datasets
def create_datasets(df):
    # Split the data
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,  # 60% train, 40% temp
        stratify=df['label'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # Split temp into 20% val and 20% test
        stratify=temp_df['label'],
        random_state=42
    )
    
    print("\nTrain Set Size:", len(train_df))
    print("Validation Set Size:", len(val_df))
    print("Test Set Size:", len(test_df))
    
    return train_df, val_df, test_df

# Function to preprocess data for transformer models
def preprocess_transformer_data(data):
    
    # Handle missing values
    data.fillna('', inplace=True)
    
    # Prepare the data
    sentences = []
    labels = []
    
    for index, row in data.iterrows():

        sentence = base_preprocessing(row["text"])
        bias = row["rating"]

        if sentence and bias:
            sentences.append(str(sentence))
            labels.append(1 if bias >=3 else 0)
    
    # Create DataFrame
    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    
    # Remove empty sentences
    df = df[df['sentence'].str.strip() != '']
    
    print("\nTransformer DataFrame:")
    print(df.head())
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    
    return df

# Function to train and evaluate Logistic Regression with TF-IDF
def train_evaluate_logistic_regression(train_df, val_df, test_df, vect_name, vectorizer):
    print(f"\n--- Logistic Regression with {vect_name} ---")
    
    # Fit on training data and transform
    X_train = vectorizer.fit_transform(train_df['preprocessed_text'])
    X_val = vectorizer.transform(val_df['preprocessed_text'])
    X_test = vectorizer.transform(test_df['preprocessed_text'])
    
    # Labels
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']
    
    # Initialize and train the classifier
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['Not Biased', 'Biased']))
    
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Biased', 'Biased'], yticklabels=['Not Biased', 'Biased'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Logistic Regression ({vect_name})')
    plt.show()
    plt.savefig(f"lr_{vect_name}_conf_mat.png")
    plt.close()
    
    # Return metrics for comparison
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    return {
        'model': f'Logistic Regression ({vect_name})',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Function to train and evaluate transformer-based models (BERT & DistilBERT)
def train_evaluate_transformers(train_df, val_df, test_df, model_name):
    print(f"\n--- {model_name} ---")
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    features = Features({
        'sentence': Value('string'),
        'label': ClassLabel(names=['Not Biased', 'Biased']),
    })
    
    train_dataset = Dataset.from_pandas(train_df, features=features, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, features=features, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, features=features, preserve_index=False)
    
    datasets = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    })
    
    # Tokenization Function
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)
    
    # Tokenize the datasets
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    
    # Remove the original sentence column
    tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
    
    # Set the format for PyTorch
    tokenized_datasets.set_format('torch')
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs",
        logging_steps=10,
    )
    
    # Load pre-trained Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: 'Not Biased', 1: 'Biased'},
        label2id={'Not Biased': 0, 'Biased': 1},
    )
    
    # Define compute cetrics function
    def compute_metrics(eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        
        # Calculate Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model on the test set
    print(f"\nEvaluating {model_name} on Test Set...")
    metrics = trainer.evaluate(tokenized_datasets['test'])
    print(metrics)

    # Generate predictions to build confusion matrix
    predictions = trainer.predict(tokenized_datasets['test'])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Compute the confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Biased', 'Biased'], yticklabels=['Not Biased', 'Biased'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    plt.savefig(f"{model_name}_conf_mat.png")
    plt.close()
    
    return metrics

# Function to compare results across models
def compare_models(results):
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df)
    
    # Plot F1 Scores
    plt.figure(figsize=(8,6))
    sns.barplot(x='model', y='f1_score', data=results_df, palette='viridis')
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.show()
    plt.savefig("f1_comp.png")
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(8,6))
    sns.barplot(x='model', y='accuracy', data=results_df, palette='magma')
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.show()
    plt.savefig("acc_comp.png")
    plt.close()
    
    # Plot Precision and Recall
    metrics_melted = results_df.melt(id_vars='model', value_vars=['precision', 'recall'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(8,6))
    sns.barplot(x='model', y='Score', hue='Metric', data=metrics_melted, palette='coolwarm')
    plt.title('Precision and Recall Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.show()
    plt.savefig("prec_rec_comp.png")
    plt.close()


def main():
    # Load the dataset
    data_url = 'https://raw.githubusercontent.com/skymoonlight/biased-sents-annotation/refs/heads/master/Sora_LREC2020_biasedsentences.csv'
    data_df = pd.read_csv(data_url)

    grouped_df = aggregate_ratings_for_article(data_df)
    
    # TRADITIONAL MODEL WORKFLOW

    # Preprocess data for traditional approaches
    df_traditional = preprocess_traditional_data(grouped_df)
    
    # Split data
    train_df_trad, val_df_trad, test_df_trad = create_datasets(df_traditional)
    
    # 4) Train and evaluate Logistic Regression with TF-IDF
    lr_tfidf_metrics = train_evaluate_logistic_regression(train_df_trad, val_df_trad, test_df_trad, "TF-IDF", TfidfVectorizer())
    lr_bow_metrics = train_evaluate_logistic_regression(train_df_trad, val_df_trad, test_df_trad, "BOW", CountVectorizer())
    
    # TRANSFORMER MODEL WORKFLOW

    # Preprocess data for transformer models
    df_transformer = preprocess_transformer_data(grouped_df)

    train_df_trans, val_df_trans, test_df_trans = create_datasets(df_transformer)
    
    # Train and evaluate Transformer models
    results = []
    results.append(lr_tfidf_metrics)  # Add Logistic Regression result tfidf
    results.append(lr_bow_metrics)  # Add Logistic Regression result bow
    
    for model_name in ["bert-base-uncased", "distilbert-base-uncased"]:
        metrics = train_evaluate_transformers(train_df_trans, val_df_trans, test_df_trans, model_name)
        
        results.append({
            'model': model_name,
            'accuracy': metrics['eval_accuracy'],
            'precision': metrics['eval_precision'],
            'recall': metrics['eval_recall'],
            'f1_score': metrics['eval_f1'],
        })
    
    # COMPARE MODELS

    compare_models(results)

# Execute the main function
if __name__ == "__main__":
    main()
