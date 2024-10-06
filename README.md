# Fibe - Hack the Vibe! 2.0 - ML Challenge

### Author by Debojyoti Chanda


## About Challenge
Welcome to **Fibe - Hack the Vibe! 2.0 - ML Challenge**.

Dive into the ultimate AI challenge with **Fibe - Hack the Vibe! 2.0**! This is your chance to shine in the fast-paced world of artificial intelligence and make a real impact with your innovative solutions.

We're inviting you to harness the power of AI by developing cutting-edge models using a dataset of news articles. With over **870,000 samples** at your disposal, you'll tackle the complexities of text classification and contribute to advancing AI technologies.

### Who's it for?
Build an epic text classification model using machine learning and other state-of-the-art techniques. Whether you're a **solo visionary** or part of a **team of 2 members**, we want you to bring your best ideas to the table.

### Eligibility Criteria:
- **Open to All**

Join us to unleash your creativity, push the boundaries of AI, and be part of a groundbreaking experience in technology. Get ready to innovate, compete, and succeed at **Fibe - Hack the Vibe! 2.0**!

---

## Dataset Information

### Train Data Info:
- **Entries:** 697,527
- **Columns:** 3
- **Column Details:**
  - `text`: Contains the news article text. (697,527 non-null values, dtype: object)
  - `target`: Contains the corresponding labels. (697,527 non-null values, dtype: object)
  - `Word Count`: Contains total word count of text(697,527 non-null values, dtype: int64)
- **Memory Usage:** 10.6+ MB

### Test Data Info:
- **Entries:** 174,382
- **Columns:** 3
- **Column Details:**
  - `text`: Contains the news article text. (174,382 non-null values, dtype: object)
  - `Word Count`: The count of words in each article. (174,382 non-null values, dtype: int64)
  - `Index`: A reference index for evaluation purposes. (174,382 non-null values, dtype: object)
- **Memory Usage:** 4.0+ MB

### Submission Data Info:
- **Entries:** 174,382
- **Columns:** 2
- **Column Details:**
  - `Index`: A reference index for each entry. (174,382 non-null values, dtype: object)
  - `target`: Contains the predicted labels for each sample. (174,382 non-null values, dtype: object)
- **Memory Usage:** 2.7+ MB
---

## Introduction
This project leverages TF-IDF for feature extraction, enabling the model to capture essential word frequencies while minimizing the impact of common words. The core of the model is a `MLPClassifier` with hidden layers, which helps capture complex patterns and interactions in text data, providing more predictive power than linear classifiers. 

Key techniques include early stopping to prevent overfitting, batch processing for optimized training, and a focus on the weighted F1 score for evaluating model performance on imbalanced datasets. The use of a validation set ensures reliable performance estimates before final predictions.

---

## Approach

### 1. Data Loading
- Pandas is used to load the training and test datasets (`pd.read_csv()`).
- The dataset encoding is set to `ISO-8859-1` to accommodate potential non-ASCII characters.
- `.shape` is used to verify the dimensions of the dataset.

### 2. Stopword Removal with NLTK
- NLTK is used for text processing, including downloading tokenizers and stopword lists.
- Common stopwords like "the", "is", and "in" are removed to reduce noise in the data.

### 3. Text Cleaning
- **Normalization:** Convert text to lowercase, remove punctuation, numeric values, and extra spaces.
- **Tokenization:** The text is split into individual words, with stopwords filtered out.

### 4. Applying the Cleaning Function
- The `clean_text()` function is applied to both the training and test datasets to process each row.

### 5. Saving Cleaned Data
- The cleaned datasets are saved as `train_cleaned.csv` and `test_cleaned.csv` for later use.

### 6. Reloading the Saved Data
- Reload the saved datasets to free up memory during large dataset operations.

### 7. TF-IDF Vectorization
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Used to convert text into numerical features.
- **Parameters:** 
  - `max_features=9000`: Limits the vocabulary to the top 8000 features.
  - `ngram_range=(1, 1)`: Only unigrams are extracted.
- Training data is transformed using the fitted TF-IDF vectorizer.

### 8. Train-Test Split
- The training data is split into 80% for training and 20% for validation using `train_test_split`.
- `random_state=42` ensures reproducibility of the split.

### 9. Model Training - MLPClassifier
- **Model Configuration:**
  - Hidden layers: `(128, 64)`
  - Activation: ReLU
  - Optimizer: Adam
  - `max_iter=20`: Limits training epochs.
  - `batch_size=128`: Optimizes computation with mini-batches.
  - Early stopping to avoid overfitting.

### 10. Predictions on Validation Set
- Predictions on the validation set are generated and evaluated against the actual labels.

### 11. Final Predictions on Test Set
- Predictions on the test dataset are generated for final submission.

### 12. Prepare Submission File
- A submission file (`submission.csv`) is created with the predicted labels for the test set.

### 13. F1 Score Calculation
- **F1 Score:** Weighted F1 score is calculated to evaluate the model, which ensures balanced class performance for imbalanced datasets.

---

## Evaluation Metrics
- **Weighted F1 Score:** 
  - Ensures the model performance is fair across different classes.
  - Computed using `metrics.f1_score()` from scikit-learn.
  - The final score is scaled by 100 for readability.

## Submission
- The final submission is saved as `submission.csv` and can be evaluated using the competition’s platform.
