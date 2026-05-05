import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Added for Step 1: Saving the model

# 1. Load your dataset 
df = pd.read_csv(r'C:\Users\leone\OneDrive\Documents\BS Computer Science\4th Year\Manuscript\Training\student_feedback.csv')

# 2. Preprocessing (Cleaning & Tokenization) 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower() # Normalization 
    tokens = text.split()
    # Remove stop words 
    cleaned = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned)

df['cleaned_feedback'] = df['Feedback'].apply(clean_text)

# 3. Feature Extraction (TF-IDF) 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_feedback'])
y = df['Sentiment']

# 4. Split data (80% Training, 20% Testing) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 5. Initialize Models 
nb_model = MultinomialNB()
svm_model = SVC(kernel='linear') # Linear kernel is best for text

# 6. Training
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

print("Training Complete. Ready for Evaluation.")

# 7. Model Evaluation Function 
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"--- {model_name} Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # 8. Generate Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=['Positive', 'Negative', 'Neutral'])
    print(f"{model_name} Confusion Matrix:")
    print(cm)
    print("-" * 30)
    return acc, predictions

# Execute Evaluation
nb_accuracy, nb_preds = evaluate_model(nb_model, X_test, y_test, "Naïve Bayes")
svm_accuracy, svm_preds = evaluate_model(svm_model, X_test, y_test, "Support Vector Machine")

# --- STEP 1 FOR WEB APP: SAVE THE MODEL AND VECTORIZER ---
# We save the SVM model because it had the best Recall for Negative sentiments.
joblib.dump(svm_model, 'svm_sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Step 1 Success: svm_sentiment_model.pkl and tfidf_vectorizer.pkl have been saved.")

# --- 9. Data Visualization (Keep commented out or uncomment as needed) ---
# ... (rest of your visualization code)