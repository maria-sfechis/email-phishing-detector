import imaplib
import email
from email.header import decode_header
import string
import re
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Download required NLTK resources (uncomment if needed)
#nltk.download('stopwords', download_dir='D:/nltk_data')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.data.path.append('D:/nltk_data')

class EmailPhishingDetector:
    def __init__(self, model_path=None):
        """Initialize the phishing detector with optional pre-trained model."""
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english') + stopwords.words('romanian'))

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def connect_to_mail_server(self, email_account, password_account, server="imap.mail.yahoo.com"):
        """Connect to mail server and return connection."""
        try:
            mail = imaplib.IMAP4_SSL(server)
            mail.login(email_account, password_account)
            return mail
        except Exception as e:
            print(f"Error connecting to mail server: {e}")
            return None

    def fetch_emails(self, mail, folder="inbox", limit=50):
        """Fetch emails from specified folder."""
        try:
            mail.select(folder)
            status, messages = mail.search(None, 'ALL')
            email_ids = messages[0].split()

            if not email_ids:
                print(f"No emails found in {folder}!")
                return []

            # Get the most recent emails (limited by 'limit')
            email_data = []
            for num in email_ids[-limit:]:
                status, msg_data = mail.fetch(num, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])

                        # Extract subject
                        subject, encoding = decode_header(msg['Subject'])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8")

                        # Extract sender
                        sender = msg.get("From")

                        # Extract body
                        body = self._extract_body(msg)

                        # Extract URLs
                        urls = self._extract_urls(body)

                        email_data.append({
                            'id': num.decode(),
                            'sender': sender,
                            'subject': subject,
                            'body': body,
                            'urls': urls,
                            'folder': folder
                        })

            return email_data
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []

    def _extract_body(self, msg):
        """Extract email body from message."""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Extract text content
                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode()
                    except:
                        body += str(part.get_payload(decode=True))
                elif content_type == "text/html":
                    try:
                        html_content = part.get_payload(decode=True).decode()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        body += soup.get_text()
                    except:
                        pass
        else:
            # Handle non-multipart messages
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                try:
                    body = msg.get_payload(decode=True).decode()
                except:
                    body = str(msg.get_payload(decode=True))
            elif content_type == "text/html":
                try:
                    html_content = msg.get_payload(decode=True).decode()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    body = soup.get_text()
                except:
                    pass

        return body

    def _extract_urls(self, text):
        """Extract URLs from text."""
        # Simple URL extraction using regex
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        found_urls = re.findall(url_pattern, text)

        # Extract URL features
        url_features = []
        for url in found_urls:
            parsed_url = urlparse(url)
            url_features.append({
                'url': url,
                'domain': parsed_url.netloc,
                'path_length': len(parsed_url.path),
                'query_params': len(parsed_url.query),
                'has_ip': self._is_ip_address(parsed_url.netloc)
            })

        return url_features

    def detect_language(self, text):
        """Detect language: RO/EN/unknown"""
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        try:
            return detect(text)
        except:
            return "unknown"
    def _is_ip_address(self, domain):
        """Check if domain is an IP address."""
        ip_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'
        match = re.match(ip_pattern, domain)
        if match:
            return all(0 <= int(octet) <= 255 for octet in match.groups())
        return False

    def extract_features(self, emails):
        """Extract features from email data for training or prediction."""
        features = []

        for email_item in emails:
            # Basic email features
            subject = email_item.get('subject', '')
            body = email_item.get('body', '')
            sender = email_item.get('sender', '')
            receiver = email_item.get('receiver', '')
            date = email_item.get('date', '')
            urls = email_item.get('urls', [])

            # Process text features
            processed_subject = self.preprocess_text(subject)
            processed_body = self.preprocess_text(body)

            # URL-based features
            url_count = len(urls) if isinstance(urls, list) else 1 if urls else 0
            suspicious_url_count = sum(int(url) for url in urls if url in ['0', '1'])
            avg_path_length = 0

            # Combine features
            combined_text = f"{processed_subject} {processed_body}"

            # Detect language
            language = self.detect_language(combined_text)
            is_romanian = 1 if language == "ro" else 0
            is_english = 1 if language == "en" else 0

            # Add email structure features
            email_features = {
                'combined_text': combined_text,
                'url_count': url_count,
                'suspicious_url_count': suspicious_url_count,
                'avg_url_path_length': avg_path_length,
                'has_suspicious_sender': self._check_suspicious_sender(sender),
                'contains_urgency_words': self._check_urgency_words(combined_text),
                'contains_sensitive_words': self._check_sensitive_words(combined_text),
                'receiver': receiver,
                'is_romanian': is_romanian,
                'is_english': is_english
            }

            features.append(email_features)

        return features

    def _check_suspicious_sender(self, sender):
        """Check for suspicious sender characteristics."""
        if not sender:
            return False

        suspicious_patterns = [
            r'@.*\.(xyz|top|club|gq|ml|ga|cf|tk|pw)',  # Suspicious TLDs
            r'@.*\d{4,}',  # Many numbers in domain
            r'noreply@',  # Generic no-reply addresses can be suspicious in context
            r'@.*-.*-.*\.'  # Multiple hyphens in domain
        ]

        return any(re.search(pattern, sender.lower()) for pattern in suspicious_patterns)

    def _check_urgency_words(self, text):
        """Check for urgency words that might indicate phishing."""
        urgency_words = [
            'urgent', 'immediately', 'alert', 'attention', 'important',
            'verify', 'suspended', 'restricted', 'limited', 'security',
            'update', 'login', 'unusual', 'suspicious', 'authorize',
            'urgenta', 'imediat', 'alerta', 'atentie', 'importanta'  # Romanian words
        ]

        text_tokens = text.lower().split()
        return any(word in text_tokens for word in urgency_words)

    def _check_sensitive_words(self, text):
        """Check for words related to sensitive information requests."""
        sensitive_words = [
            'password', 'credit card', 'social security', 'ssn', 'account', 'block'
            'login', 'credentials', 'verify', 'confirm', 'update', 'banking',
            'parola', 'card', 'cont', 'banca', 'actualizare', 'urgent'  # Romanian words
        ]

        text_lower = text.lower()
        return any(word in text_lower for word in sensitive_words)

    def preprocess_text(self, text):
        """Preprocess text for feature extraction."""
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs (they're handled separately)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags if any
        text = re.sub(r'<.*?>', '', text)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and apply lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        # Rejoin tokens
        return ' '.join(tokens)

    def prepare_training_data(self, email_features, labels):
        """Prepare training data from extracted features."""
        # Create DataFrame from features
        df = pd.DataFrame(email_features)

        # Prepare text data for TF-IDF
        X_text = df['combined_text'].tolist()

        # Create TF-IDF vectors
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_text_vectorized = self.vectorizer.fit_transform(X_text)
        else:
            X_text_vectorized = self.vectorizer.transform(X_text)

        # Get non-text features
        X_meta = df[['url_count', 'suspicious_url_count', 'avg_url_path_length',
                     'has_suspicious_sender', 'contains_urgency_words',
                     'contains_sensitive_words']].values

        # Combine all features - this requires converting sparse matrix to dense
        X_text_dense = X_text_vectorized.toarray()
        X_combined = np.hstack((X_text_dense, X_meta))

        return X_combined, np.array(labels)

    def train_model(self, X, y, model_type='ensemble', use_smote=True, cv=5):
        """Train the phishing detection model with cross-validation."""
        print("Starting model training...")

        # Handle class imbalance with SMOTE if requested
        if use_smote:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print(f"Data shape after SMOTE: {X.shape}, Class distribution: {np.bincount(y)}")

        # Define model based on type
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42)
        elif model_type == 'ensemble':
            # Create an ensemble voting classifier
            from sklearn.ensemble import VotingClassifier
            estimators = [
                ('nb', MultinomialNB()),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42))
            ]
            self.model = VotingClassifier(estimators=estimators, voting='soft')
        else:
            # Default to Random Forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Perform cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                                    scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Train final model on all data
        print("Training final model on all data...")
        self.model.fit(X, y)
        print("Model training completed")

        return cv_scores.mean()

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance on test data."""
        if self.model is None:
            print("Model not trained yet!")
            return None

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print metrics
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Plot ROC curve if probability estimates are available
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.savefig('roc_curve.png')
            plt.close()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict(self, email_data):
        """Predict if emails are phishing or legitimate."""
        if self.model is None or self.vectorizer is None:
            print("Model not trained or loaded yet!")
            return None

        # Extract features
        features = self.extract_features(email_data)

        # Prepare data for prediction
        df = pd.DataFrame(features)
        X_text = df['combined_text'].tolist()
        X_text_vectorized = self.vectorizer.transform(X_text)

        # Get non-text features
        X_meta = df[['url_count', 'suspicious_url_count', 'avg_url_path_length',
                     'has_suspicious_sender', 'contains_urgency_words',
                     'contains_sensitive_words']].values

        # Combine all features
        X_text_dense = X_text_vectorized.toarray()
        X_combined = np.hstack((X_text_dense, X_meta))

        # Make predictions
        predictions = self.model.predict(X_combined)
        probabilities = self.model.predict_proba(X_combined)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Combine predictions with email data
        for i, email_item in enumerate(email_data):
            email_item['is_phishing'] = bool(predictions[i])
            if probabilities is not None:
                email_item['phishing_probability'] = float(probabilities[i])

            # Add explanation
            email_item['explanation'] = self._generate_explanation(features[i], predictions[i])

        return email_data

    def _generate_explanation(self, features, prediction):
        """Generate explanation for prediction."""
        explanation = []

        if prediction == 1:  # Phishing
            if features['url_count'] > 0:
                explanation.append(f"Contains {features['url_count']} URLs")

            if features['suspicious_url_count'] > 0:
                explanation.append(f"Contains {features['suspicious_url_count']} suspicious URLs")

            if features['has_suspicious_sender']:
                explanation.append("Sender domain looks suspicious")

            if features['contains_urgency_words']:
                explanation.append("Contains urgency language")

            if features['contains_sensitive_words']:
                explanation.append("Requests sensitive information")
        else:  # Legitimate
            explanation.append("No suspicious elements detected")

        return ", ".join(explanation)

    def save_model(self, path="phishing_model.pkl"):
        """Save trained model and vectorizer."""
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet!")
            return False

        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")
        return True

    def load_model(self, path="phishing_model.pkl"):
        """Load trained model and vectorizer."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']

            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


    def interactive_training(detector, csv_path=None):
        """Interactive training function that either loads data from CSV or allows manual labeling."""
        email_data = []
        labels = []

        if csv_path and os.path.exists(csv_path):
            # Load existing data from CSV
            print(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path, dtype={'urls': str}, low_memory=False)
            df.fillna("", inplace=True)

            # Prepare email data structure
            existing_emails = set(
                (row['sender'], row['subject'], row['body']) for _, row in df.iterrows()
            )

            #email_data = []
            for _, row in df.iterrows():
                email_item = {
                    'sender': row.get('sender', ''),
                    'receiver': row.get('receiver', ''),
                    'date': row.get('date', ''),
                    'subject': row.get('subject', ''),
                    'body': row.get('body', ''),
                    'urls': str(row.get('urls','')).split(';') if pd.notna(row.get('urls','')) else []  # URLs would need to be re-extracted if needed
                }
                email_data.append(email_item)
                labels.append(row['label'])
        else:
            print("Please provide path to CSV file!")

        # Interactive training through email fetching
        email_account = input("Enter email account: ")
        password_account = input("Enter password: ")

        mail = detector.connect_to_mail_server(email_account, password_account)
        if not mail:
             print("Failed to connect to mail server.")
             return

        # Fetch new emails from inbox
        inbox_emails = detector.fetch_emails(mail, "inbox", limit=10)
        for email_item in inbox_emails:
            if (email_item['sender'], email_item['subject'], email_item['body']) not in existing_emails:
                print(f"\nFrom: {email_item['sender']}")
                print(f"Subject: {email_item['subject']}")
                print(f"Body: {email_item['body'][:200]}...")
                label = int(input("Is this phishing? (1 = Yes, 0 = No): "))
                if label in [0, 1]:
                    email_data.append(email_item)
                    labels.append(label)
                    existing_emails.add((email_item['sender'], email_item['subject'], email_item['body']))

        # Close connection
        mail.logout()

        # Save labeled emails to CSV
        if csv_path:
            print(f"Saving new data to {csv_path}...")
            new_emails = [
                {
                    'sender': email['sender'],
                    'receiver': email.get('receiver', ''),
                    'date': email.get('date', ''),
                    'subject': email['subject'],
                    'body': email['body'],
                    'label': label,
                    'urls': ';'.join(
                        [url.get('url', '') for url in email.get('urls', []) if isinstance(url, dict)]) if email.get(
                        'urls') else ''
                }
                for email, label in zip(email_data, labels)
            ]
            df_new = pd.DataFrame(new_emails)

            if os.path.exists(csv_path):
                # Append without header
                df_new.to_csv(csv_path, mode='a', index=False, header=False)
            else:
                df_new.to_csv(csv_path, index=False)

            print(f"New data have been saved to {csv_path}")

        # Extract features
        features = detector.extract_features(email_data)

        # Split data
        X, y = detector.prepare_training_data(features, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Starting model training...")
        # Train model
        detector.train_model(X_train, y_train)
        # Evaluate model
        detector.evaluate_model(X_test, y_test)
        # Save model
        detector.save_model()

        print("Training complete. Model saved.")

def test_on_new_email(detector):
    """Test the model on a random email from inbox or a CSV file."""
    print("Choose testing method for emails from: ")
    print("1. Inbox")
    print("2. CSV file")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        email_account = input("Enter email account: ")
        password_account = input("Enter password: ")

        mail = detector.connect_to_mail_server(email_account, password_account)
        if not mail:
            print("Failed to connect to mail server.")
            return

        # Fetch multiple emails (ex.30) to choose randomly from
        email_data = detector.fetch_emails(mail, "inbox", limit=30)
        mail.logout()

        if not email_data:
            print("No emails found!")
            return

        random_email = random.choice(email_data)
        # Display email info
        print(f"\nFrom: {random_email['sender']}")
        print(f"Subject: {random_email['subject']}")
        print(f"Body: {random_email['body'][:200]}...")

    elif choice == '2':
        csv_path = input("Enter path to CSV file: ").strip()

        if not os.path.exists(csv_path):
            print("CSV file not found!")
            return

        print(f"\nLoading test emails from {csv_path}...")
        df = pd.read_csv(csv_path, usecols=['sender', 'receiver', 'subject', 'body'])
        df.fillna('', inplace=True)

        if df.empty:
            print("No emails found!")
            return

        random_email = df.sample(1).iloc[0].to_dict()

        # Display email info
        print(f"\nFrom: {random_email.get('sender', '')}")
        print(f"Subject: {random_email.get('subject', '')}")
        print(f"Body: {random_email.get('body', '')[:200]}...")

    # Make prediction
    results = detector.predict([random_email])
    if results:
        result = results[0]
        print("\nPhishing Detection Result:")
        # Check result: phishing or legitimate
        if result['is_phishing']:
            print("⚠️ PHISHING DETECTED!")
            if 'phishing_probability' in result:
                print(f"Confidence: {result['phishing_probability']:.2%}")
        else:
            print("✅ Email appears to be legitimate")
            if 'phishing_probability' in result:
                print(f"Confidence: {1 - result['phishing_probability']:.2%}")

        print(f"Explanation: {result.get('explanation', 'No explanation available.')}")

def main():
    print("Email Phishing Detection System")
    print("===============================")

    # Create detector instance
    detector = EmailPhishingDetector()

    # Check for existing model
    if os.path.exists("phishing_model.pkl"):
        load_existing = input("Found existing model. Load it? (y/n): ")
        if load_existing.lower() == 'y':
            detector.load_model()

    while True:
        print("\nOptions:")
        print("1. Train model with CSV data")
        print("2. Train model interactively with emails")
        print("3. Test model on new email")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            csv_path = input("Enter path to CSV file to load data: ").strip()
            detector.interactive_training(csv_path)
        elif choice == '2':
            detector.interactive_training()
        elif choice == '3':
            test_on_new_email(detector)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()