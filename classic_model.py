import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os
import nltk


def model(clf, name):
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print(f"{name} Train Accuracy score: {accuracy_score(y_train, y_train_pred)}")
    print(f"{name} Train F1-score: {f1_score(y_train, y_train_pred)}")
    print(f"{name} Test Accuracy score: {accuracy_score(y_test, y_test_pred)}")
    print(f"{name} Test F1-score: {f1_score(y_test, y_test_pred)}")


def process_text(data):
    lemma = nltk.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    processed_data = list()
    for sent in data:
        first_pass = list()
        for word in sent.split():
            if word not in stopwords:
                first_pass.append(word)
        first_pass = ' '.join(first_pass)
        processed_sent = ' '.join([
            lemma.lemmatize(word) for word in nltk.word_tokenize(first_pass)
            if lemma.lemmatize(word) not in stopwords
        ])
        processed_data.append(processed_sent)
    return processed_data


code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
train_file = os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json')
test_file = os.path.join(data_dir, 'Sarcasm_Headlines_Dataset_v2.json')
df1 = pd.read_json(train_file, lines=True)
df2 = pd.read_json(test_file, lines=True)
# df_train, df_test = df1, df2
df = pd.concat([df1, df2], axis=0)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], shuffle=True)
X_train, y_train = df_train['headline'], df_train['is_sarcastic']
X_test, y_test = df_test['headline'], df_test['is_sarcastic']

# print("Processing train data")
# X_train = process_text(X_train)
# print("\nProcessing test data")
# X_test = process_text(X_test)

tfidf = TfidfVectorizer()
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

print("\nTraining RF classifier")
clf_rf = RandomForestClassifier()
model(clf_rf, 'RF')

print("\nTraining MLP classifier")
clf_mlp = MLPClassifier()
model(clf_rf, 'MLP')
