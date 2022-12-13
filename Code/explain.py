from sklearn.feature_extraction.text import TfidfVectorizer
import os
import nltk
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict


def model(clf, name):
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print(f"{name} Train Accuracy score: {accuracy_score(y_train, y_train_pred)}")
    print(f"{name} Train F1-score: {f1_score(y_train, y_train_pred)}")
    print(f"{name} Test Accuracy score: {accuracy_score(y_test, y_test_pred)}")
    print(f"{name} Test F1-score: {f1_score(y_test, y_test_pred)}")
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


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
print(data_dir)
train_file = os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json')
test_file = os.path.join(data_dir, 'Sarcasm_Headlines_Dataset_v2.json')
df1 = pd.read_json(train_file, lines=True)
df2 = pd.read_json(test_file, lines=True)
# df_train, df_test = df1, df2
df = pd.concat([df1, df2], axis=0)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], shuffle=True)
X_train, y_train = df_train['headline'], df_train['is_sarcastic']
X_test, y_test = df_test['headline'], df_test['is_sarcastic']


tfidf = TfidfVectorizer()
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)


clf_rf = LogisticRegression(n_jobs=1, C=1e5)
model(clf_rf, 'LR')


idx = df_test.index[np.random.randint(0,len(df_test)-1)]
print(f'Headline = {df_test["headline"][idx]}')

c = make_pipeline(tfidf, clf_rf)
class_names = ["Non Sarcastic", "Sarcastic"]
explainer = LimeTextExplainer(class_names = class_names)
exp = explainer.explain_instance(df_test["headline"][idx], c.predict_proba, num_features = 10)

print("Question: \n", df_test["headline"][idx])
print("Probability (Non sarcastic) =", c.predict_proba([df_test["headline"][idx]])[0, 1])
print("Probability (sarcastic) =", c.predict_proba([df_test["headline"][idx]])[0, 0])
print("True Class is:", class_names[df_test["is_sarcastic"][idx]])

print(exp.as_list())
exp.as_pyplot_figure()

plt.show()
weights = OrderedDict(exp.as_list())
lime_weights = pd.DataFrame({"words": list(weights.keys()), "weights": list(weights.values())})


sns.barplot(x = "words", y = "weights", data = lime_weights)
plt.xticks(rotation = 45)
plt.title("Sample {} features weights given by LIME" .format(idx))
plt.text(0.5, 0.5, s=df_test["headline"][idx], fontsize=5)
plt.show()

# exp.show_in_notebook()  #Uncomment this line if running in notebook

