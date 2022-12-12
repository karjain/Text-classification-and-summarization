# Sarcasm Detection and News Article Summarization 

This is a comparative study to evalute different text classification and text summarization models

<h3>Models tested</h3>
<ul>
    <li>Classification</li>
        <ul>
            <li>Classical model</li>
            <li>MLP and RNN model using GloVe embeddings</li>
            <li>Transformers - BERT + MLP</li>
            <li>Transformers - RoBERTa + MLP</li>
        </ul>
    <li>Summarization</li>
        <ul>
            <li>Transformer model -T5-small-headline-generator</li>
            <li>Transformer model - T5-small</li>
        </ul>
</ul>

## Instructions to run

```bash
  python3 -m pip install -r requirements.txt
  cd Code
  python3 -u classification.py
  python3 -u text_summarization.py
  
```
Additionally <b><u>-train</u></b> argument can be used to begin the traning phase, otherwise, only prediction method is called on the test data using the already trained model
<ul> 
<li><b>classification.py</b> file runs the text summarization pipeline and generate summarize text from the news article. It uses a Transformers based RoBERTa + MLP architecture and achieves 98% accuracy on the test data </li>
<br>
<li><b>text_summarization.py </b>file runs the text summarization pipeline and generate summarize text from the news article. It uses a Transformers based T5small architecture pre-trained on News articles. Achieves Roug1 score of 0.52 on the test data  </li>
</ul>


## Dataset
Two data set are used to train the model
<br>
<a href="https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection?select=Sarcasm_Headlines_Dataset_v2.json">News Headlines Dataset For Sarcasm Detection V2</a>
<br><a href="https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection">News Headlines Dataset For Sarcasm Detection </a>

<h4> Dataset example</h4>


```bash
 "root":{3 items
 "article_link":string"https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5"
 "headline":string"former versace store clerk sues over secret 'black code' for minority shoppers"
 "is_sarcastic":int0
  
```
<h4>About this file</h4>
Each record consists of three attributes:
<ul> 
<li><b>is_sarcastic</b>: 1 if the record is sarcastic otherwise 0</li>
<li><b>headline</b>: the headline of the news article</li>
<li><b>article_link</b>: link to the original news article. Useful for collecting supplementary data</li>
</ul>
