---

title: ANONYMIZATION OF TEXT DATA USING NAMED-ENTITY RECOGNITION
excerpt: Using CONDITIONAL RANDOM FIELD ALGORITHM TO ANONYMIZE SENSITIVE INFORMATION
date: 2020-01-08
cardimage: './serverless-r-card.jpg'
---
Anonymization of data became a very trendy topic in recent years. It had been widely adressed by the data community due to its growing importance to all companies collecting personal and sensitive information. 

While structured data is standardized and relatively easy to anonymize, dealing with unstructured data is more tedious. There is no database schema that can be used to measure privacy risk. In this blog, I propose to use a named-entity recognition system (NER) to automatically detect textual confidential attributes such as identifiers, sensitive information, etc. In this case study, I will consider that these confidential information to be detected are disease names in medical diagnoses.

# Named-Entity Recognition

Named-entity recognition (also known as entity identification) seeks to identify and classify words in an unstructured text into pre-defined categories.

NER is a very challenging learning problem. On the one hand, Supervised training data is very scarce. On the other, this task require language specific knowledge to construct efficient structured features.

 In our example, we are looking to anonymize medical diagnoses reports by identifying disease names. Thus, our problem is equivalent to a binary classification of names.

> Example : **Testicular cancer ** and **endometriosis**  have increased in incidence during the last decades .



# Methodology

![method](/images/uploads/blog2020/methodo.png)

As a proof of concept, We will be focusing on locating disease names in medical reports using a model based on conditional random fields. 

In practice, given a sentence, the model will tag each word with a `"DISEASE" ` tag if it is a disease name and ` "O" ` tag otherwise,  which indicates that a token belongs to no chunk (outside).

1. First, we load the labeled data, which is a list of sentences and their corresponding labels
2. The second step is the tokenizer, which splits sentences into tokens
3. We use a feature generator to extract reliable features with a window of 3 words (the current word, the previous and the next words)
4. The final step uses a CRF to train a NER model.

# Training Data

In our experiments, we took advantage of medical texts that were labeled to study the semantic relationships between diseases and treatments. These [files](https://biotext.berkeley.edu/dis_treat_data.html) were obtained from MEDLINE 2001 using the first 100 titles and the first 40 abstracts from the 59 files medline01n*.xml. These data contain 3,654 labeled sentences. The labels are: ”DISONLY”, ”TREATONLY”, ”TREAT PREV”, ”DIS PREV”, ”TREAT SIDE EFF”, ”DIS SIDE EFF”, ”DIS VAG”, ”TREAT VAG”, ”TREAT NO” and”DIS NO”. Because we were only interested in diseases, we only used the 629 sentences with the ”DISONLY” label.

After formatting and tokenizing raw text data, it looks like this :

```python
import random
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)
filepath = 'data/sentences_with_roles_and_relations.txt'
disease_data = read_text_labeled_sentences(filepath, tokenizer)
sample_data_point = random.choice(disease_data)
print("Tokens:\n {}".format(sample_data_point[0]))
print("Labels:\n {}".format(sample_data_point[1]))

```

Out:

![method](/images/uploads/blog2020/input_data.png)

So basically, disease data is a list of tuples, each tuple (tokens, labels) represent a sentence divided to tokens and corresponding labels. Where 1 stands for disease name and 0 for other than disease name.

The disease name in the example above  is "Head-neck carcinomas". Thus, the last 2 labels are both equal to 1.

# Conditional Random Fields

CRFs have seen wide application in many areas, including natural language processing and computer vision. They are often used for structured prediction and tasks that require predicting variables that depend on each other as well as on observed variables.  

CRFs models combine the ability of graphical models to compactly model the dependence between multivariate data and the ability of classification methods to predict outputs using large sets of input features. 

These models are considered to be the discriminative equivalents of the hidden Markov models. But first, let's explain what is the difference between generative and discriminative models.

#### Generative and Discriminative Models

Generative models' approach might seem counterintuitive. They describe how a target vector y (type of words in our case) can probabilistically "generate" a feature vector x (words in our case). Discriminative models are more intuitive because they are working backwards, they describe how to assign a label y to a feature vector x.

In principle, we can see that the approaches are distinct. They work in two opposite directions,  but theoretically, we can always convert between the two methods using Bayes rule.

For example, in the naive Bayes model, it is easy to convert the joint probability **p(x,y) = p(y)p(x/y)** into a conditional distribution **p(y/x)**. But in practice, we never have the exact true distribution to calculate the conditional distribution. We can end up with two different estimations of p(y/x). 

To sum up, Generative and discriminative may have the same purpose which is calculating the conditional probability p(y/x), but they proceed in two different ways.

> The difference between generative models and CRFs is exactly analogous to the difference between the naive Bayes and logistic regression classifiers.

CRFs methods can be seen as the discriminative analog of  generative Hidden Markov models. They can also be understood as a generalization of the logistic regression classifier to arbitrary graphical structures.

Since our named-entity recognition task rely on predicting labels based on context and not only on each word's features, CRFs methods might be a good choice to begin with.

We will try in the next section ton implement CRFs methods using [pycrfsuite](https://python-crfsuite.readthedocs.io/en/latest/).

#### Implementation

Now that we explained the motivation behind using CRFs model for Named Entity recognition, let's dive directly into code.

First, we begin by calculating features for each word with a window of 3 words, which means that we also include features of next and previous word.

Here, we calculate features like : word parts, POS tags, word dependencies, lemma, shape (the shape of the word, example: "CRFs" -> "XXXx"), and other boolean variables like : isupper (check if characters are in uppercase), istitle (check if the first character is in uppercase), isdigit (check whether the word consists of digits only), is_stop (check whether the word is a stop word), etc.

Sklearn-crfsuite supports several input formats; here we use feature lists.

The code below was adapted from the [official documentation](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html)



```python
import random
import pycrfsuite
def word2features(train_sample, i):
    token = train_sample[i]
    word = token.text
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.pos='+token.pos_,
        'word.dep='+token.dep_,
        'word.is_stop=%s' %token.is_stop,
        'word.lemma=' + token.lemma_,
        'word.tag=' + token.tag_,
        'word.shape=' + token.shape_,
        'word.is_alpha=%s' %token.is_alpha,        
    ]
    if i > 0:
        token1 = train_sample[i-1]
        word1 = token1.text
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.pos='+token1.pos_,
            '-1:word.dep='+token1.dep_,
            '-1:word.is_stop=%s' %token1.is_stop,
            '-1:word.lemma=' + token1.lemma_,
            '-1:word.tag=' + token1.tag_,
            '-1:word.shape=' + token1.shape_,
            '-1:word.is_alpha=%s' %token1.is_alpha,    
        ])
    else:
        features.append('BOS')
        
    if i < len(train_sample)-1:
        token1 = train_sample[i+1]
        word1 = token1.text
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.pos='+token1.pos_,
            '+1:word.dep='+token1.dep_,
            '+1:word.is_stop=%s' %token1.is_stop,
            '+1:word.lemma=' + token1.lemma_,
            '+1:word.tag=' + token1.tag_,
            '+1:word.shape=' + token1.shape_,
            '+1:word.is_alpha=%s' %token1.is_alpha,   
        ])
    else:
        features.append('EOS')       
    return features

def sent2features(train_sample):
    return [word2features(train_sample, i) for i in range(len(train_sample))]

def encode_labels(labels):
    return ["DISEASE" if label==1 else "O" for label in labels]
```

We use the encode_labels function to encode integer labels (0 or 1) to string labels ("DISEASE" or "O") to respect the target format needed by Sklearn-crfsuite.

```python
random.shuffle(disease_data)
training_data = disease_data[int(0.3*len(disease_data)):]
test_data = disease_data[:int(0.3*len(disease_data))]
#%%
X_train = [sent2features(s[0]) for s in training_data]
y_train = [encode_labels(s[1]) for s in training_data]

X_test = [sent2features(s[0]) for s in test_data]
y_test = [s[1] for s in test_data]
```

Once we defined the training and the test sets, we can begin training the model

```Python
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 0.44,   # coefficient for L1 penalty
    'c2': 1e-4,  # coefficient for L2 penalty
    'max_iterations': 60,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
print("Model's parameters : {}".format(trainer.params()))
trainer.train('models/crf_model.crfsuite')
print("Last iteration log {}".format(trainer.logparser.last_iteration))
```

Out:

![log](/images/uploads/blog2020/CRF_log.png)

We begin by defining a trainer which is mainly our CRFs model, we then define some parameters like c1 and c2 (Coefficients used for regularization) and the max_iteration parameter (used to stop model's iterations). After training, the model is automatically stored in the directory specified in the train function.

# Results and conclusion 

After training the model let's see how to calculate the labels for a given test example:

```Python
tagger = pycrfsuite.Tagger()
tagger.open("models/crf_model.crfsuite")

print("sentence: {}".format(test_data[0][0]))
print("predicted labels: {}". format(tagger.tag(X_test[0])))
print("real labels {}".format(encode_labels(y_test[0])))

```

Out:

![output](/images/uploads/blog2020/output.png)

As we can see, the model predict it right this time, but it may surely miss some disease names. Let's measure the overall performance.

```python
outputs = []
for i in range(len(X_test)):
    outputs.append(tagger.tag(X_test[i]))

targets = sum(y_test, [])
outputs = sum(outputs, [])
outputs = [0 if output=="O" else 1 for output in outputs]

print("conf_matrix: \n", confusion_matrix(targets, outputs))
print("precision score:\n", precision_score(targets, outputs))
print("recall score:\n", recall_score(targets, outputs))
print("F1 score:\n", f1_score(targets, outputs))
```

Out:

![measure](/images/uploads/blog2020/measure.png)

We can see that the precision is much better than recall, which means that the number of false positives (Other names that are detected as disease names) is not that high. Still, there is a lot of disease names that are indetectable by the model. It might be explained by the size of training dataset.

It is worth mentioning that this model is very sensitive to features choice. I recommend the reader to delete or add features of his choice to test the model.

It is true that CRFs are well suited to named-entity recognition task, but other deep learning models have shown a better performance. 

RNNs architectures are known to be able to capture the dependence between input variables. LSTM deep learning models can be a good alternative to try. Some recent works also include contextual embedding of words using attention (BERT, XLNet, etc.) which might boost model's performance.

# Reference

1. [*Classifying Semantic Relations in Bioscience Text*, Barbara Rosario and Marti A. Hearst, in the proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL 2004), Barcelona, July 2004.](https://biotext.berkeley.edu/dis_treat_data.html)
2. [sklearn-crfsuite documentation](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html)
3. [An Introduction to Conditional Random Fields By Charles Sutton and Andrew McCallum](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
4. Anonymization of Unstructured Data Via Named-Entity Recognition by Fadi Hassan Et Al.









 