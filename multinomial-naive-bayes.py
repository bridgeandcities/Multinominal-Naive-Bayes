#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load in the training set .csv
training_set = pd.read_csv("trg.csv")
training_set.head()


# In[2]:


# Process the text, find a 'good model' with cross-validation
print("Text processing...")
import re

# Text processing by removing symbols
replaceSpace = re.compile('[/(){}\[\]\|@,;]')
replaceSymbols = re.compile('[^a-z #+_]')
stopWordsList = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# cleans the text and splits into words with most common english words removed
def clean_text(text):
    text = text.lower() # lowercase text
    text = replaceSpace.sub(' ', text) # replace symbols by space in text
    text = replaceSymbols.sub('', text) # delete symbols from text
    text = [word for word in text.split() if word not in stopWordsList]
    return text
    
training_set['abstract'] = training_set['abstract'].apply(clean_text)
training_set.head()


# In[3]:


# Train the NBC with this data (your own NBC code)
print("Training the NBC...")

# Create dictionaries of words in each class
from collections import Counter

bigListA = training_set.loc[training_set['class'] == 'A', 'abstract'].sum()
bigDictA = Counter(bigListA)
bigListB = training_set.loc[training_set['class'] == 'B', 'abstract'].sum()
bigDictB = Counter(bigListB)
bigListE = training_set.loc[training_set['class'] == 'E', 'abstract'].sum()
bigDictE = Counter(bigListE)
bigListV = training_set.loc[training_set['class'] == 'V', 'abstract'].sum()
bigDictV = Counter(bigListV)

# word list for each class
bigListALen = len(bigListA)
print("total number of words in A: ", len(bigListA))
bigListBLen = len(bigListB)
print("total number of words in B: ", len(bigListB))
bigListELen = len(bigListE)
print("total number of words in E: ", len(bigListE))
bigListVLen = len(bigListV)
print("total number of words in V: ", len(bigListV))

print("total words over all classes: ", bigListALen + bigListBLen + bigListELen + bigListVLen)

# unique dict
bigDictALen = len(bigDictA)
print("unique words in A: ", len(bigDictA))
bigDictBLen = len(bigDictB)
print("unique words in B: ", len(bigDictB))
bigDictELen = len(bigDictE)
print("unique words in E: ", len(bigDictE))
bigDictVLen = len(bigDictV)
print("unique words in V: ", len(bigDictV))

uniqueOverAllClasses = len(bigDictA) + len(bigDictB) + len(bigDictE) + len(bigDictV)
print(uniqueOverAllClasses)


# In[4]:


# Calculating priors P(c)
import math

classCountDict = training_set['class'].value_counts().to_dict()
print(classCountDict)

numberOfColumns = len(training_set.index)
print(numberOfColumns)

priorDict = {}

# prior ratios
for key in classCountDict:
    prior = (classCountDict[key]/numberOfColumns)
    #print(prior)
    classCountDict[key] = prior
    priorDict[key] = math.log10(prior)

print(classCountDict)
print(priorDict)


# In[5]:


# Calculating conditional probabilities P(w|c)

condLogDictA = {}
condLogDictB = {}
condLogDictE = {}
condLogDictV = {}

for word in bigDictA:
    log_answer = math.log10((bigDictA[word] + 1) / (bigListALen + uniqueOverAllClasses))
#   print(word, bigDictA[word], log_answer)
    condLogDictA[word] = log_answer
#   print(condLogDictA[word])
for word in bigDictB:
    log_answer = math.log10((bigDictB[word] + 1) / (bigListBLen + uniqueOverAllClasses))
    condLogDictB[word] = log_answer
for word in bigDictE:
    log_answer = math.log10((bigDictE[word] + 1) / (bigListELen + uniqueOverAllClasses))
    condLogDictE[word] = log_answer
for word in bigDictV:
    log_answer = math.log10((bigDictV[word] + 1) / (bigListVLen + uniqueOverAllClasses))
    condLogDictV[word] = log_answer
    
condLogSmoothingA = math.log10(1 / (bigListALen + uniqueOverAllClasses))
condLogSmoothingB = math.log10(1 / (bigListBLen + uniqueOverAllClasses))
condLogSmoothingE = math.log10(1 / (bigListELen + uniqueOverAllClasses))
condLogSmoothingV = math.log10(1 / (bigListVLen + uniqueOverAllClasses))

print(condLogSmoothingA, condLogSmoothingB, condLogSmoothingE, condLogSmoothingV,)


# In[6]:


def classify(abstract):
    classifiedList = []
    abstract = abstract.apply(clean_text)
    #print(abstract)
    for entry in abstract:
        abstractClassDict = {'A':priorDict['A'], 'B':priorDict['B'], 'E':priorDict['E'], 'V':priorDict['V']}
        for word in entry:
            #print("word: ", word)
            if condLogDictA.get(word):
                abstractClassDict['A'] += condLogDictA[word]
            else:
                abstractClassDict['A'] += condLogSmoothingA
            if condLogDictB.get(word):
                abstractClassDict['B'] += condLogDictB[word]
            else:
                abstractClassDict['B'] += condLogSmoothingB
            if condLogDictE.get(word):
                abstractClassDict['E'] += condLogDictE[word]
            else:
                abstractClassDict['E'] += condLogSmoothingE
            if condLogDictV.get(word):
                abstractClassDict['V'] += condLogDictV[word]
            else:
                abstractClassDict['V'] += condLogSmoothingV
        #print(abstractClassDict)
        abstractClass = max(abstractClassDict,key=abstractClassDict.get)
        #print("final class: ", abstractClass)
        classifiedList.append(abstractClass)
    #print(classifiedList)
    return classifiedList


# In[7]:


'''training'''
# classify using multinominal naive bayes classifier
# Use this 'good model' to generate classifications.  
test_set = pd.read_csv("trg.csv")

# Apply the model to the test set
test_set_class_predictions = classify(test_set["abstract"])
#print(test_set_class_predictions)
test_set["class"] = test_set_class_predictions


# Write the test set classifications to a .csv so it can be submitted to Kaggle
test_set.drop(["abstract"], axis = 1).to_csv("trg_test.csv", index=False)
#print(test_set)


# In[8]:


'''kaggle'''
# classify using multinominal naive bayes classifier
test_set = pd.read_csv("tst.csv")

# Apply the model to the test set
test_set_class_predictions = classify(test_set["abstract"])
#print(test_set_class_predictions)
test_set["class"] = test_set_class_predictions


# Write the test set classifications to a .csv so it can be submitted to Kaggle
test_set.drop(["abstract"], axis = 1).to_csv("tst_kaggle_2.csv", index=False)
#print(test_set)


# In[ ]:
