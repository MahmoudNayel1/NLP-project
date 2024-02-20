



# MAHMOUD MOHAMED HASSAN   20045532



import pandas as pd
import numpy as np
from nltk.corpus import stopwords   # For Removing Stopwords (For Unimportant Words)
import nltk
nltk.download('stopwords')
import re                           # Regular Expressions
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# reading excel file
Data_File = pd.read_csv("Barcelona_reviews.csv")
Labels = Data_File['sample']
Data = Data_File['review_full']

Positive_ID = []      # indices of positive labels
Negative_ID = []      # indices of Negative labels


# getting the ID of First 3000 Positive and 3000 Negative Values.
for label_count in range(len(Labels)):
    if Labels[label_count] == "Positive":
        if len(Positive_ID) < 3000:
            Positive_ID.append(label_count)
    else:
        if len(Negative_ID) < 3000:
          Negative_ID.append(label_count)

    if len(Negative_ID) == 3000 and len(Positive_ID) == 3000:
        break


# Spliting Data and its Labels into Train , Test , Validation  (50% positive , 50% Negative)
Train_Data = []      # 70% of the data set
Test_Data = []       # 20% of the data set
Valid_Data = []      # 10% of the data set

Train_Labels = []
Test_Labels = []
Valid_Labels = []



# Train Data = 4200 ( 2100 Positive , 2100 Negative )
# Test = 1200 ( 600 Positive , 600 Negative )
# Validation = 600 ( 300 Positive , 300 Negative )
for i, POS_lbl in enumerate(Positive_ID):
    if i < 2100:
        Train_Data.append(Data[POS_lbl])
        Train_Labels.append(Labels[POS_lbl])
    elif i >= 2100 and i < 2700:
        Test_Data.append(Data[POS_lbl])
        Test_Labels.append(Labels[POS_lbl])
    else:
        Valid_Data.append(Data[POS_lbl])
        Valid_Labels.append(Labels[POS_lbl])

for i , NEG_lbl in enumerate(Negative_ID):
    if i < 2100:
        Train_Data.append(Data[NEG_lbl])
        Train_Labels.append(Labels[NEG_lbl])
    elif i >= 2100 and i < 2700:
        Test_Data.append(Data[NEG_lbl])
        Test_Labels.append(Labels[NEG_lbl])
    else:
        Valid_Data.append(Data[NEG_lbl])
        Valid_Labels.append(Labels[NEG_lbl])




def Word_Extraction(Sentence):
    Words = re.sub("[^\w]" , " " , Sentence).split()           # Replacing the non word character by Spaces.
    Cleaned_Text = [W.lower() for W in Words if W not in stopwords.words('english')]  # Normalization (Transformed all the text to lowcase no CAPS) and Stopwords removal
    return Cleaned_Text


def BOW_VOCAB (Sentences):
    Words = []
    for sentence in Sentences:
        W = Word_Extraction(sentence)                  # W is a list of words after normalization and stopwords removal
        Words.extend(W)
    Words = list(set(Words))                           # Transforming list into set to store only unique values
    return Words


def BAG_OF_WORDS_TRAIN(all_sentences):
    Vocab = BOW_VOCAB(all_sentences)                   # Main Vocab : the list thet has the all the unique words
    all_sentences_vocab = []                           # Vocab of all sentences
    for sentence in all_sentences:
        words = Word_Extraction(sentence)
        BOW_VECTOR = np.zeros(len(Vocab))              # vocab of sentence initialized by zero
        for W in words:
            for i, Word in enumerate(Vocab):           # searching for the word and its index in the vocab
                if Word == W:                          # increasing the index of the word by 1
                    BOW_VECTOR[i] += 1
        all_sentences_vocab.append(BOW_VECTOR)
    return all_sentences_vocab , Vocab


def BAG_OF_WORDS_TEST_VALID(all_sentences , Vocab):
    all_sentences_vocab = []                            # Vocab of all sentences
    for sentence in all_sentences:
        words = Word_Extraction(sentence)
        BOW_VECTOR = np.zeros(len(Vocab))               # vocab of sentence initialized by zero
        for W in words:
            for i, Word in enumerate(Vocab):            # searching for the word and its index in the vocab
                if Word == W:                           # increasing the index of the word by 1
                    BOW_VECTOR[i] += 1
        all_sentences_vocab.append(BOW_VECTOR)
    return all_sentences_vocab



All_Sentances_Vocab , Global_Vocab = BAG_OF_WORDS_TRAIN(Train_Data)      # Getting the Vocab of each sentence in Train Data
All_Train_Data = []

# Loop for assigning each Data to its Label for shuffling
for i in range(len(All_Sentances_Vocab)):
    All_Train_Data.append([All_Sentances_Vocab[i], Train_Labels[i]])

np.random.shuffle(All_Train_Data)        # Shuffling Data

Final_TrainData = []
Final_TrainLabel = []

for data, label in All_Train_Data:         # Splitting Data and Labels after Shuffling
    Final_TrainData.append(data)
    Final_TrainLabel.append(label)



# Naive Bayes Classifire
Classifire = MultinomialNB()

Classifire.fit(Final_TrainData, Final_TrainLabel)    # Classifire Training


# Getting Accuracy of Train Data Predictions
Train_Predictions = Classifire.predict(Final_TrainData)
Train_Accuracy = accuracy_score(Final_TrainLabel , Train_Predictions)
print("Train Accuracy = " +str(Train_Accuracy*100))


# Getting the vocab of each sentence in Validation Data
# Getting Accuracy of valid Data Predictions
Valid_Vocab = BAG_OF_WORDS_TEST_VALID(Valid_Data, Global_Vocab)
Valid_Prediction = Classifire.predict(Valid_Vocab)
Valid_Accuracy = accuracy_score(Valid_Labels, Valid_Prediction)
print("Validation Accuracy = " +str(Valid_Accuracy*100))


# Getting the vocab of each sentence in Testing Data
# Getting Accuracy of Testing Data Predictions
Test_Vocab = BAG_OF_WORDS_TEST_VALID(Test_Data , Global_Vocab)
Test_Predictions = Classifire.predict(Test_Vocab)
Test_Accuracy = accuracy_score(Test_Labels, Test_Predictions)
print("Test Accuracy = " +str(Test_Accuracy*100))




# Reviews checker That checks if the Review i provide the Model with is Positive or Negative
Review = input("Please Enter review: ")
Review = [Review]
Test_Vocab = BAG_OF_WORDS_TEST_VALID(Review, Global_Vocab)
Prediction = Classifire.predict(Test_Vocab)
print("This Review is: ")
print(Prediction)












