# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) CAPSTONE-Twitter COVID19 Vaccine Posts: Data Labeling and Text Classification

# Problem Statement
Social networks services(SNS) such as Facebook, Twitter, and Instagram serve as great resources for public opinions on different topics. Same applies to COVID19 and its vaccine industries, as COVID virus continues to live through in our communities. However, such open-text data are hard to be used as it is in machine learning or deep learning process as they usually don't have necessary y-labels to predict on. The goal of this project is to **find out an optimal labeling method for twitter posts regarding COVID19 vaccines** to classify them into "pro-vaccination" vs "anti-vaccination". With the proper labeling method established, it can have various future usecases such as  predicting future vaccine outlook or user-specific advertising based on their view on the vaccines.  

# Background
COVID19 pandemic marks an unusual case in the modern world; because its unexpected rise and unusual mandates that many governments implemented, public opinion has been conflicting over how to deal with the pandemic. One of the biggest arguments regarding the topic is whether or not to require or mandate the vaccines, and the United States public has been divided over whether or not to obtain COVID-19 vaccines. 

Social network services serve as great resources for examining such public opinions, and predicting vaccine outlooks; yet, the open-source nature makes it difficult to take technical approach to these data. In the specific case of whether the writer of the post will be pro or anti vaccine is easier to determine with our human eyes, but it would be too time consuming and inefficient for a man to sit and try to label each of the millions of SNS text data. 

One mode of labeling open-source sns text data is by specific hashtags. In such case, hashtags can serve as a method of crowdsource labeling, that is given by individuals who upload the posts. In order to obtain such "crowdsource" labeled data via hashtag, one can scrape sns data based on hashtaged parameters. 

There are other programmatic approaches that data scientists and machine learning scientists take in order to tackle this problem, and [Snorkel](https://www.snorkel.org/) is one of them. Snorkel's labeling function operator allows us to programmatically label such dataset by incorporating and voting on various Labeling functions that represent many heuristic and/or noisy strategies aka weak supervision. Snorkel puts together these low-accuracy, simple labeling functions, reweights, and combines the output labels for final labeling.

This project will examine effectiveness and accuracy these two types of labeling methods on twitter posts that discuss COVID vaccines. By successfully achieving this, the open-source tweets can gain future usage such as predicting the vaccine rollout. 

# Data
All datasets were collected using [SNScrape API](https://github.com/JustAnotherArchivist/snscrape) by specifying the keywords and hashtags(as needed).

   1. [Hashtag_labeled tweets](data/tweets_hash_label.csv)
   2. [Unlabeled tweets](data/tweets_unlabeled.csv)

Then, Snorkel labeling functions were used to label the unlabeled data.
   3. [Snorkel_labeled tweets](data/snorklabeled_tweets_train.csv)


# Results

**Snorkel Labeling Function**
- Manually labeled development set (300 rows): 59.2%
- Subset of hashtag labeled dataset (1000 rows): 60%

The accuracy for both are pretty low. For the manually labeled dataset, the accuracy is lower than the baseline which is at about 0.7. However, the hashtag labeled dataset shows higher than the baseline accuracy. The low accuracy in the manually labeled development set may be due to very small sample size. To keep the coverage, the function was applied to the unlabeled dataset. 

- **Coverage**: 75,644 out of 111,959 rows
- **label distribution**: pro_vax(0.86), anti_vax(0.14)

The coverage was 68% and yielded decently large labeled dataset, as the original dataset was big. The labeling function yielded very imbalanced data, likely because the labeling functions for class 0 did not have the high coverage. The imbalanced data was taken care of in the modeling process. 

**Hashtag Labeled Model**
- TF_IDF with logistic regression

|          | precision | recall | f1-score |
|----------|-----------|--------|----------|
| 0        | 0.75      | 0.82   | 0.78     |
| 1        | 0.79      | 0.72   | 0.76     |
| accuracy |           |        | 0.77     |


**Snorkel Labeled Model**
- TF_IDF with XGBoost

|          | precision | recall | f1-score |
|----------|-----------|--------|----------|
| 0        | 0.61      | 0.84   | 0.78     |
| 1        | 0.97      | 0.92   | 0.94     |
| accuracy |           |        | 0.91     |

**Other Models**
- BERT

With both the hashtag-labeled and Snorkel-labeled dataset, the best performing model does beat the baseline accuracy.

TFIDF vectorizing with regularized logistic regression on Hashtag labeled data scored 0.77 accuracy, where baseline score was 0.5. Recall for the 0 class label (anti_vax) was higher than that of 1 class, meaning that the model had higher coverage of 0 class prediction. However, the precision was slightly lower, likely indicating that the model tends to overestimate the anti_vax texts.

Predicting Snorkel labeled data (TFIDF vectors with XGBoost Classification) scored 0.91 in accuracy. Considering the baseline accuracy of 0.87, the model beat the baseline by about 0.04. Referring to the confusion matrix, the model over-predicted on the class 0 anti_vax, lowering its precision. There seems to be some trade off between 0 class precision and its recall of score 0.84.

# Conclusions & Next Steps
----
The goal of this project is to **find out an optimal labeling method for twitter posts regarding COVID19 vaccines** to classify them into "pro-vaccination" vs "anti-vaccination". With the proper labeling method established, it can have various future usecases such as  predicting future vaccine outlook or user-specific advertising based on their view on the vaccines.  

----
There seems to be many caveats in terms of choosing the optimal labeling method, as each one has its own short comings. Although it is easy to obtain the labels, the hashtag labeling method limits the collection of text to posts with decently trending hashtags. Texts with hashtags could be somewhat contextually different from any open-source tweets, and considering how predictive models work the best with the data similar to the trained data, the accuracy of the application could be limited to the similarly hashtag labeled data. 

In order to see its applicability to completely open-sourced data, my next step on this model would be to apply this model to my unlabeled dataset and comparing across the two different labels. 

Strictly Looking at the accuracy of the model, the snorkel labeled model with Tf-Idf Vectorizer and XGBoost performed better. Even with the highly imbalanced dataset towards pro_vax group(class 1) the recall score for anti-vax was pretty high at 0.84, although its precision was low at 0.64. It's precision and recall for pro-vax class was decent all at 90%. This means the model is optimized at predicting the pro-vax twitter population, and further development of this model can have possible application of predicting the vaccine outlook. 

Aside from the model performance, however, it is slightly questionable how the model decently performed on the snorkel labeled dataset. This could be some investigation point in the next step for Snorkel labeled model, to look into whether the model is picking up the features from data with wrong label or if the Snorkel development set accuracy simply underestimated the accuracy of Snorkel labeling function.
