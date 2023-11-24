# OC_Projet_8

# Deploy a Model in the Cloud
A very young AgriTech start-up named "Fruits!" is looking to offer innovative solutions for fruit harvesting.

The company's goal is to preserve fruit biodiversity by enabling specific treatments for each fruit species and developing intelligent fruit-picking robots.

![Logo](https://github.com/AnniRanok/OC_Projet_8/blob/main/fruits.jpg)


Initially, the start-up wishes to gain recognition by providing the general public with a mobile app that allows users to take a picture of a fruit and obtain information about it.

For the start-up, this app will raise public awareness about fruit biodiversity and establish the first version of a fruit image classification engine.

Moreover, the development of the mobile app will enable the construction of the necessary initial Big Data architecture.


## The Data
Our colleague Paul informs us about a document formalized by an intern who has just left the company. They tested a first approach in an AWS EMR Big Data environment using a dataset of fruit images and associated labels, available for direct download at  [this link](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P8/fruits.zip). The notebook created by the intern will serve as a starting point for building part of the data processing chain.


## The Mission
We are therefore tasked with appropriating the work done by the intern and completing the processing chain.

There is no need to train a model at this time.

The important thing is to establish the first processing steps that will serve when scaling up in terms of data volume!

## Constraints
During his initial briefing, Paul warned us of the following points:

We must consider in our developments that the data volume will increase rapidly after the delivery of this project. Therefore, we will continue to develop PySpark scripts and use the AWS cloud to take advantage of a Big Data architecture (EMR, S3, IAM).

We must demonstrate the setup of an operational EMR instance, as well as explain step by step the PySpark script we will have completed:

A process for broadcasting TensorFlow model weights across clusters (broadcasting the model's "weights") that was overlooked by the intern;
A PCA type dimension reduction step in PySpark.
We will respect GDPR constraints: in our context, we will ensure our setup uses servers located within European territory.

Our critical feedback on this solution will also be valuable before deciding to generalize it.

The implementation of an EMR type Big Data architecture will incur costs. We will therefore ensure to maintain the EMR instance operational only for tests and demos.
