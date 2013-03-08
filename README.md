accord-census-classifier
========================

Uses Accord.Net C4.5 decision tree to classify UCI Census Income Data Set.  Solution is
designed to be opened with Visual Studio 2012.

## Setup

The required Accord.Net/AForge.net DLLs are in the src/libs/ directory for convenience.  The original DLLs are available via the Accord.Net and AForge.Net installers.  For installation info see the [Getting Started Guide](http://code.google.com/p/accord/wiki/GettingStarted)

Example data is under data/ and is provided by [UCI](http://archive.ics.uci.edu/ml/datasets/Census+Income)

## Running

After building, run the classifier using: AccordCensusClassifier.exe [training data] [test data]
