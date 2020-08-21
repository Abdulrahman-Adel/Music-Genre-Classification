# Music-Genre-Classification

Classifying 1.5 GB of audio files into 10 classes

-----

## Dataset
<!-- Link -->
The [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. It consists of 1000 audio files each having 30 seconds duration. There are 10 classes ( 10 music genres) each containing 100 audio tracks. Each track is in .wav format. 

----

## Approach 1

I've created a dataset by extracting features of every indiviual audio file, these features are [Spectral Centroid, Spectral Rolloff, Spectral Bandwidth,  Zero-Crossing Rate, Mel-Frequency Cepstral Coefficients(MFCCs), Chroma feature], and using these feature for prediction.

----

## Approach 2

By using A visual representation for each audio file (Spectograms). Classifying the music genre with CNN 
<!-- Image -->
![pop00005](https://user-images.githubusercontent.com/57441828/90922266-fd9d8500-e3eb-11ea-9f6b-5f2c94004c6c.png)
Des: Spectogram of sample Pop song

-----

## Libraries
<!-- UL -->
* librosa
* IPython
* os
* keras
* pandas
* numpy
* matplotlib
* csv

-----

## Classes
<!-- UL -->
* Blues
* Classical
* Country
* Disco
* Hiphop
* Jazz
* Metal
* Pop
* Reggae
* Rock
