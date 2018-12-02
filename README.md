# An RNN Architecture for Predicting Clinical Risks

Introduction

The problem for this project is predicting how many patients with a specific disease will come to a hospital in a certain date. 
Predicting the number of patients could have many advantages for hospitals. For example, if the model predicts increasing the risk of flu in winter, then the government can arrange some programs for flu vaccinations in schools. Also, hospitals will get ready for accepting many flu patients in that period 
of time.
In this work, I developed a deep model to predict how many patients will visit the hospital with a certain disease in a specific day. I used LSTM architecture that can encode the similarity of two sequences and dynamically match temporal patterns in the hospital data.
Preparing Data

There are several CSV files for different hospitals. Each file consists information of five years in seven columns. Each column refers to count of a level of disease (1 to 5, then 2 more at critical levels) for a day. The numbers in each row represent the number of patients for each day.
I used LSTM (Long Short-Term Memory) recurrent neural network to implement this project because LSTM has the promise of learning long sequences of observations. It seems a perfect match for time series forecasting. 
My proposed method uses the dataset to train deep learning network and then make predictions on a test dataset to verify the accuracy of the prediction.
Performing deep learning relies on a train and a test dataset. The train dataset is used to train the model, by pairing an input with its expected output. The test dataset is to estimate how well the model has been trained. I used the first 1000 data (rows) as the training dataset. Also, 300 rows used as 
test dataset.
The process to make data ready for the LSTM model includes of three main steps, namely transform the time series data to stationary data, transform the time series to a supervised learning problem, and transform the observations to have a specific scale. In the following paragraphs, I explain how I achieved these three steps.
1) Transform the time-series data to stationary data:
Stationary data do not have any trend and are easier to model and will very likely result in more skillful forecasts. One way to remove the trend is by differencing the data. Therefore, the observation from the previous time step is subtracted from the current observation. Also, after model applied, to find the correct numbers it is needed to invert this process.
2) Transform the time series to a supervised learning problem: 
The LSTM model assumes that data is divided into input (X) and output (y) components. However, in a system with time series, there is only one data in each timestamp. To have input data and output data in each timestamp, I used the observation from the last timestamp as the input and the observation at the current timestamp as the output.
3) Transform the observations to have a specific scale:
Neural networks expect data to be within the scale of the activation function used by the network. The default activation function for LSTM is the hyperbolic tangent (tanh), with the range of (-1,1) which is the preferred range for the time series data. To make the experiment fair, the scaling coefficients (min and max) values must be calculated on the training dataset and applied to scale the test dataset and any forecasts. This is to avoid contaminating the experiment with knowledge from the test dataset, which might give the model a small edge. To transform the dataset to the range (-1,1), I used the MinMaxScaler class. Like other scikit-learn transform classes, it requires data provided in a matrix format with rows and columns. Therefore, I reshaped the NumPy arrays before transforming.
 I inverted the scale on forecasts to return the values back to the original scale so that the results can be interpreted, and a comparable error score can be calculated.
Predicting Model

I used Keras library to implement the LSTM model for this project. The LSTM model that I implemented, takes the last observation from the training data and history accumulated by walk-forward validation and use that to predict the current time step. A batch of data is a fixed sized number of rows from the training dataset that defines how many patterns to process before updating the weights of the network. State in the LSTM layer between batches is cleared by default, therefore, if we want to use the previous states, then we must make the LSTM stateful. This gives us fine-grained control over when state of the LSTM layer is cleared, by calling the reset-states() function.
I considered 30 neurons as the number of memory units. Batch size represents the number of train data points. Because the network train one step at a time and predict one step forward, I set the batch size as one. For each value in train dataset (X-train) there are one corresponding value which is the next value in the sequence (y-train). Then, for training, each value in the (X-train) is fed one by one and update weights by using an optimization method. After training the network there is an RNN that can predict one step ahead for each input.

