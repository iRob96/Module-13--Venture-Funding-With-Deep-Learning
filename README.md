# Module-13--Venture-Funding-With-Deep-Learning

## Goal

You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

Your task is To predict whether Alphabet Soup funding applicants will be successful, you will create a binary classification model using a deep neural network.

This challenge consists of three technical deliverables. You will do the following:

Preprocess data for a neural network model.

Use the model-fit-predict pattern to compile and evaluate a binary classification model.

Optimize the model.

## Optimization

I attempted to optimze the first model by increasing the nodes and using 50 epochs. This slightly increased the accuracy each time.

On the second model I tried to optimize it by decreasing the nodes and using 100 epochs. This slightly decreased the accuracy each time.

Neural Network Model<img width="1440" alt="Screen Shot 2022-11-30 at 11 21 30 PM" src="https://user-images.githubusercontent.com/107821891/204965548-3133e48b-b4e2-400a-86ee-2260849ebc32.png">


Optimization 1<img width="1440" alt="Screen Shot 2022-11-30 at 11 21 52 PM" src="https://user-images.githubusercontent.com/107821891/204965580-4954f3d6-c041-4f5c-bec4-a2ffda78fb10.png">


Optimization 2<img width="1440" alt="Screen Shot 2022-11-30 at 11 22 10 PM" src="https://user-images.githubusercontent.com/107821891/204965601-a7abfaef-5347-4940-95dc-16301ba6ddbb.png">

## Tech

GoogleColab
Python

## Libraries

import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
