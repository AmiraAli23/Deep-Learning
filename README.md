# LSTM Stock Predictor

![image](https://user-images.githubusercontent.com/99091066/172749313-7fb28d9c-15a4-478d-9a80-b00085e86b9d.jpeg)


In this assignment, I will use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.


## Prepare the data for training and testing

For both data files , [btc historic](https://github.com/AmiraAli23/Deep-Learning/blob/87358f40159047e02de42421a6729a0919cadd00/btc_historic.csv) and [btc sentiment](https://github.com/AmiraAli23/Deep-Learning/blob/87358f40159047e02de42421a6729a0919cadd00/btc_sentiment.csv), I joined and cleaned the data. 

Each model used 70% of the data for training and 30% for testing. To reflect this, i used the following code:

```` python

split=int(0.7*len(X))

`````

I then scaled and reshaped the data: 

```` python
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

y_train=scaler.transform(y_train)
y_test=scaler.transform(y_test)

X_train=X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test=X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

````

## Build and train custom LSTM RNNs

Output summary for the FNG Model :


<img width="516" alt="Screen Shot 2022-06-08 at 10 25 27 PM" src="https://user-images.githubusercontent.com/99091066/172750927-a2cb8398-3b40-45ca-ad64-145450f6a5db.png">

Output summary for the model using Closing Prices:

<img width="514" alt="Screen Shot 2022-06-08 at 10 26 24 PM" src="https://user-images.githubusercontent.com/99091066/172751054-887cf645-3c82-4840-bff9-4b615c475986.png">



## Evaluate the performance of each model

### Which model has a lower loss?

Model using FNG prices:

<img width="76" alt="Screen Shot 2022-06-08 at 11 18 36 PM" src="https://user-images.githubusercontent.com/99091066/172756922-83001085-f6c3-4b1d-b793-ea902f3cee86.png">

Model using closing prices: 

<img width="89" alt="Screen Shot 2022-06-08 at 11 19 02 PM" src="https://user-images.githubusercontent.com/99091066/172756970-d311b4c2-29ac-4b1f-be1e-73a6ab324534.png">


Comparing the two, the model using the closing prices only has a lower loss `0.0290` vs `0.1710`

### Which model tracks the actual values better over time?

Model using FNG prices:

<img width="398" alt="Screen Shot 2022-06-08 at 10 37 39 PM" src="https://user-images.githubusercontent.com/99091066/172752324-e7393b16-c949-4f6b-98c4-b4908150a91a.png">

  > The predicted line and real line do not seem correlated at all.


Model using closing prices: 

<img width="391" alt="Screen Shot 2022-06-08 at 10 38 49 PM" src="https://user-images.githubusercontent.com/99091066/172752446-84d83c14-be49-44e3-afde-6323c3ede246.png">

  > The predicted line is very close to the real line. 


Based on these graphs, the model with the closing prices tracks actual values better over time. 

### Which window size works best for the model?

The graphs above were for the window size 10.

### Window size 4

FNG:

<img width="391" alt="Screen Shot 2022-06-08 at 10 47 03 PM" src="https://user-images.githubusercontent.com/99091066/172753378-d9889f9b-7244-498b-b567-f6240dab2b86.png">


Closing Prices:

<img width="396" alt="Screen Shot 2022-06-08 at 10 50 30 PM" src="https://user-images.githubusercontent.com/99091066/172753831-25cada5d-48d0-421c-a9d2-d5d989fb47e2.png">

### Window size 6 

FNG:

<img width="387" alt="Screen Shot 2022-06-08 at 10 54 31 PM" src="https://user-images.githubusercontent.com/99091066/172754285-3a25f930-dd44-435a-8198-1e096088063d.png">


Closing Prices: 

<img width="391" alt="Screen Shot 2022-06-08 at 10 55 12 PM" src="https://user-images.githubusercontent.com/99091066/172754346-c7ef14c0-82f7-4846-be96-02e3c79925e5.png">


By changing the window from 10 to 4 and 6, it is clear that a larger window size corresponds with smoother predicted lines. At window size 4, the predicted line follows the same harsh fluctuations as the real values. Decreasing the window size makes the predicted line more similar to the real line and thus, a more accurate representation. 






