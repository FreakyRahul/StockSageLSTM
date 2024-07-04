import numpy as np
import pandas as pd
import yfinance as yf
from  keras.models import load_model 
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras import initializers

model = load_model(r"pathname")

st.header("Stock Market Predictior")

stock = st.text_input("Enter Stock Symbol","GOOG")
start = "2013-01-01"
end = "2023-12-31"

data = yf.download(stock, start, end)

st.subheader("Stock Data")
st.write(data)


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

#scaling

sc = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days , data_test], ignore_index = True)
data_test_scaled = sc.fit_transform(data_test)

st.subheader("Price vs MA50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,"r",label="MA50")
plt.plot(data.Close,"g",label="Price")
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader("Price vs MA50 vs MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,"r",label="MA50")
plt.plot(ma_100_days,"b",label="MA100")
plt.plot(data.Close,"g",label="Price")
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader("Price vs MA100 vs MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,"r",label="MA100")
plt.plot(ma_200_days,"b",label="MA200")
plt.plot(data.Close,"g", label="Price")
plt.legend()
plt.show()
st.pyplot(fig3)


x = []
y = []
for i in range(100,data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i,0])
    
x,y = np.array(x),np.array(y)

predict = model.predict(x)

scale = 1/sc.scale_

predict = predict * scale

y  = y*scale

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,"r",label="Predicted Price")
plt.plot(y,"g",label="Original Price")
plt.legend()
plt.show()
st.pyplot(fig4)