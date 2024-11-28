import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

os.chdir("C://Users/dipra/source/repos/FarmAI App/FarmerAI")
data = pd.read_csv('TEMP_ANNUAL_SEASONAL_MEAN.csv')

#Train Variables
X = data[['YEAR', 'JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']]
y = data['ANNUAL']

#Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Synthesis
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#Compile
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#Training
history = model.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

#Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)               

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

#Main Predict
last_year = data['YEAR'].iloc[-1]
next_year = last_year + 1

#Backend data
next_year_data = pd.DataFrame({
    'YEAR': [next_year],
    'JAN-FEB': [data['JAN-FEB'].mean()],
    'MAR-MAY': [data['MAR-MAY'].mean()],
    'JUN-SEP': [data['JUN-SEP'].mean()],
    'OCT-DEC': [data['OCT-DEC'].mean()]
})

#Normalize
next_year_scaled = scaler.transform(next_year_data)

#Prediction
next_year_prediction = model.predict(next_year_scaled)

print(f"The predicted Annual Temperature {next_year} is {next_year_prediction[0][0]}")

#Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(data['YEAR'], data['ANNUAL'], label='Historical Data', color='blue')
plt.scatter([next_year], [next_year_prediction[0][0]], label='Next Year Prediction', color='red')
plt.plot(data['YEAR'], data['ANNUAL'], label='Historical Trend', color='green')
plt.xlabel('Year')
plt.ylabel('Annual Temperature')
plt.title('Temperature Prediction for Next Year')
plt.legend()
plt.show()
