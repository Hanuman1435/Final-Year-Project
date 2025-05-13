import serial
import joblib
import numpy as np
import time

# Load trained Random Forest model
model = joblib.load('rf_model.pkl')

# Configure the serial port (adjust 'COM3' or '/dev/ttyUSB0' as needed)
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

print("Listening to Arduino...")

while True:
    try:
        if arduino.in_waiting:
            line = arduino.readline().decode('utf-8').strip()
            if line:
                print(f"Received: {line}")
                try:
                    values = list(map(float, line.split(',')))
                    #print(len(values))
                    if len(values) >= 4:
                        input_features = np.array([values])
                        prediction = model.predict(input_features)[0]
                        
                        # Send prediction back to Arduino
                        arduino.write(f"{prediction}\n".encode())
                        print(f"Sent prediction: {prediction}")
                    else:
                        print("Invalid data format from Arduino")
                except ValueError as ve:
                    print(f"Value error: {ve}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        break
