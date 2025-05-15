import serial
import joblib
import numpy as np
import pandas as pd
import time

# Load trained Random Forest model
model = joblib.load('rf_model.pkl')

# Configure the serial port (change 'COM3' to your port, e.g., 'COM4' or '/dev/ttyUSB0' on Linux)
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

                    # Ensure at least 5 values received
                    if len(values) >= 5:
                        # Extract the 4 features in correct order
                        battery_voltage = values[0]
                        load_voltage = values[1]
                        set_voltage = values[2]
                        pwm_duty_cycle = values[4]  # Skipping index 3 (e.g., motor voltage)

                        # Use DataFrame to avoid feature name warning
                        input_df = pd.DataFrame([[battery_voltage, load_voltage, set_voltage, pwm_duty_cycle]],
                                                columns=["battery_voltage", "load_voltage", "set_voltage", "pwm_duty_cycle"])
                        
                        prediction = model.predict(input_df)[0]

                        # Send prediction back to Arduino
                        arduino.write(f"{prediction}\n".encode())
                        print(f"Sent prediction: {prediction}")
                    else:
                        print("Invalid data format from Arduino (need 5 values)")

                except ValueError as ve:
                    print(f"Value error: {ve}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break
