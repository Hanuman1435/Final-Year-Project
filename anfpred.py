import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf_model.pkl')

print("Random Forest Prediction Started (Ctrl+C to stop)\n")

while True:
    try:
        battery_voltage = float(input("Enter battery voltage: "))
        load_voltage = float(input("Enter load voltage: "))
        set_voltage = float(input("Enter set voltage: "))
        pwm_duty_cycle = float(input("Enter PWM duty cycle (%): "))

        input_features = np.array([[battery_voltage, load_voltage, set_voltage, pwm_duty_cycle]])
        prediction = model.predict(input_features)

        if prediction[0] == 1:
            print("Prediction: Load Voltage < Set Voltage → OUTPUT = 1\n")
        else:
            print("Prediction: Load Voltage >= Set Voltage → OUTPUT = 2\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break
    except Exception as e:
        print(f"Invalid input: {e}\n")
