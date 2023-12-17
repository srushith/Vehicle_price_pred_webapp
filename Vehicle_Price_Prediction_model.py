import pickle
import streamlit as st
import numpy as np

# Load the model
model = pickle.load(open("vehicle_pred_model1.pkl", 'rb'))

def convert_to_native_types(data):
    converted_data = {}
    for key, value in data.items():
        try:
            # Attempt to convert the value to a native Python type
            converted_data[key] = np.asscalar(np.float64(value)) if isinstance(value, np.float32) else value
        except Exception as e:
            converted_data[key] = value
    return converted_data

def main():
    st.markdown(
    "<h1 style='text-align: center; color: brown; font-size: 50px;'>Vehicle Price Prediction</h1>",
    unsafe_allow_html=True,
)

    col1, col2 = st.columns(2)

    with col1:
        year = st.text_input('Year')
        fuel_consumption = st.text_input('Fuel Consumption per 100km')
        kilometers = st.text_input('Kilometers')
        brand = st.selectbox('Brand', ['Alfa', 'Audi', 'BMW', 'Chrysler', 'Citroen', 'Fiat', 'Ford',
                                       'GWM', 'Great', 'HSV', 'Haval', 'Holden', 'Honda', 'Hyundai',
                                       'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'LDV', 'Land', 'Lexus', 'MG',
                                       'Maserati', 'Mazda', 'Mercedes-Benz', 'Mini', 'Mitsubishi',
                                       'Nissan', 'Peugeot', 'Porsche', 'Ram', 'Renault', 'Skoda',
                                       'Ssangyong', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo', 'other'])
        transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])

    with col2:
        drive = st.selectbox('DriveType', ['Front', 'AWD', '4WD', 'Rear'])
        fuel = st.selectbox('Fuel Type', ['Premium', 'Diesel', 'Hybrid', 'Unleaded', 'LPG', 'Leaded'])
        body_style = st.selectbox('Body Style', ['Hatchback', 'Sedan', 'Wagon', 'Coupe', 'SUV', 'Convertible',
                                                 'Commercial', 'Ute / Tray', 'People Mover'])
        seats = st.selectbox('Number of Seats', ['2', '3', '4', '5', '6', '7', '8', '9', '11', '14'])
        engine = st.selectbox('Engine', ['0.7', '0.9', '1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8',
                                          '1.9', '2', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8',
                                          '2.9', '3', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8',
                                          '4', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '5',
                                          '5.3', '5.4', '5.5', '5.7', '5.9', '6', '6.1', '6.2', '6.4', '6.8'])

    if st.button('Predict'):
        data = {
            'year': year,
            'fuelConsump_per_100km': fuel_consumption,
            'kilometers': kilometers,
            'brand': brand,
            'transmission': transmission,
            'drive': drive,
            'fuel': fuel,
            'body_style': body_style,
            'seats': seats,
            'engine': engine
        }

        data = convert_to_native_types(data)

        # Adjust column names here according to your model
        column_names = ['year', 'fuelConsump_per_100km', 'kilometers', 'brand', 'transmission', 'drive', 'fuel', 'body_style', 'seats', 'engine']

        # Create a dictionary with default values for all columns
        default_data = {col: None for col in column_names}

        # Update default_data with received data
        default_data.update(data)

        # Example: Adjusting input data to match the expected 10 features
        new_data = np.array(list(default_data.values()))[:10].reshape(1, -1)

        output = model.predict(new_data)
        st.write(f"Predicted output: {output[0]}")

if __name__ == "__main__":
    main()
