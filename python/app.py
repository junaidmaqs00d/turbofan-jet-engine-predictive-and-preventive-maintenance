from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

loaded_model = load('random_forest_model.pkl')

def RUL_calculator(df, df_max_cycles):
    max_cycle = df_max_cycles["cycle"]
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='id', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["cycle"]
    result_frame.drop(['max_cycle'], axis=1, inplace=True)
    return result_frame

def preprocess_data(input_data):
    # Drop irrelevant columns
    columns_to_drop = ['cycle', 'op1', 'op2', 'op3', 'sensor1', 'sensor5', 'sensor6', 'sensor10',
                       'sensor16', 'sensor18', 'sensor19', 'sensor14', 'sensor13', 'sensor12',
                       'sensor11', 'sensor9', 'sensor22', 'sensor23']
    input_data.drop(columns_to_drop, axis=1, inplace=True)

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(input_data.drop(['id', 'RUL'], axis=1))
    scaled_features = pd.DataFrame(scaled_features, columns=input_data.drop(['id', 'RUL'], axis=1).columns)

    # Add the 'id' and 'RUL' columns back to the scaled features
    scaled_features['id'] = input_data['id']
    scaled_features['RUL'] = input_data['RUL']

    return scaled_features

def predict_remaining_useful_life(input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data)
    # Remove the 'RUL' feature before prediction
    preprocessed_data.drop('RUL', axis=1, inplace=True)

    # Use the trained model to predict labels
    predicted_labels = loaded_model.predict(preprocessed_data)
    
    
    # Create a DataFrame with engine ID and predicted class
    result_df = pd.DataFrame({'Engine_ID': input_data['id'],
                              'Predicted_Class': predicted_labels})

    return result_df.to_dict(orient='records')

def applyingmodel(input_data):
    jet_id_and_rul = input_data.groupby(['id'])[["id", "cycle"]].max()
    jet_id_and_rul.set_index('id', inplace=True)
    jet_data = RUL_calculator(input_data, jet_id_and_rul)

    # Predict the remaining useful life (RUL) using the processed input data
    predicted_RUL = predict_remaining_useful_life(jet_data)
    print(predicted_RUL)

    return predicted_RUL

@app.route('/predict', methods=['POST'])
@cross_origin()  # Allow cross-origin requests for this route
def predict():
    data = request.json  # Get JSON data from the request
    print("Received JSON data:", data)  # Log the received JSON data

    # Convert JSON data to DataFrame
    input_data = pd.DataFrame([data])
    print("DataFrame created from JSON:", input_data)  # Log the created DataFrame

    csv_file = 'testdata.csv'
    
    if os.path.exists(csv_file):
        input_data.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        input_data.to_csv(csv_file, mode='w', header=True, index=False)

    df = pd.read_csv(csv_file)
    
    # Apply the model to predict remaining useful life
    predicted_RUL = applyingmodel(df)
    print("Predicted RUL:", predicted_RUL)  # Log the predicted RUL data

    # Prepare the response data
    response_data = {
        'Engine_ID': [entry['Engine_ID'] for entry in predicted_RUL],
        'Predicted_Class': [entry['Predicted_Class'] for entry in predicted_RUL]
    }

    print("Response data:", response_data)  # Log the response data

    return jsonify(response_data) # Return the response as JSON

if __name__ == '__main__':
    app.run(debug=True)



























# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# import pandas as pd
# from joblib import load
# from sklearn.preprocessing import MinMaxScaler

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# loaded_model = load('random_forest_model.joblib')

# def RUL_calculator(df, df_max_cycles):
#     max_cycle = df_max_cycles["cycle"]
#     result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='id', right_index=True)
#     result_frame["RUL"] = result_frame["max_cycle"] - result_frame["cycle"]
#     result_frame.drop(['max_cycle'], axis=1, inplace=True)
#     return result_frame

# def preprocess_data(input_data):
#     # Drop irrelevant columns
#     columns_to_drop = ['cycle', 'op1', 'op2', 'op3', 'sensor1', 'sensor5', 'sensor6', 'sensor10',
#                        'sensor16', 'sensor18', 'sensor19', 'sensor14', 'sensor13', 'sensor12',
#                        'sensor11', 'sensor9', 'sensor22', 'sensor23']
#     input_data.drop(columns_to_drop, axis=1, inplace=True)

#     # Scale the features using MinMaxScaler
#     scaler = MinMaxScaler()
#     scaled_features = scaler.fit_transform(input_data.drop(['id', 'RUL'], axis=1))
#     scaled_features = pd.DataFrame(scaled_features, columns=input_data.drop(['id', 'RUL'], axis=1).columns)

#     # Add the 'id' and 'RUL' columns back to the scaled features
#     scaled_features['id'] = input_data['id']
#     scaled_features['RUL'] = input_data['RUL']

#     return scaled_features

# def predict_remaining_useful_life(input_data):
#     # Preprocess the input data
#     preprocessed_data = preprocess_data(input_data)

#     # Remove the 'RUL' feature before prediction
#     preprocessed_data.drop('RUL', axis=1, inplace=True)

#     # Use the trained model to predict labels
#     predicted_labels = loaded_model.predict(preprocessed_data)

#     # Create a DataFrame with engine ID and predicted class
#     result_df = pd.DataFrame({'Engine_ID': input_data['id'],
#                               'Predicted_Class': predicted_labels})

#     return result_df.to_dict(orient='records')

# def applyingmodel(input_data):
#     jet_id_and_rul = input_data.groupby(['id'])[["id", "cycle"]].max()
#     jet_id_and_rul.set_index('id', inplace=True)
#     jet_data = RUL_calculator(input_data, jet_id_and_rul)

#     # Predict the remaining useful life (RUL) using the processed input data
#     predicted_RUL = predict_remaining_useful_life(jet_data)

#     return predicted_RUL

# @app.route('/predict', methods=['POST'])
# @cross_origin()  # Allow cross-origin requests for this route
# def predict():
#     data = request.json  # Get JSON data from the request
#     print("Received JSON data:", data)  # Log the received JSON data

#     df = pd.DataFrame([data])  # Convert JSON data to DataFrame
#     print("DataFrame created from JSON:", df)  # Log the created DataFrame

#     # Apply the model to predict remaining useful life
#     predicted_RUL = applyingmodel(df)
#     print("Predicted RUL:", predicted_RUL)  # Log the predicted RUL data

#     # Prepare the response data
#     response_data = {
#         'Engine_ID': [entry['Engine_ID'] for entry in predicted_RUL],
#         'Predicted_Class': [entry['Predicted_Class'] for entry in predicted_RUL]
#     }

#     print("Response data:", response_data)  # Log the response data

#     return jsonify(response_data)  # Return the response as JSON

# if __name__ == '__main__':
#     app.run(debug=True)
