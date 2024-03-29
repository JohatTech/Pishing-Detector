import json
from flask import Flask, request, jsonify, json
from flask_cors import CORS
import os 
import pandas as pd 
import joblib as jb 
import traceback
import sys 
# with this we crete the app

app = Flask(__name__)
CORS(app)


# receive or fetch the attribute given by our user
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if model:
		try:
			if request.method == 'POST':
    				#get the json data 
					posted_data = request.json
					print(posted_data)
					#transform json to panda data frame
					query_df = pd.DataFrame(posted_data, index=[0])
					query = query_df.reindex(columns = model_columns, fill_value = 0)
					prediction = model.predict(query)
					return jsonify({'prediction':str(prediction)})
			
		except:
			return jsonify({'trace': traceback.format_exc()})
	else:
		print ('Train the model first')
		return ('No model here to use')
# this is for run the app yeah just for run
if __name__== "__main__":
	try:
		port = int(sys.argv[1]) # This is for a command-line input
	except:
		port = 5000 # If you don't provide any port the port will be set to 12345
		model = jb.load("Pishing-model.pkl") # Load "model.pkl"
		print ('Model loaded')
		model_columns = jb.load("models_columns.pkl") # Load "model_columns.pkl"
		print ('Model columns loaded')
		app.run(port=port, debug=True)


# [
#     {"SSLfinal_State":1,"URL_of_Anchor":1,"web_traffic":-1,"links_in_tags":1,"Domain_registration_length":-1, "having_Sub_Domain":1, "Prefix_Suffix":1}
 
# ]