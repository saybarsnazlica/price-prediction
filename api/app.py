import json

import numpy as np
import xgboost as xgb
from flask import Flask
from flask_restful import Api, Resource, reqparse
from scipy.sparse import csr_matrix

app = Flask(__name__)
api = Api(app)

model = xgb.XGBRegressor()
model.load_model("models/model.json")

parser = reqparse.RequestParser()
parser.add_argument("query", location="form")


class PredictPrice(Resource):
    def get(self):
        return "Predict Price"
    
    def post(self):
        # use parser and find the user's query
        args = parser.parse_args()
        X = csr_matrix(json.loads(args["query"]))
        prediction = np.expm1(model.predict(X))

        # create JSON object
        output = {"price": int(prediction[0])}
        output = json.dumps(output)  
        
        return output, 200


# Route the URL to the resource
api.add_resource(PredictPrice, "/predict")


if __name__ == "__main__":
    app.run(debug=True)
