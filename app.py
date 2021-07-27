from flask import Flask, render_template, request
import joblib
import numpy as np

import warnings
warnings.simplefilter("ignore")
      
app = Flask(__name__)
model = joblib.load(open('./model/model.pkl', 'rb'))
vector = joblib.load(open('./model/vector.pkl', 'rb'))

@app.route('/')
def index():
 return render_template('reviewform.html')

@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
      review = request.form.get('moviereview')
      vec=vector.transform([review]).toarray()
      pred=model.predict(vec)

      pred="".join(pred)
      
      return render_template('reviewform.html', content=review,prediction=pred)
      
if __name__ == '__main__':
 app.run(debug=False)
