from flask import Flask, render_template, request
import pickle
import numpy as np

# import sklearn

# Load the Logistic Regression model
classifier = pickle.load(open('decisiontree.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Pclass= float(request.form['Pclass'])
        Age= float(request.form['Age'])
        SibSp= float(request.form['SibSp'])
        Parch= float(request.form['Parch'])
        Fare= float(request.form['Fare'])
        Sex_New= float(request.form['Sex_New'])

        # int_features = [int(x) for x in request.form.values()]

        # final_features = [np.array(int_features)]

        data = np.array([[Pclass, Age, SibSp,Parch,Fare,Sex_New]])
        my_prediction = classifier.predict(data)

        if my_prediction == 1:
            result = "Great! Survived."
        if my_prediction == 0:
            result = "Oops! Died."

        return render_template('result.html', Prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)