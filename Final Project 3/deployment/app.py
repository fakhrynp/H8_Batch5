from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
scaler = StandardScaler()
rf = joblib.load("base_randomforest_model.pkl")  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        usia = float(request.form['usia'])
        anaemia = int(request.form['anaemia'])
        creatinin_fosfokinase = int(request.form['creatinin_fosfokinase'])
        diabetes = int(request.form['diabetes'])
        fraksi_ejeksi = int(request.form['fraksi_ejeksi'])
        tekanan_darah_tinggi = int(request.form['tekanan_darah_tinggi'])
        platelets = float(request.form['platelets'])
        kreatinin_serum = float(request.form['kreatinin_serum'])
        sodium_serum = int(request.form['sodium_serum'])
        jenis_kelamin = int(request.form['jenis_kelamin'])
        perokok = int(request.form['perokok'])
        time = int(request.form['time'])

        x_input = [
            [usia, anaemia, creatinin_fosfokinase, diabetes, fraksi_ejeksi, tekanan_darah_tinggi, platelets,
             kreatinin_serum, sodium_serum, jenis_kelamin, perokok, time]
        ]

        x_input = scaler.transform(x_input)
        y_output = rf.predict(x_input)

        result = "Tidak Meninggal" if y_output == 0 else "Meninggal"
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
