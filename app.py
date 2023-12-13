from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


## Unpacking the models
breast_model = pickle.load(open('breast_model.pkl', 'rb'))
lung_model = pickle.load(open('lung_model.pkl', 'rb'))
cervical_model = pickle.load(open('cervical_model.pkl', 'rb'))
prostate_model = pickle.load(open('prostate_model.pkl', 'rb'))
cervical_scaler = pickle.load(open('cervical_scaler.pkl', 'rb'))
breast_scaler = pickle.load(open('breast_scaler.pkl', 'rb'))
prostate_scaler = pickle.load(open('prostate_scaler.pkl', 'rb'))
lung_scaler = pickle.load(open('lung_scaler.pkl', 'rb'))

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/lung_cancer_predictor', methods=['POST', 'GET'])
def lung():
    if request.method == 'GET':
        return render_template('lung.html', **locals())
    
    if request.method == 'POST':
        try:
            predictor_values = []
            pressure = request.form.get('pressure')
            allergies = request.form.get('allergy')
            alcohol = request.form.get('alcohol')
            coughing = request.form.get('coughing')
            swallowing = request.form.get('swallowing')
            
            predictor_values.extend([pressure, allergies, alcohol, coughing, swallowing])
            predictor_values = [float(i) for i in predictor_values]
            predictor_values = np.array(predictor_values).reshape(1, -1)

            predictor_values = lung_scaler.fit_transform(predictor_values)
            result = lung_model.predict(predictor_values)
            
            if result ==  1:
                prediction = 'You have a high risk of lung cancer'
            if result == 0:
                prediction = 'You have a low risk of lung cancer'
            accuracy = 0.88 * 100
        except ValueError:
            prediction = 'Invalid input. Please enter valid numbers.'
        except Exception as e:
            prediction = 'An error occurred: ' + str(e)
            
    return render_template('lung.html', **locals())

@app.route('/breast_cancer_predictor', methods=['POST', 'GET'])
def breast():
    if request.method == 'POST':
        try:
            predictor_values = []
            texture_mean = float(request.form.get('texture_mean'))
            concave_points_mean = float(request.form.get('concave_points_mean'))
            perimeter_se = float(request.form.get('perimeter_se'))
            area_se = float(request.form.get('area_se'))
            texture_worst = float(request.form.get('texture_worst'))
            perimeter_worst = float(request.form.get('perimeter_worst'))
            area_worst = float(request.form.get('area_worst'))
            smoothness_worst = float(request.form.get('smoothness_worst'))
            concavity_worst = float(request.form.get('concavity_worst'))
            concave_points_worst = float(request.form.get('concave_points_worst'))
            symmetry_worst = float(request.form.get('symmetry_worst'))
            
            predictor_values.extend([texture_mean, concave_points_mean, perimeter_se, area_se,texture_worst, perimeter_worst, area_worst, smoothness_worst,concavity_worst, concave_points_worst, symmetry_worst])
            predictor_values = np.array(predictor_values).reshape(1, -1)

            predictor_values = breast_scaler.transform(predictor_values)
            result = breast_model.predict(predictor_values)
            if result ==  1:
                prediction = 'You have a high risk of breast cancer'
            if result == 0:
                prediction = 'You have a low risk of breast cancer'
            accuracy = 0.979 * 100
        except ValueError:
            prediction = 'Invalid input. Please enter valid numbers.'
        except Exception as e:
            prediction = 'An error occurred: ' + str(e)
    return render_template('breast.html', **locals())


@app.route('/prostate_cancer_predictor', methods=['GET', 'POST'])
def prostate():
    # Get data from form
    if request.method == 'POST':
        try:
            
            perimeter = float(request.form['perimeter'])
            area = float(request.form['area'])
            smoothness = float(request.form['smoothness'])
            compactness = float(request.form['compactness'])
            fractal_dimension = float(request.form['fractal_dimension'])

        # Create feature array and scale
            features = np.array([perimeter, area, smoothness, compactness, fractal_dimension]).reshape(1, -1)
            features = prostate_scaler.transform(features)

            # Make prediction
            result = prostate_model.predict(features)[0]
            if result ==  1:
                prediction = 'You have a high risk of prostate cancer'
            if result == 0:
                prediction = 'You have a low risk of prostate cancer'
            
            accuracy = 0.84 * 100
        except ValueError:
            prediction = 'Invalid input. Please enter valid numbers.'
        except Exception as e:
            prediction = 'An error occurred: ' + str(e)
    # Render template with prediction
    return render_template('prostate.html', **locals())

    
@app.route('/cervical_cancer_predictor', methods=['POST', 'GET'])
def cervical():
    # Get data from form
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            number_of_sexual_partners = float(request.form['number_of_sexual_partners'])
            first_sexual_intercourse = float(request.form['first_sexual_intercourse'])
            num_of_pregnancies = float(request.form['num_of_pregnancies'])
            smokes = float(request.form['smokes'])
            smokes_years = float(request.form['smokes_years'])
            smokes_packs_year = float(request.form['smokes_packs_year'])
            hormonal_contraceptives = float(request.form['hormonal_contraceptives'])
            hormonal_contraceptives_years = float(request.form['hormonal_contraceptives_years'])
            iud = float(request.form['iud'])
            iud_years = float(request.form['iud_years'])
            stds = float(request.form['stds'])
            stds_genital_herpes = float(request.form['stds_genital_herpes'])
            stds_molluscum_contagiosum = float(request.form['stds_molluscum_contagiosum'])
            stds_hepatitis_b = float(request.form['stds_hepatitis_b'])
            stds_number_of_diagnosis = float(request.form['stds_number_of_diagnosis'])
            dx_cin = float(request.form['dx_cin'])
            dx_hpv = float(request.form['dx_hpv'])
            dx = float(request.form['dx'])
            hinselmann = float(request.form['hinselmann'])
            schiller = float(request.form['schiller'])
            citology = float(request.form['citology'])
            biopsy = float(request.form['biopsy'])

            # Create feature array and scale
            features = np.array([age, number_of_sexual_partners, first_sexual_intercourse, num_of_pregnancies, smokes, smokes_years, smokes_packs_year, hormonal_contraceptives, hormonal_contraceptives_years, iud, iud_years, stds, stds_genital_herpes, stds_molluscum_contagiosum, stds_hepatitis_b, stds_number_of_diagnosis, dx_cin, dx_hpv, dx, hinselmann, schiller, citology, biopsy]).reshape(1, -1)
            features = cervical_scaler.transform(features)

            # Make prediction
            result = cervical_model.predict(features)[0]
            if result ==  1:
                prediction = 'You have a high risk of cervical cancer'
            if result == 0:
                prediction = 'You have a low risk of cervical cancer'
            accuracy = 0.994 * 100
        except ValueError:
            prediction = 'Invalid input. Please enter valid numbers.'
        except Exception as e:
            prediction = 'An error occurred: ' + str(e)
    # Render template with prediction
    return render_template('cervical.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)
