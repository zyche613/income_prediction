from CONST import MODEL_PATH
from preproc_util import preprocess_dataset

import pickle
from flask import Flask, render_template, flash, request
from wtforms import *

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SeCrEt'


loaded_lg_model = pickle.load(open(MODEL_PATH, 'rb'))

class Questionnaire(Form):
    age = StringField('Age:', [validators.InputRequired()])
    workclass = SelectField('Work Class:', \
        choices={'--', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', \
            'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'})
    weight = StringField('Weight:',[validators.InputRequired()])
    education = SelectField('Education:', \
        choices={'--', 'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',\
            'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', \
            '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'})
    education_num = StringField('Years of Education:', [validators.InputRequired()])
    marital_status = SelectField('Marital Status:', \
        choices={'--', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', \
            'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'})
    occupation = SelectField('Occupation:', \
        choices={'--', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', \
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', \
            'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', \
            'Protective-serv', 'Armed-Forces'})
    relationship = SelectField('Relationship:', \
        choices={'--', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'})
    race = SelectField('Race:', \
        choices={'--', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'})
    sex = SelectField('Sex:', \
        choices={'--', 'Female', 'Male'})
    capital_gain = StringField('Capital Gain:', [validators.InputRequired()])
    capital_loss = StringField('Capital Loss:', [validators.InputRequired()])
    hours_per_week = StringField('Hours Per Week:', [validators.InputRequired()]) 
    native_country = SelectField('Native Country:', \
        choices={'--', 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', \
            'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', \
            'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', \
            'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', \
            'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', \
            'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', \
            'Hong', 'Holand-Netherlands'})

@app.route("/", methods=['GET', 'POST'])
def form_handling():
    form = Questionnaire(request.form)
    print_out = ''
    if request.method == 'POST':
        age = form.age.data
        workclass = form.workclass.data
        weight = form.weight.data
        education = form.education.data
        education_num = form.education_num.data
        marital_status = form.marital_status.data
        occupation = form.occupation.data
        relationship = form.relationship.data
        race = form.race.data
        sex = form.sex.data
        capital_gain = form.capital_gain.data
        capital_loss = form.capital_loss.data
        hours_per_week = form.hours_per_week.data
        native_country = form.native_country.data
        feature = [age, workclass, weight, education, education_num, marital_status, \
            occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, \
            native_country]
        
        x = preprocess_dataset(feature)
        # res = loaded_lg_model.predict([x])[0]
        # print(res)

        # if res == 1:
        #     print_out = '<50K'
        # else:
        #     print_out = '>=50k'
        print_out = "{:.2%} chance of living under poverty!".format(loaded_lg_model.predict_proba([x])[0][1])
        print('prob:', print_out)
    return render_template('index.html', form=form, prob=print_out)

if __name__ == "__main__":
    app.run()

