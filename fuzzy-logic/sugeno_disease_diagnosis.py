import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt


# custom function
def sugeno_polynomial_f(x, label):    
    if label == 'not_risk' and x != 0:
        b = 1.1
        a = 0.1
        # y = bx - a 
        return b * x - a

    elif label == 'little_risk' and x != 0:
        b = 1.2
        a = 0.12
        # y = bx - a 
        return b * x - a

    elif label == 'middle_risk' and x != 0:
        b = 1.3
        a = 0.13
        # y = bx - a 
        return b * x - 0.13
        
    elif label == 'high_risk' and x != 0:
        b = 1.4
        a = 0.14
        # y = bx - a 
        return b * x - a
            
    elif label == 'very_high_risk' and x != 0:
        b = 1.4
        a = 0.14
        # y = bx - a
        return b * x - a
    
    elif x == 0:
        return 0

# define vectorized sigmoid
sugeno_f = np.vectorize(sugeno_polynomial_f, otypes=[float])


# Input
input_age = int(input("Age: "))
input_blood_pressure = int(input("Blood Pressure: "))
input_cholesterol = int(input("Cholesterol: "))
input_blood_sugar = int(input("Blood Sugar: "))
input_ldl = int(input("LDH: "))
input_hdl = int(input("HDH: "))

# Define range
x_age = np.arange(0, 101, 1)
x_blood_pressure = np.arange(0, 221, 1)
x_cholesterol = np.arange(100, 251, 1)
x_blood_sugar = np.arange(0, 121, 1)
x_hdl = np.arange(0, 71, 1)
x_ldl = np.arange(0, 191, 1)
y_risk = np.arange(0, 46, 1)

# Range mapping
age = {
    'young': mf.trapmf(x_age, [0, 0, 30, 40]),
    'middle': mf.trapmf(x_age, [30, 40, 50, 60]) ,
    'old': mf.trapmf(x_age, [50, 60, 100, 100])
}

blood_pressure = {
    'low': mf.trapmf(x_blood_pressure, [0, 0, 100, 120]),
    'middle': mf.trapmf(x_blood_pressure, [100, 120, 140, 160]),
    'high': mf.trapmf(x_blood_pressure, [140, 160, 180, 200]),
    'very_high': mf.trapmf(x_blood_pressure, [180, 200, 220, 220])
}

cholesterol = {
    'low': mf.trapmf(x_cholesterol, [0, 0, 180, 200]),
    'middle': mf.trapmf(x_cholesterol, [180, 200, 220, 240]),
    'high': mf.trapmf(x_cholesterol, [220, 240, 250, 270])
}

blood_sugar = {
    'very_high': mf.trimf(x_blood_sugar, [90, 120, 130])
}

ldl = {
    'normal': mf.trimf(x_ldl, [0, 0, 100,]),
    'limit': mf.trimf(x_ldl, [100, 130, 160,]),
    'high': mf.trimf(x_ldl, [130, 160, 190,]),
    'very_high': mf.trapmf(x_ldl, [160, 190, 200, 200])
}

hdl = {
    'low': mf.trapmf(x_hdl, [0, 0, 30, 40]),
    'middle': mf.trapmf(x_hdl, [30, 40, 50, 60]),
    'high': mf.trapmf(x_hdl, [50, 60, 80, 80])
}

risk = {
    'not': mf.trapmf(y_risk, [0 ,0 ,5 ,10]),
    'little': mf.trapmf(y_risk, [5 ,10 ,15 ,20]),
    'middle': mf.trapmf(y_risk, [15 ,20 ,25 ,30]),
    'high': mf.trapmf(y_risk, [25 ,30 ,35 ,40]),
    'very_high': mf.trapmf(y_risk, [35, 40, 45, 50])
}

# Fuzzification
age_fuzz = {
    'young': fuzz.interp_membership(x_age, age['young'], input_age),
    'middle': fuzz.interp_membership(x_age, age['middle'], input_age),
    'old': fuzz.interp_membership(x_age, age['old'], input_age)
}

blood_pressure_fuzz = {
    'low': fuzz.interp_membership(x_blood_pressure, blood_pressure['low'], input_blood_pressure),
    'middle': fuzz.interp_membership(x_blood_pressure, blood_pressure['middle'], input_blood_pressure),
    'high': fuzz.interp_membership(x_blood_pressure, blood_pressure['high'] , input_blood_pressure),
    'very_high': fuzz.interp_membership(x_blood_pressure, blood_pressure['very_high'], input_blood_pressure)
}

cholesterol_fuzz = {
    'low': fuzz.interp_membership(x_cholesterol, cholesterol['low'], input_cholesterol),
    'middle': fuzz.interp_membership(x_cholesterol, cholesterol['middle'], input_cholesterol),
    'high': fuzz.interp_membership(x_cholesterol, cholesterol['high'], input_cholesterol)
}


blood_sugar_fuzz = {
    'very_high': fuzz.interp_membership(x_blood_sugar, blood_sugar['very_high'], input_blood_sugar)
}

ldl_fuzz = {
    'normal': fuzz.interp_membership(x_ldl, ldl['normal'], input_ldl),
    'limit': fuzz.interp_membership(x_ldl, ldl['limit'], input_ldl),
    'high': fuzz.interp_membership(x_ldl,ldl['high'] , input_ldl),
    'very_high': fuzz.interp_membership(x_ldl, ldl['very_high'], input_ldl)
}

hdl_fuzz = {
    'low': fuzz.interp_membership(x_hdl, hdl['low'], input_hdl),
    'middle': fuzz.interp_membership(x_hdl, hdl['middle'], input_hdl),
    'high': fuzz.interp_membership(x_hdl, hdl['high'], input_hdl)
}

# Inference
infer_rules = {
    1: np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['low'] ,cholesterol_fuzz['low']), ldl_fuzz['normal']), hdl_fuzz['high']), risk['not']),
    2: np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['low'] ,cholesterol_fuzz['low']), ldl_fuzz['limit']), hdl_fuzz['high']), risk['little']),
    3: np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['low'] ,cholesterol_fuzz['low']), ldl_fuzz['high']), hdl_fuzz['high']), risk['middle']),
    4: np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['low'] ,cholesterol_fuzz['low']), ldl_fuzz['very_high']), hdl_fuzz['high']), risk['high']),
    5: np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['middle'] ,cholesterol_fuzz['low']), hdl_fuzz['high']), risk['not']),
    6: np.fmin(np.fmin(np.fmin(age_fuzz['young'], blood_pressure_fuzz['middle']), cholesterol_fuzz['middle']), risk['not']),
    7: np.fmin(np.fmin(np.fmin(age_fuzz['middle'], blood_pressure_fuzz['middle']), cholesterol_fuzz['middle']), risk['not']),
    8: np.fmin(np.fmin(np.fmin(age_fuzz['old'], blood_pressure_fuzz['middle']), cholesterol_fuzz['middle']), risk['not']),
    9: np.fmin(np.fmin(np.fmin(age_fuzz['young'], blood_pressure_fuzz['high']), cholesterol_fuzz['high']), risk['middle']),
    10: np.fmin(np.fmin(np.fmin(age_fuzz['middle'], blood_pressure_fuzz['high']), cholesterol_fuzz['high']), risk['high']),
    11: np.fmin(np.fmin(np.fmin(age_fuzz['old'], blood_pressure_fuzz['high']), cholesterol_fuzz['high']), risk['very_high']),
    12: np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fuzz['young'], blood_pressure_fuzz['middle']), cholesterol_fuzz['low']), ldl_fuzz['normal']), hdl_fuzz['low']), risk['not']),
    13: np.fmin(np.fmin(age_fuzz['young'], blood_sugar_fuzz['very_high']), risk['little']),
    14: np.fmin(np.fmin(age_fuzz['middle'], blood_sugar_fuzz['very_high']), risk['high']),
    15: np.fmin(np.fmin(age_fuzz['old'], blood_sugar_fuzz['very_high']), risk['very_high']),
    16: np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fuzz['young'], blood_pressure_fuzz['low']), cholesterol_fuzz['low']), blood_sugar_fuzz['very_high']), ldl_fuzz['normal']), hdl_fuzz['high']), risk['little']),
    17: np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fuzz['middle'], blood_pressure_fuzz['low']), cholesterol_fuzz['low']), blood_sugar_fuzz['very_high']), ldl_fuzz['normal']), hdl_fuzz['high']), risk['high']),
    18: np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fuzz['old'], blood_pressure_fuzz['low']), cholesterol_fuzz['low']), blood_sugar_fuzz['very_high']), ldl_fuzz['normal']), hdl_fuzz['high']), risk['very_high']),
    19: np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fuzz['middle'], blood_pressure_fuzz['low']), cholesterol_fuzz['low']), blood_sugar_fuzz['very_high']), ldl_fuzz['very_high']), hdl_fuzz['high']), risk['very_high']),
    20: np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['very_high'], cholesterol_fuzz['high']), ldl_fuzz['very_high']), hdl_fuzz['high']), risk['very_high']),
    21: np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fuzz['high'], cholesterol_fuzz['high']), ldl_fuzz['high']), hdl_fuzz['middle']), risk['very_high']),
    22: np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fuzz['young'], blood_pressure_fuzz['very_high']), cholesterol_fuzz['high']), ldl_fuzz['very_high']), hdl_fuzz['middle']), risk['middle']),
    23: np.fmin(np.fmin(age_fuzz['middle'], blood_pressure_fuzz['very_high']), risk['very_high']),
    24: np.fmin(np.fmin(age_fuzz['old'], blood_pressure_fuzz['very_high']), risk['very_high'])
}

infer_output = {
    'not': sugeno_f(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(infer_rules[1], infer_rules[5]), infer_rules[6]), infer_rules[7]), infer_rules[8]), infer_rules[12]), 'not_risk'),
    'little': sugeno_f(np.fmax(np.fmax(infer_rules[2], infer_rules[13]), infer_rules[16]), 'little_risk'),
    'middle': sugeno_f(np.fmax(np.fmax(infer_rules[3], infer_rules[9]), infer_rules[22]), 'middle_risk'),
    'high': sugeno_f(np.fmax(np.fmax(np.fmax(infer_rules[4], infer_rules[10]), infer_rules[14]), infer_rules[17]), 'high_risk'),
    'very_high': sugeno_f(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(infer_rules[11], infer_rules[15]), infer_rules[18]), infer_rules[19]), infer_rules[20]), infer_rules[21]),infer_rules[23]), infer_rules[24]), 'very_high_risk')    
}

out_risk = np.fmax(np.fmax(np.fmax(np.fmax(infer_output['not'], infer_output['little']), infer_output['middle']), infer_output['high']), infer_output['very_high'])

# Defuzzification
defuzzified  = fuzz.defuzz(y_risk, out_risk, 'mom')

result = fuzz.interp_membership(y_risk, out_risk, defuzzified)

print("\nCoroner Heart Diagnosis:", defuzzified)


def diagnosed_as(output):
    if np.sum(output):
        return fuzz.defuzz(y_risk, output, 'mom')
    else:
        return 0

if defuzzified >= 0 and defuzzified < 5:
    print("Diagnosed as Not Risk")

if defuzzified >= 5 and defuzzified < 10 and diagnosed_as(infer_output['not']) > diagnosed_as(infer_output['little']):
    print("Diagnosed as Not Risk")

if defuzzified >= 5 and defuzzified < 10 and diagnosed_as(infer_output['not']) < diagnosed_as(infer_output['little']):
    print("Diagnosed as Little Risk")

if defuzzified >= 10 and defuzzified < 15:
    print("Diagnosed as Little Risk")

if defuzzified >= 15 and defuzzified < 20 and diagnosed_as(infer_output['little']) > diagnosed_as(infer_output['middle']):
    print("Diagnosed as Little Risk")
    
if defuzzified >= 15 and defuzzified < 20 and diagnosed_as(infer_output['little']) < diagnosed_as(infer_output['middle']):
    print("Diagnosed as Middle Risk")

if defuzzified >= 20 and defuzzified < 25:
    print("Diagnosed as Middle Risk")

if defuzzified >= 25 and defuzzified < 30 and diagnosed_as(infer_output['middle']) > diagnosed_as(infer_output['high']):
    print("Diagnosed as Middle Risk")

if defuzzified >= 25 and defuzzified < 30 and diagnosed_as(infer_output['middle']) < diagnosed_as(infer_output['high']):
    print("Diagnosed as High Risk")

if defuzzified >= 30 and defuzzified < 35:
    print("Diagnosed as High Risk")

if defuzzified >= 35 and defuzzified < 40 and diagnosed_as(infer_output['high']) > diagnosed_as(infer_output['very_high']):
    print("Diagnosed as High Risk")

if defuzzified >= 40 and defuzzified < 50:
    print("Diagnosed as Very High Risk")

if defuzzified >= 35 and defuzzified < 40 and diagnosed_as(infer_output['high']) < diagnosed_as(infer_output['very_high']):
    print("Diagnosed as Very High Risk")