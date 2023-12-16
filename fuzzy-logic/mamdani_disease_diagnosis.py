import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt


# Input
input_age = int(input("Age: "))
input_blood_pressure = int(input("Blood Pressure: "))
input_cholesterol = int(input("Cholesterol: "))
input_blood_sugar = int(input("Blood Sugar: "))
input_ldl = int(input("LDH: "))
input_hdl = int(input("HDH: "))


x_age = np.arange(0, 101, 1)
x_blood_pressure = np.arange(0, 221, 1)
x_cholesterol = np.arange(100, 251, 1)
x_blood_sugar = np.arange(0, 121, 1)
x_hdl = np.arange(0, 71, 1)
x_ldl = np.arange(0, 191, 1)
y_risk = np.arange(0, 46, 1)

age_young = mf.trapmf(x_age, [0, 0, 30, 40])
age_mid = mf.trapmf(x_age, [30, 40, 50, 60]) 
age_old = mf.trapmf(x_age, [50, 60, 100, 100])

blood_pressure_low = mf.trapmf(x_blood_pressure, [0, 0, 100, 120])
blood_pressure_mid = mf.trapmf(x_blood_pressure, [100, 120, 140, 160])
blood_pressure_high = mf.trapmf(x_blood_pressure, [140, 160, 180, 200])
blood_pressure_very_high = mf.trapmf(x_blood_pressure, [180, 200, 220, 220])

cholesterol_low = mf.trapmf(x_cholesterol, [0, 0, 180, 200])
cholesterol_mid = mf.trapmf(x_cholesterol, [180, 200, 220, 240])
cholesterol_high = mf.trapmf(x_cholesterol, [220, 240, 250, 270])

blood_sugar_very_high = mf.trimf(x_blood_sugar, [90, 120, 130])

ldl_normal= mf.trimf(x_ldl, [0, 0, 100,])
ldl_limit= mf.trimf(x_ldl, [100, 130, 160,])
ldl_high= mf.trimf(x_ldl, [130, 160, 190,])
ldl_very_high= mf.trapmf(x_ldl, [160, 190, 200, 200])

hdl_low= mf.trapmf(x_hdl, [0, 0, 30, 40])
hdl_mid= mf.trapmf(x_hdl, [30, 40, 50, 60])
hdl_high= mf.trapmf(x_hdl, [50, 60, 80, 80])

risk_not = mf.trapmf(y_risk, [0 ,0 ,5 ,10])
risk_little = mf.trapmf(y_risk, [5 ,10 ,15 ,20])
risk_mid = mf.trapmf(y_risk, [15 ,20 ,25 ,30])
risk_high = mf.trapmf(y_risk, [25 ,30 ,35 ,40])
risk_very_high = mf.trapmf(y_risk, [35, 40, 45, 50])

age_fit_young = fuzz.interp_membership(x_age, age_young, input_age)
age_fit_mid = fuzz.interp_membership(x_age, age_mid, input_age)
age_fit_old = fuzz.interp_membership(x_age, age_old, input_age)

blood_pressure_fit_low = fuzz.interp_membership(x_blood_pressure, blood_pressure_low, input_blood_pressure)
blood_pressure_fit_mid = fuzz.interp_membership(x_blood_pressure, blood_pressure_mid, input_blood_pressure)
blood_pressure_fit_high = fuzz.interp_membership(x_blood_pressure, blood_pressure_high , input_blood_pressure)
blood_pressure_fit_very_high = fuzz.interp_membership(x_blood_pressure, blood_pressure_very_high, input_blood_pressure)

cholesterol_fit_low = fuzz.interp_membership(x_cholesterol, cholesterol_low, input_cholesterol)
cholesterol_fit_mid = fuzz.interp_membership(x_cholesterol, cholesterol_mid, input_cholesterol)
cholesterol_fit_high = fuzz.interp_membership(x_cholesterol, cholesterol_high, input_cholesterol)

blood_sugar_fit_very_high = fuzz.interp_membership(x_blood_sugar, blood_sugar_very_high, input_blood_sugar)

ldl_fit_normal = fuzz.interp_membership(x_ldl, ldl_normal, input_ldl)
ldl_fit_limit = fuzz.interp_membership(x_ldl, ldl_limit, input_ldl)
ldl_fit_high = fuzz.interp_membership(x_ldl,ldl_high , input_ldl)
ldl_fit_very_high = fuzz.interp_membership(x_ldl, ldl_very_high, input_ldl)

hdl_fit_low = fuzz.interp_membership(x_hdl, hdl_low, input_hdl)
hdl_fit_mid = fuzz.interp_membership(x_hdl, hdl_mid, input_hdl)
hdl_fit_high = fuzz.interp_membership(x_hdl, hdl_high, input_hdl)

rule1 = np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fit_low ,cholesterol_fit_low),ldl_fit_normal), hdl_fit_high), risk_not)
rule2 = np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fit_low ,cholesterol_fit_low),ldl_fit_limit), hdl_fit_high), risk_little) 
rule3 = np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fit_low ,cholesterol_fit_low),ldl_fit_high), hdl_fit_high), risk_mid) 
rule4 = np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fit_low ,cholesterol_fit_low),ldl_fit_very_high), hdl_fit_high), risk_high) 
rule5 = np.fmin(np.fmin(np.fmin(blood_pressure_fit_mid ,cholesterol_fit_low), hdl_fit_high), risk_not) 

rule6 = np.fmin(np.fmin(np.fmin(age_fit_young, blood_pressure_fit_mid), cholesterol_fit_mid), risk_not)
rule7 = np.fmin(np.fmin(np.fmin(age_fit_mid, blood_pressure_fit_mid), cholesterol_fit_mid), risk_not) 
rule8 = np.fmin(np.fmin(np.fmin(age_fit_old, blood_pressure_fit_mid), cholesterol_fit_mid), risk_not) 
rule9 = np.fmin(np.fmin(np.fmin(age_fit_young, blood_pressure_fit_high), cholesterol_fit_high), risk_mid) 
rule10 = np.fmin(np.fmin(np.fmin(age_fit_mid, blood_pressure_fit_high), cholesterol_fit_high), risk_high) 
rule11 = np.fmin(np.fmin(np.fmin(age_fit_old, blood_pressure_fit_high), cholesterol_fit_high), risk_very_high) 

rule12 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fit_young, blood_pressure_fit_mid), cholesterol_fit_low), ldl_fit_normal), hdl_fit_low), risk_not) 
rule13 = np.fmin(np.fmin(age_fit_young, blood_sugar_fit_very_high), risk_little) 
rule14 = np.fmin(np.fmin(age_fit_mid, blood_sugar_fit_very_high), risk_high) 
rule15 = np.fmin(np.fmin(age_fit_old, blood_sugar_fit_very_high), risk_very_high) 
rule16 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fit_young, blood_pressure_fit_low), cholesterol_fit_low), blood_sugar_fit_very_high), ldl_fit_normal), hdl_fit_high), risk_little) 
rule17 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fit_mid, blood_pressure_fit_low), cholesterol_fit_low), blood_sugar_fit_very_high), ldl_fit_normal), hdl_fit_high), risk_high) 
rule18 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fit_old, blood_pressure_fit_low), cholesterol_fit_low), blood_sugar_fit_very_high), ldl_fit_normal), hdl_fit_high), risk_very_high) 
rule19 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fit_mid, blood_pressure_fit_low), cholesterol_fit_low), blood_sugar_fit_very_high), ldl_fit_very_high), hdl_fit_high), risk_very_high) 

rule20 = np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fit_very_high, cholesterol_fit_high), ldl_fit_very_high), hdl_fit_high), risk_very_high) 
rule21 = np.fmin(np.fmin(np.fmin(np.fmin(blood_pressure_fit_high, cholesterol_fit_high), ldl_fit_high), hdl_fit_mid), risk_very_high) 
rule22 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(age_fit_young, blood_pressure_fit_very_high), cholesterol_fit_high), ldl_fit_very_high), hdl_fit_mid), risk_mid) 
rule23 = np.fmin(np.fmin(age_fit_mid, blood_pressure_fit_very_high), risk_very_high)  
rule24 = np.fmin(np.fmin(age_fit_old, blood_pressure_fit_very_high), risk_very_high) 

out_not = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(rule1,rule5),rule6),rule7),rule8),rule12)
out_little = np.fmax(np.fmax(rule2,rule13), rule16)
out_mid = np.fmax(np.fmax(rule3, rule9), rule22)
out_high = np.fmax(np.fmax(np.fmax(rule4, rule10),rule14),rule17)
out_very_high = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(rule11,rule15),rule18),rule19),rule20),rule21),rule23),rule24)

out_risk = np.fmax(np.fmax(np.fmax(np.fmax(out_not, out_little), out_mid), out_high), out_very_high)

defuzzified  = fuzz.defuzz(y_risk, out_risk, 'centroid')

result = fuzz.interp_membership(y_risk, out_risk, defuzzified)

print("\nCoroner Heart Diagnosis:", defuzzified)


def diagnosed_as(output):
    if np.sum(output):
        return fuzz.defuzz(y_risk, output, 'centroid')
    else:
        return 0

if defuzzified >= 0 and defuzzified < 5:
    print("Diagnosed as Not Risk")

if defuzzified >= 5 and defuzzified < 10 and diagnosed_as(out_not) > diagnosed_as(out_little):
    print("Diagnosed as Not Risk")

if defuzzified >= 5 and defuzzified < 10 and diagnosed_as(out_not) < diagnosed_as(out_little):
    print("Diagnosed as Little Risk")

if defuzzified >= 10 and defuzzified < 15:
    print("Diagnosed as Little Risk")

if defuzzified >= 15 and defuzzified < 20 and diagnosed_as(out_little) > diagnosed_as(out_mid):
    print("Diagnosed as Little Risk")
    
if defuzzified >= 15 and defuzzified < 20 and diagnosed_as(out_little) < diagnosed_as(out_mid):
    print("Diagnosed as Middle Risk")

if defuzzified >= 20 and defuzzified < 25:
    print("Diagnosed as Middle Risk")

if defuzzified >= 25 and defuzzified < 30 and diagnosed_as(out_mid) > diagnosed_as(out_high):
    print("Diagnosed as Middle Risk")

if defuzzified >= 25 and defuzzified < 30 and diagnosed_as(out_mid) < diagnosed_as(out_high):
    print("Diagnosed as High Risk")

if defuzzified >= 30 and defuzzified < 35:
    print("Diagnosed as High Risk")

if defuzzified >= 35 and defuzzified < 40 and diagnosed_as(out_high) > diagnosed_as(out_very_high):
    print("Diagnosed as High Risk")

if defuzzified >= 40 and defuzzified < 50:
    print("Diagnosed as Very High Risk")

if defuzzified >= 35 and defuzzified < 40 and diagnosed_as(out_high) < diagnosed_as(out_very_high):
    print("Diagnosed as Very High Risk")