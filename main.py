import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import signal
import streamlit as st
st.write("hii")
df = pd.read_csv("priyadharshini_1.txt",skiprows=6,header=None)

df.columns=['index','channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8','acc1','acc2','acc3','time_std','timestamp']
df.drop(['index'],axis=1,inplace=True)

df['channel1'] = ss.detrend(df['channel1'])
df['channel2'] = ss.detrend(df['channel2'])
df['channel3'] = ss.detrend(df['channel3'])
df['channel4'] = ss.detrend(df['channel4'])
df['channel5'] = ss.detrend(df['channel5'])
df['channel6'] = ss.detrend(df['channel6'])
df['channel7'] = ss.detrend(df['channel7'])
df['channel8'] = ss.detrend(df['channel8'])

df['channel1'] = (df['channel1']-np.mean(df['channel1']))/np.std(df['channel1'])
df['channel2'] = (df['channel2']-np.mean(df['channel2']))/np.std(df['channel2'])
df['channel3'] = (df['channel3']-np.mean(df['channel3']))/np.std(df['channel3'])
df['channel4'] = (df['channel4']-np.mean(df['channel4']))/np.std(df['channel4'])
df['channel5'] = (df['channel5']-np.mean(df['channel5']))/np.std(df['channel5'])
df['channel6'] = (df['channel6']-np.mean(df['channel6']))/np.std(df['channel6'])
df['channel7'] = (df['channel7']-np.mean(df['channel7']))/np.std(df['channel7'])
df['channel8'] = (df['channel8']-np.mean(df['channel8']))/np.std(df['channel8'])

samp_freq = 250  # Sample frequency (Hz)
notch_freq = 50  # Frequency to be removed from signal (Hz)
quality_factor = 250  # Quality factor
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
freq, h = signal.freqz(b_notch, a_notch, fs = samp_freq)
# apply notch filter to signal
df['channel1'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel1']))
df['channel2'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel2']))
df['channel3'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel3']))
df['channel4'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel4']))
df['channel5'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel5']))
df['channel6'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel6']))
df['channel7'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel7']))
df['channel8'] =pd.DataFrame(signal.filtfilt(b_notch, a_notch, df['channel8']))

st.write(df)

channels=['FP1','FP2','C3','C4','T5','T6','O1','O2']

no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
beta_power1 = np.sum(psd[(f >= 13) & (f <= 22)]) # Compute the beta wave power

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
beta_power2 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
beta_power3 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
beta_power4 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
beta_power5 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
beta_power6 = np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
beta_power7 =np.sum(psd[(f >= 13) & (f <= 22)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
beta_power8 = np.sum(psd[(f >= 13) & (f <= 22)])   

beta_power=[beta_power1,beta_power2,beta_power3,beta_power4,beta_power5,beta_power6,beta_power7,beta_power8]

print(beta_power1,beta_power2,beta_power3,beta_power4,beta_power5,beta_power6,beta_power7,beta_power8)

import plotly.express as px

fig = px.bar(df, x=channels, y=beta_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
st.write(fig)

no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
alpha_power1 = np.sum(psd[(f >= 8) & (f <= 13)]) # Compute the alpha wave power

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
alpha_power2 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
alpha_power3 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
alpha_power4 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
alpha_power5 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
alpha_power6 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
alpha_power7 = np.sum(psd[(f >= 8) & (f <= 13)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
alpha_power8 = np.sum(psd[(f >= 8) & (f <= 13)])  

alpha_power=[alpha_power1,alpha_power2,alpha_power3,alpha_power4,alpha_power5,alpha_power6,alpha_power7,alpha_power8]

print(alpha_power1,alpha_power2,alpha_power3,alpha_power4,alpha_power5,alpha_power6,alpha_power7,alpha_power8)

import plotly.express as px

fig = px.bar(df, x=channels, y=alpha_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
st.write(fig)


no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
gamma_power1 = np.sum(psd[(f >= 30) & (f <= 100)])  # Compute the alpha wave power

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
gamma_power2 = np.sum(psd[(f >= 30 )& (f <= 100)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
gamma_power3 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
gamma_power4 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
gamma_power5 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
gamma_power6 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
gamma_power7 = np.sum(psd[(f >= 30) & (f <= 100)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
gamma_power8 = np.sum(psd[(f >= 30) & (f <= 100)])  

gamma_power=[gamma_power1,gamma_power2,gamma_power3,gamma_power4,gamma_power5,gamma_power6,gamma_power7,gamma_power8]


import plotly.express as px

fig = px.bar(df, x=channels, y=gamma_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()
st.write(fig)



no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
delta_power1 = np.sum(psd[(f >= 0.5) & (f <= 4)]) 

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
delta_power2 = np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
delta_power3 =np.sum(psd[(f >= 0.5) & (f <= 4)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
delta_power4 = np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
delta_power5 = np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
delta_power6 =np.sum(psd[(f >= 0.5) & (f <= 4)])

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
delta_power7 =np.sum(psd[(f >= 0.5) & (f <= 4)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
delta_power8 = np.sum(psd[(f >= 0.5) & (f <= 4)])   

delta_power=[delta_power1,delta_power2,delta_power3,delta_power4,delta_power5,delta_power6,delta_power7,delta_power8]

print(delta_power1,delta_power2,delta_power3,delta_power4,delta_power5,delta_power6,delta_power7,delta_power8)

no=len(df)
f, psd = ss.welch(df['channel1'], fs=250,nperseg=61302)
theta_power1 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel2'], fs=250,nperseg=61302)
theta_power2 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel3'], fs=250,nperseg=61302)
theta_power3 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel4'], fs=250,nperseg=61302)
theta_power4 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel5'], fs=250,nperseg=61302)
theta_power5 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel6'], fs=250,nperseg=61302)
theta_power6 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel7'], fs=250,nperseg=61302)
theta_power7 = np.sum(psd[(f >= 4) & (f <= 7)]) 

f, psd = ss.welch(df['channel8'], fs=250,nperseg=61302)
theta_power8 = np.sum(psd[(f >= 4) & (f <= 7)])   

theta_power=[theta_power1,theta_power2,theta_power3,theta_power4,theta_power5,theta_power6,theta_power7,theta_power8]


      
import plotly.express as px

fig = px.bar(df, x=channels, y=theta_power, color=channels,
              pattern_shape_sequence=[".", "x", "+"])
fig.show()


st.write(fig)

left_alpha=np.sum([alpha_power1,alpha_power3,alpha_power5,alpha_power7])
right_alpha=np.sum([alpha_power2,alpha_power4,alpha_power6,alpha_power8])

left_beta=np.sum([beta_power1,beta_power3,beta_power5,beta_power7])
right_beta=np.sum([beta_power2,beta_power4,beta_power6,beta_power8])

left_delta=np.sum([delta_power1,delta_power3,delta_power5,delta_power7])
right_delta=np.sum([delta_power2,delta_power4,delta_power6,delta_power8])

left_gamma=np.sum([gamma_power1,gamma_power3,gamma_power5,gamma_power7])
right_gamma=np.sum([gamma_power2,gamma_power4,gamma_power6,gamma_power8])

left_theta=np.sum([theta_power1,theta_power3,theta_power5,theta_power7])
right_theta=np.sum([theta_power2,theta_power4,theta_power6,theta_power8])


import plotly.graph_objects as px

x = ['left', 'right']
 
plot = px.Figure(data=[px.Bar(
    name = 'alpha',
    x = x,
    y = [left_alpha,right_alpha]
   ),
    px.Bar(
    name = 'beta',
    x = x,
    y = [left_beta,right_beta]
   ),
        px.Bar(
    name = 'delta',
    x = x,
    y = [left_delta,right_delta]
   ),                   
         px.Bar(
    name = 'gamma',
    x = x,
    y = [left_gamma,right_gamma]
   ),  
          px.Bar(
    name = 'gamma',
    x = x,
    y = [left_theta,right_theta]
   ),                       
])
                  
plot.show()

st.write(plot)


