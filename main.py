# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:36:09 2023

@author: PRAMILA
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import signal
import streamlit as st
st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt; font-weight: bold;'>Analysis of eeg signal</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:white;background-color:black;font-size:14pt'>📂 Upload your CSV or Excel file. (200MB max) 📂</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="",type=['txt'])
import tensorflow as tf
import pickle
global df
import plotly.express as px
from sklearn.preprocessing import StandardScaler


if(st.submit("PREDICT"):
   if uploaded_file is not None:
      print(uploaded_file)

      try:
         df = pd.read_csv(uploaded_file,skiprows=6,header=None)
         #st.write(df)
      except:
         #df = pd.read_csv(r"C:\Users\PRAMILA\Downloads\BCI_data-20230110T035009Z-001\BCI_data\normal\priyadharshini_1.txt",skiprows=6,header=None)
         st.write("file not found")

   #df = pd.read_csv("priyadharshini_1.txt",skiprows=6,header=None)
   #df = pd.read_csv(uploaded_file,skiprows=6,header=None)
   df.columns=['index','channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8','acc1','acc2','acc3','time_std','timestamp']
   df.drop(['index'],axis=1,inplace=True)
   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Raw data</h1>", unsafe_allow_html=True)
   st.write(df)
   fig = px.line(df,x=np.arange(len(df)),y=[df['channel1'],df['channel2'],df['channel3'],df['channel4'],df['channel5'],df['channel6'],df['channel7'],df['channel8']])
   fig.update_layout( yaxis_title='EEG Readings (microvolts)',xaxis_title='Timesteps (milli seconds)')
   st.write(fig)


   # import plotly.express as px
   # col1, col2 = st.columns(2)


   # with col1:
   #     #plt.figure(figsize=[6.4, 2.4]) 
   #     for column in df[['channel1', 'channel2','channel3', 'channel4','channel5', 'channel6','channel7', 'channel8']]:    
   #         plt.plot(df[column],label=column)
   #         plt.legend(loc='best')
   #     st.pyplot(plt)

   # with col2:
   #     fig = px.line(df,x=np.arange(len(df)),y=[df['channel1'],df['channel2'],df['channel3'],df['channel4'],df['channel5'],df['channel6'],df['channel7'],df['channel8']],width=100, height=400)
   #     fig.update_layout( yaxis_title='EEG Readings (microvolts)',xaxis_title='Timesteps (milli seconds)')
   #     st.write(fig)



   df['channel1'] = ss.detrend(df['channel1'])
   df['channel2'] = ss.detrend(df['channel2'])
   df['channel3'] = ss.detrend(df['channel3'])
   df['channel4'] = ss.detrend(df['channel4'])
   df['channel5'] = ss.detrend(df['channel5'])
   df['channel6'] = ss.detrend(df['channel6'])
   df['channel7'] = ss.detrend(df['channel7'])
   df['channel8'] = ss.detrend(df['channel8'])
   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Detrended data (removing drift)</h1>", unsafe_allow_html=True)
   st.write(df)
   fig = px.line(df,x=np.arange(len(df)),y=[df['channel1'],df['channel2'],df['channel3'],df['channel4'],df['channel5'],df['channel6'],df['channel7'],df['channel8']])
   fig.update_layout( yaxis_title='EEG Readings (microvolts)',xaxis_title='Timesteps (milli seconds)')
   st.write(fig)


   # for column in df[['channel1', 'channel2','channel3', 'channel4','channel5', 'channel6','channel7', 'channel8']]:    
   #     plt.plot(df[column],label=column)
   #     plt.legend(loc='best')
   # st.pyplot(plt)

   chan1 = (df['channel1']-np.mean(df['channel1']))/np.std(df['channel1'])
   chan2 = (df['channel2']-np.mean(df['channel2']))/np.std(df['channel2'])
   chan3 = (df['channel3']-np.mean(df['channel3']))/np.std(df['channel3'])
   chan4 = (df['channel4']-np.mean(df['channel4']))/np.std(df['channel4'])
   chan5 = (df['channel5']-np.mean(df['channel5']))/np.std(df['channel5'])
   chan6 = (df['channel6']-np.mean(df['channel6']))/np.std(df['channel6'])
   chan7 = (df['channel7']-np.mean(df['channel7']))/np.std(df['channel7'])
   chan8 = (df['channel8']-np.mean(df['channel8']))/np.std(df['channel8'])
   df['channel1'] = chan1
   df['channel2'] = chan2
   df['channel3'] = chan3
   df['channel4'] = chan4
   df['channel5'] = chan5
   df['channel6'] = chan6
   df['channel7'] = chan7
   df['channel8'] = chan8
   # df['channel1'] = (df['channel1']-np.mean(df['channel1']))/np.std(df['channel1'])
   # df['channel2'] = (df['channel2']-np.mean(df['channel2']))/np.std(df['channel2'])
   # df['channel3'] = (df['channel3']-np.mean(df['channel3']))/np.std(df['channel3'])
   # df['channel4'] = (df['channel4']-np.mean(df['channel4']))/np.std(df['channel4'])
   # df['channel5'] = (df['channel5']-np.mean(df['channel5']))/np.std(df['channel5'])
   # df['channel6'] = (df['channel6']-np.mean(df['channel6']))/np.std(df['channel6'])
   # df['channel7'] = (df['channel7']-np.mean(df['channel7']))/np.std(df['channel7'])
   # df['channel8'] = (df['channel8']-np.mean(df['channel8']))/np.std(df['channel8'])


   # plt.figure(figsize=[6.4, 2.4]) 
   # i=1
   # for column in [chan1, chan2,chan3, chan4,chan5, chan6,chan7, chan8]:    
   #     plt.plot(column,label="channel"+str(i))
   #     plt.legend(loc='best')
   #     i+=1

   # st.pyplot(plt)

   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Artifact/Deviations Removed data</h1>", unsafe_allow_html=True)
   fig = px.line(df,x=np.arange(len(df)),y=[df['channel1'],df['channel2'],df['channel3'],df['channel4'],df['channel5'],df['channel6'],df['channel7'],df['channel8']])
   fig.update_layout( yaxis_title='EEG Readings (microvolts)',xaxis_title='Timesteps (milli seconds)')
   st.write(fig)


   # b, a = ss.iirfilter(1, Wn=50, fs=250, btype="high", ftype="butter")
   # print(b, a, sep="\n")
   # df['channel1'] = ss.filtfilt(b, a, df['channel1'])
   # df['channel2'] = ss.filtfilt(b, a, df['channel2'])
   # df['channel3'] = ss.filtfilt(b, a, df['channel3'])
   # df['channel4'] = ss.filtfilt(b, a, df['channel4'])
   # df['channel5'] = ss.filtfilt(b, a, df['channel5'])
   # df['channel6'] = ss.filtfilt(b, a, df['channel6'])
   # df['channel7'] = ss.filtfilt(b, a, df['channel7'])
   # df['channel8'] = ss.filtfilt(b, a, df['channel8'])

   # chan1 = ss.filtfilt(b, a, chan1)
   # chan2 = ss.filtfilt(b, a, chan2)
   # chan3 = ss.filtfilt(b, a, chan3)
   # chan4 = ss.filtfilt(b, a, chan4)
   # chan5 = ss.filtfilt(b, a, chan5)
   # chan6 = ss.filtfilt(b, a, chan6)
   # chan7 = ss.filtfilt(b, a, chan7)
   # chan8 = ss.filtfilt(b, a, chan8)

   # plt.figure(figsize=[6.4, 2.4]) 
   # i=1
   # for column in [chan1, chan2,chan3, chan4,chan5, chan6,chan7, chan8]:    
   #     plt.plot(column,label="channel"+str(i))
   #     plt.legend(loc='best')
   #     i+=1

   # st.pyplot(plt)

   chan1[np.abs(chan1)>5] = 0
   chan2[np.abs(chan2)>5] = 0
   chan3[np.abs(chan3)>5] = 0
   chan4[np.abs(chan4)>5] = 0
   chan5[np.abs(chan5)>5] = 0
   chan6[np.abs(chan6)>5] = 0
   chan7[np.abs(chan7)>5] = 0
   chan8[np.abs(chan8)>5] = 0
   # plt.figure(figsize=[6.4, 2.4]) 
   # i=1
   # for column in [chan1, chan2,chan3, chan4,chan5, chan6,chan7, chan8]:    
   #     plt.plot(column,label="channel"+str(i))
   #     plt.legend(loc='best')
   #     i+=1

   # st.pyplot(plt)

   df['channel1'] = chan1
   df['channel2'] = chan2
   df['channel3'] = chan3
   df['channel4'] = chan4
   df['channel5'] = chan5
   df['channel6'] = chan6
   df['channel7'] = chan7
   df['channel8'] = chan8

   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Noise Removed data</h1>", unsafe_allow_html=True)
   st.write(df)
   fig = px.line(df,x=np.arange(len(df)),y=[df['channel1'],df['channel2'],df['channel3'],df['channel4'],df['channel5'],df['channel6'],df['channel7'],df['channel8']])
   fig.update_layout( yaxis_title='EEG Readings (microvolts)',xaxis_title='Timesteps (milli seconds)')
   st.write(fig)

   from scipy.integrate import simps
   import scipy.stats as sst
   from matplotlib.mlab import psd
   fs=250
   no=fs*2
   # def bandpower(trace,band):
   #     f, psd = ss.welch(trace, fs=250,nperseg=no)
   #     #total_power1 = simps(psd, dx=0.1)
   #     power = np.sum(psd[(f >= band[0]) & (f <= band[1])])
   #     return power



   # alpha = np.zeros((12,8))
   # beta = np.zeros((12,8))
   # gamma = np.zeros((12,8))
   # theta = np.zeros((12,8))
   # delta = np.zeros((12,8))

   # c=0
   # for i in np.arange(0,len(df),len(df)//10):
   #     print(i)
   #     X1=df['channel1']
   #     X2=df['channel2']
   #     X3=df['channel3']
   #     X4=df['channel4']
   #     X5=df['channel5']    
   #     X6=df['channel6']
   #     X7=df['channel7']
   #     X8=df['channel8']

   #     print(str(i)+" : "+str(i+len(df)//10))
   #     end=i+len(df)//10
   #     X1=X1[i:end]
   #     X2=X2[i:end]
   #     X3=X3[i:end]
   #     X4=X4[i:end]
   #     X5=X5[i:end]
   #     X6=X6[i:end]
   #     X7=X7[i:end]  
   #     X8=X8[i:end] 

   #     alpha[c,0] = bandpower(X1,[8,12])
   #     alpha[c,1] = bandpower(X2,[8,12])
   #     alpha[c,2] = bandpower(X3,[8,12])
   #     alpha[c,3] = bandpower(X4,[8,12])
   #     alpha[c,4] = bandpower(X5,[8,12])
   #     alpha[c,5] = bandpower(X6,[8,12])
   #     alpha[c,6] = bandpower(X7,[8,12])
   #     alpha[c,7] = bandpower(X8,[8,12])

   #     beta[c,0] = bandpower(X1,[12,30])
   #     beta[c,1] = bandpower(X2,[12,30])
   #     beta[c,2] = bandpower(X3,[12,30])
   #     beta[c,3] = bandpower(X4,[12,30])
   #     beta[c,4] = bandpower(X5,[12,30])
   #     beta[c,5] = bandpower(X6,[12,30])
   #     beta[c,6] = bandpower(X7,[12,30])
   #     beta[c,7] = bandpower(X8,[12,30])


   #     gamma[c,0] = bandpower(X1,[30,100])
   #     gamma[c,1] = bandpower(X2,[30,100])
   #     gamma[c,2] = bandpower(X3,[30,100])
   #     gamma[c,3] = bandpower(X4,[30,100])
   #     gamma[c,4] = bandpower(X5,[30,100])
   #     gamma[c,5] = bandpower(X6,[30,100])
   #     gamma[c,6] = bandpower(X7,[30,100])
   #     gamma[c,7] = bandpower(X8,[30,100])


   #     theta[c,0] = bandpower(X1,[4,7])
   #     theta[c,1] = bandpower(X2,[4,7])
   #     theta[c,2] = bandpower(X3,[4,7])
   #     theta[c,3] = bandpower(X4,[4,7])
   #     theta[c,4] = bandpower(X5,[4,7])
   #     theta[c,5] = bandpower(X6,[4,7])
   #     theta[c,6] = bandpower(X7,[4,7])
   #     theta[c,7] = bandpower(X8,[4,7])

   #     delta[c,0] = bandpower(X1,[0.5,4])
   #     delta[c,1] = bandpower(X2,[0.5,4])
   #     delta[c,2] = bandpower(X3,[0.5,4])
   #     delta[c,3] = bandpower(X4,[0.5,4])
   #     delta[c,4] = bandpower(X5,[0.5,4])
   #     delta[c,5] = bandpower(X6,[0.5,4])
   #     delta[c,6] = bandpower(X7,[0.5,4])
   #     delta[c,7] = bandpower(X8,[0.5,4])

   #     c+=1

   # alpha_bands = pd.DataFrame(alpha, columns = ['alpha_power_1','alpha_power_2','alpha_power_3','alpha_power_4','alpha_power_5','alpha_power_6','alpha_power_7','alpha_power_8'])
   # beta_bands = pd.DataFrame(beta, columns = ['beta_power_1','beta_power_2','beta_power_3','beta_power_4','beta_power_5','beta_power_6','beta_power_7','beta_power_8'])
   # gamma_bands = pd.DataFrame(gamma, columns = ['gamma_power_1','gamma_power_2','gamma_power_3','gamma_power_4','gamma_power_5','gamma_power_6','gamma_power_7','gamma_power_8'])
   # theta_bands = pd.DataFrame(theta, columns = ['theta_power_1','theta_power_2','theta_power_3','theta_power_4','theta_power_5','theta_power_6','theta_power_7','theta_power_8'])
   # delta_bands = pd.DataFrame(delta, columns = ['delta_power_1','delta_power_2','delta_power_3','delta_power_4','delta_power_5','delta_power_6','delta_power_7','delta_power_8'])


   # df_combined = pd.concat([alpha_bands,beta_bands,gamma_bands,theta_bands,delta_bands], axis=1)
   # st.write(df_combined)
   # df_combined.to_csv("eeg_datacombined_yoga.csv")

   channels=['FP1','FP2','C3','C4','T5','T6','O1','O2']
   fs=250
   no=fs*2
   f, psd = ss.welch(df['channel1'], fs=250,nperseg=no)
   beta_power1 = np.sum(psd[(f >= 13) & (f <= 22)]) # Compute the beta wave power

   f, psd = ss.welch(df['channel2'], fs=250,nperseg=no)
   beta_power2 = np.sum(psd[(f >= 13) & (f <= 22)]) 

   f, psd = ss.welch(df['channel3'], fs=250,nperseg=no)
   beta_power3 = np.sum(psd[(f >= 13) & (f <= 22)]) 

   f, psd = ss.welch(df['channel4'], fs=250,nperseg=no)
   beta_power4 = np.sum(psd[(f >= 13) & (f <= 22)]) 

   f, psd = ss.welch(df['channel5'], fs=250,nperseg=no)
   beta_power5 = np.sum(psd[(f >= 13) & (f <= 22)]) 

   f, psd = ss.welch(df['channel6'], fs=250,nperseg=no)
   beta_power6 = np.sum(psd[(f >= 13) & (f <= 22)]) 

   f, psd = ss.welch(df['channel7'], fs=250,nperseg=no)
   beta_power7 =np.sum(psd[(f >= 13) & (f <= 22)]) 

   f, psd = ss.welch(df['channel8'], fs=250,nperseg=no)
   beta_power8 = np.sum(psd[(f >= 13) & (f <= 22)])   

   beta_power=[beta_power1,beta_power2,beta_power3,beta_power4,beta_power5,beta_power6,beta_power7,beta_power8]

   print(beta_power1,beta_power2,beta_power3,beta_power4,beta_power5,beta_power6,beta_power7,beta_power8)



   # fig = px.bar(df, x=channels, y=beta_power, color=channels,
   #               pattern_shape_sequence=[".", "x", "+"])

   # st.info("BETA waves")
   # #with st.beta_expander("Write a review 📝"):
   # st.write(fig)


   # plt.figure(figsize=[6.4, 2.4])  
   # for column in beta_bands[['beta_power_1','beta_power_2','beta_power_3','beta_power_4','beta_power_5','beta_power_6','beta_power_7','beta_power_8']]:    
   #     plt.plot(beta_bands[column],label=column)
   #     plt.xlabel("Time / s")
   #     plt.ylabel("beta power")
   #     plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
   # st.pyplot(plt)



   f, psd = ss.welch(df['channel1'], fs=250,nperseg=no)
   alpha_power1 = np.sum(psd[(f >= 8) & (f <= 13)]) # Compute the alpha wave power

   f, psd = ss.welch(df['channel2'], fs=250,nperseg=no)
   alpha_power2 = np.sum(psd[(f >= 8) & (f <= 13)]) 

   f, psd = ss.welch(df['channel3'], fs=250,nperseg=no)
   alpha_power3 = np.sum(psd[(f >= 8) & (f <= 13)]) 

   f, psd = ss.welch(df['channel4'], fs=250,nperseg=no)
   alpha_power4 = np.sum(psd[(f >= 8) & (f <= 13)]) 

   f, psd = ss.welch(df['channel5'], fs=250,nperseg=no)
   alpha_power5 = np.sum(psd[(f >= 8) & (f <= 13)]) 

   f, psd = ss.welch(df['channel6'], fs=250,nperseg=no)
   alpha_power6 = np.sum(psd[(f >= 8) & (f <= 13)]) 

   f, psd = ss.welch(df['channel7'], fs=250,nperseg=no)
   alpha_power7 = np.sum(psd[(f >= 8) & (f <= 13)]) 

   f, psd = ss.welch(df['channel8'], fs=250,nperseg=no)
   alpha_power8 = np.sum(psd[(f >= 8) & (f <= 13)])  

   alpha_power=[alpha_power1,alpha_power2,alpha_power3,alpha_power4,alpha_power5,alpha_power6,alpha_power7,alpha_power8]

   print(alpha_power1,alpha_power2,alpha_power3,alpha_power4,alpha_power5,alpha_power6,alpha_power7,alpha_power8)


   # fig = px.bar(df, x=channels, y=alpha_power, color=channels,
   #               pattern_shape_sequence=[".", "x", "+"])

   # st.info("Alpha waves")
   # st.write(fig)
   # plt.figure(figsize=[6.4, 2.4])  
   # for column in alpha_bands[['alpha_power_1','alpha_power_2','alpha_power_3','alpha_power_4','alpha_power_5','alpha_power_6','alpha_power_7','alpha_power_8']]:    
   #     plt.plot(alpha_bands[column],label=column)
   #     plt.xlabel("Time / s")
   #     plt.ylabel("alpha power")
   #     plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
   # st.pyplot(plt)



   f, psd = ss.welch(df['channel1'], fs=250,nperseg=no)
   gamma_power1 = np.sum(psd[(f >= 30) & (f <= 100)])  # Compute the alpha wave power

   f, psd = ss.welch(df['channel2'], fs=250,nperseg=no)
   gamma_power2 = np.sum(psd[(f >= 30 )& (f <= 100)]) 

   f, psd = ss.welch(df['channel3'], fs=250,nperseg=no)
   gamma_power3 = np.sum(psd[(f >= 30) & (f <= 100)]) 

   f, psd = ss.welch(df['channel4'], fs=250,nperseg=no)
   gamma_power4 = np.sum(psd[(f >= 30) & (f <= 100)]) 

   f, psd = ss.welch(df['channel5'], fs=250,nperseg=no)
   gamma_power5 = np.sum(psd[(f >= 30) & (f <= 100)]) 

   f, psd = ss.welch(df['channel6'], fs=250,nperseg=no)
   gamma_power6 = np.sum(psd[(f >= 30) & (f <= 100)]) 

   f, psd = ss.welch(df['channel7'], fs=250,nperseg=no)
   gamma_power7 = np.sum(psd[(f >= 30) & (f <= 100)]) 

   f, psd = ss.welch(df['channel8'], fs=250,nperseg=no)
   gamma_power8 = np.sum(psd[(f >= 30) & (f <= 100)])  

   gamma_power=[gamma_power1,gamma_power2,gamma_power3,gamma_power4,gamma_power5,gamma_power6,gamma_power7,gamma_power8]


   # import plotly.express as px

   # fig = px.bar(df, x=channels, y=gamma_power, color=channels,
   #               pattern_shape_sequence=[".", "x", "+"])

   # st.info("Gamma waves")
   # st.write(fig)
   # plt.figure(figsize=[6.4, 2.4])  
   # for column in gamma_bands[['gamma_power_1','gamma_power_2','gamma_power_3','gamma_power_4','gamma_power_5','gamma_power_6','gamma_power_7','gamma_power_8']]:    
   #     plt.plot(gamma_bands[column],label=column)
   #     plt.xlabel("Time / s")
   #     plt.ylabel("gamma power")
   #     plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
   # st.pyplot(plt)


   f, psd = ss.welch(df['channel1'], fs=250,nperseg=no)
   delta_power1 = np.sum(psd[(f >= 0.5) & (f <= 4)]) 

   f, psd = ss.welch(df['channel2'], fs=250,nperseg=no)
   delta_power2 = np.sum(psd[(f >= 0.5) & (f <= 4)])

   f, psd = ss.welch(df['channel3'], fs=250,nperseg=no)
   delta_power3 =np.sum(psd[(f >= 0.5) & (f <= 4)]) 

   f, psd = ss.welch(df['channel4'], fs=250,nperseg=no)
   delta_power4 = np.sum(psd[(f >= 0.5) & (f <= 4)])

   f, psd = ss.welch(df['channel5'], fs=250,nperseg=no)
   delta_power5 = np.sum(psd[(f >= 0.5) & (f <= 4)])

   f, psd = ss.welch(df['channel6'], fs=250,nperseg=no)
   delta_power6 =np.sum(psd[(f >= 0.5) & (f <= 4)])

   f, psd = ss.welch(df['channel7'], fs=250,nperseg=no)
   delta_power7 =np.sum(psd[(f >= 0.5) & (f <= 4)]) 

   f, psd = ss.welch(df['channel8'], fs=250,nperseg=no)
   delta_power8 = np.sum(psd[(f >= 0.5) & (f <= 4)])   

   delta_power=[delta_power1,delta_power2,delta_power3,delta_power4,delta_power5,delta_power6,delta_power7,delta_power8]

   print(delta_power1,delta_power2,delta_power3,delta_power4,delta_power5,delta_power6,delta_power7,delta_power8)

   # import plotly.express as px

   # fig = px.bar(df, x=channels, y=delta_power, color=channels,
   #               pattern_shape_sequence=[".", "x", "+"])
   # st.info("Delta waves")
   # st.write(fig)


   # plt.figure(figsize=[6.4, 2.4])  
   # for column in delta_bands[['delta_power_1','delta_power_2','delta_power_3','delta_power_4','delta_power_5','delta_power_6','delta_power_7','delta_power_8']]:    
   #     plt.plot(delta_bands[column],label=column)
   #     plt.xlabel("Time / s")
   #     plt.ylabel("delta power")
   #     plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
   # st.pyplot(plt)    


   f, psd = ss.welch(df['channel1'], fs=250,nperseg=no)
   theta_power1 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel2'], fs=250,nperseg=no)
   theta_power2 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel3'], fs=250,nperseg=no)
   theta_power3 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel4'], fs=250,nperseg=no)
   theta_power4 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel5'], fs=250,nperseg=no)
   theta_power5 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel6'], fs=250,nperseg=no)
   theta_power6 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel7'], fs=250,nperseg=no)
   theta_power7 = np.sum(psd[(f >= 4) & (f <= 7)]) 

   f, psd = ss.welch(df['channel8'], fs=250,nperseg=no)
   theta_power8 = np.sum(psd[(f >= 4) & (f <= 7)])   

   theta_power=[theta_power1,theta_power2,theta_power3,theta_power4,theta_power5,theta_power6,theta_power7,theta_power8]


   alpha_totalpower=np.sum([alpha_power1,alpha_power3,alpha_power5,alpha_power7,alpha_power2,alpha_power4,alpha_power6,alpha_power8])
   beta_totalpower=np.sum([beta_power1,beta_power3,beta_power5,beta_power7,beta_power2,beta_power4,beta_power6,beta_power8])
   gamma_totalpower=np.sum([gamma_power1,gamma_power3,gamma_power5,gamma_power7,gamma_power2,gamma_power4,gamma_power6,gamma_power8])
   delta_totalpower=np.sum([delta_power1,delta_power3,delta_power5,delta_power7,delta_power2,delta_power4,delta_power6,delta_power8])
   theta_totalpower=np.sum([theta_power1,theta_power3,theta_power5,theta_power7,theta_power2,theta_power4,theta_power6,theta_power8])

   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Total Power vs Frequency bands</h1>", unsafe_allow_html=True)
   bands=['alpha (Relax)','beta (Engaged)','gamma (Concentration)','delta (Deep sleep)','theta (Dowsy)']
   powers=[alpha_totalpower,beta_totalpower,gamma_totalpower,delta_totalpower,theta_totalpower]
   import plotly.express as px

   fig = px.bar(df, x=bands, y=powers,labels=dict(x="Frequency bands", y="Power spectral density [µV^2/Hz]"), color=bands,
                 pattern_shape_sequence=[".", "x", "+"],text=powers
   )

   fig.update_traces(textposition='outside')
   fig.update_layout(yaxis_range=[0,0.5])

   # fig.update_xaxes(title_font=dict(color='blue'))
   # fig.update_yaxes(title_font=dict(color='blue'))
   #px.update_traces(textposition='top center')
   st.write(fig)



   # import plotly.express as px

   # fig = px.bar(df, x=channels, y=theta_power, color=channels,
   #               pattern_shape_sequence=[".", "x", "+"])

   # st.info("Theta waves")
   # st.write(fig)
   # plt.figure(figsize=[6.4, 2.4])  
   # for column in theta_bands[['theta_power_1','theta_power_2','theta_power_3','theta_power_4','theta_power_5','theta_power_6','theta_power_7','theta_power_8']]:    
   #     plt.plot(theta_bands[column],label=column)
   #     plt.xlabel("Time / s")
   #     plt.ylabel("theta power")
   #     plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1],ncol=2, fontsize="smaller")
   # st.pyplot(plt)   


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

   plot = px.Figure(data=[
       px.Bar(
       name = 'alpha (Relax)',
       x = x,
       y = [left_alpha,right_alpha],
       text=[left_alpha,right_alpha]
       ),
       px.Bar(
       name = 'beta (Engaged)',
       x = x,
       y = [left_beta,right_beta],
       text=[left_beta,right_beta]
       ),
             px.Bar(
       name = 'gamma (Concentration)',
       x = x,
       y = [left_gamma,right_gamma],
       text=[left_gamma,right_gamma]
       ), 
           px.Bar(
       name = 'delta (Deep sleep)',
       x = x,
       y = [left_delta,right_delta],
       text=[left_delta,right_delta]
       ),                   

             px.Bar(
       name = 'theta (Dowsy)',
       x = x,
       y = [left_theta,right_theta],

       text=[left_theta,right_theta]

       )                      
   ])

   plot.update_layout(yaxis_range=[0,0.5], yaxis_title='Power spectral density [µV^2/Hz]',xaxis_title='Parts of brain')

   st.write(plot)


   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Physical and mental characteristics (Frontal lobe) </h1>", unsafe_allow_html=True)
   bands=['Relaxed','Engaged','Concentrated','Sleepy','Dowsy']

   percent_alpha=((alpha_power1+alpha_power2)/alpha_totalpower)*100
   percent_beta=((beta_power1+beta_power2)/beta_totalpower)*100
   percent_gamma=((gamma_power1+gamma_power2)/gamma_totalpower)*100
   percent_delta=((delta_power1+delta_power2)/delta_totalpower)*100
   percent_theta=((theta_power1+theta_power2)/theta_totalpower)*100


   powers=[percent_alpha,percent_beta,percent_gamma,percent_delta,percent_theta]
   import plotly.express as px

   fig = px.bar(df, x=bands, y=powers,labels=dict(x="Mental and Physical State", y="Power spectral density [%]"), color=bands)

   fig.update_traces(textposition='outside')
   fig.update_layout(yaxis_ticksuffix = "%")
   st.write(fig)


   st.markdown("<h1 style='text-align:center; color:black;background-color:#B5C489;font-size:14pt;border:5px solid black;'>Visual Processing skills (Occipital lobe) </h1>", unsafe_allow_html=True)
   bands=['Relaxed','Engaged','Concentrated','Sleepy','Dowsy']

   percent_alpha=((alpha_power7+alpha_power8)/alpha_totalpower)*100
   percent_beta=((beta_power7+beta_power8)/beta_totalpower)*100
   percent_gamma=((gamma_power7+gamma_power8)/gamma_totalpower)*100
   percent_delta=((delta_power7+delta_power8)/delta_totalpower)*100
   percent_theta=((theta_power7+theta_power8)/theta_totalpower)*100


   powers=[percent_alpha,percent_beta,percent_gamma,percent_delta,percent_theta]
   import plotly.express as px

   fig = px.bar(df, x=bands, y=powers,labels=dict(x="Visual Processing State", y="Power spectral density [%]"), color=bands)

   fig.update_traces(textposition='outside')
   fig.update_layout(yaxis_ticksuffix = "%")
   st.write(fig)



   prediction_input=np.array([[alpha_power1,alpha_power3,alpha_power5,alpha_power7,alpha_power2,alpha_power4,alpha_power6,alpha_power8,
                               beta_power1,beta_power3,beta_power5,beta_power7,beta_power2,beta_power4,beta_power6,beta_power8,
                               gamma_power1,gamma_power3,gamma_power5,gamma_power7,gamma_power2,gamma_power4,gamma_power6,gamma_power8,
                               theta_power1,theta_power3,theta_power5,theta_power7,theta_power2,theta_power4,theta_power6,theta_power8,
                               delta_power1,delta_power3,delta_power5,delta_power7,delta_power2,delta_power4,delta_power6,delta_power8
                               ]])

   scaler=StandardScaler()
   prediction_input_scaled=scaler.fit_transform(prediction_input.reshape(-1,1))
   #prediction_input_scaled
   #st.info(prediction_input)

   #st.error(prediction_input_scaled.tolist())
   result=['normal','pain','yoga']
   eegnet_model =tf.keras.models.load_model('eeg_2dcnn_model.h5')
   o=eegnet_model.predict(prediction_input_scaled.reshape(-1,40,1,1), batch_size=1)
   #st.write(o)

   e="EEGNET PREDICTS : "+ result[int(o.argmax())]
   st.success(e)


   svm_model = pickle.load(open('svm_model.pkl','rb'))
   #st.warning(prediction_input_scaled.reshape(1,-1).tolist())
   o=svm_model.predict(prediction_input_scaled.reshape(1,-1).tolist())

   #st.write(o)
   s="SVM PREDICTS : "+result[o[0]]
   st.info(s)

   bilstm_model =tf.keras.models.load_model('bilstm_model.h5')
   o=bilstm_model.predict(prediction_input_scaled.reshape(-1,1,40), batch_size=1)
   #st.write(o)
   b="BILSTM PREDICTS : "+ result[int(o.argmax())]
   st.warning(b)


