'''
 Streamlit del proyecto 
 
 '''
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import*


st.title("Proyecto 1")
st.header("Métodos cuantitativos en finanzas")


#Funciones
def obtener_datos(activo): 
    
    # Descargar los datos
    df = yf.download(activo, start = "2010-01-01", progress=False )['Close']
    #df = yf.download(activo, period='1y')['Close']
    return df

def calcular_rendimientos(df):
     return df.pct_change().dropna()

#Activo
activo_escogido = ['BTC-USD']

#activo_proyecto = activo_escogido[0]
 
 #Carga de página
with st.spinner("Descargando datos..."):
     df_precios = obtener_datos(activo_escogido)
     df_rendimientos = calcular_rendimientos(df_precios)

 
if activo_escogido:    
     #Métricas del activo
     st.subheader(f"Métricas de Rendimiento: {activo_escogido[0]}")
     rendimiento_medio = df_rendimientos[activo_escogido[0]].mean() #Columna de los datos
     Kurtosis = kurtosis(df_rendimientos[activo_escogido[0]])
     skew = skew(df_rendimientos[activo_escogido[0]])


     
     col1, col2, col3= st.columns(3)
     col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
     col2.metric("Kurtosis", f"{Kurtosis:.4}")
     col3.metric("Skew", f"{skew:.2}")
     

     # Gráfico de rendimientos diarios
     st.subheader(f"Gráfico de Rendimientos: {activo_escogido[0]}")
     fig, ax = plt.subplots(figsize=(13, 5))
     ax.plot(df_rendimientos.index, df_rendimientos[activo_escogido], label=activo_escogido)
     ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
     ax.legend()
     ax.set_title(f"Rendimientos de {activo_escogido[0]}")
     ax.set_xlabel("Fecha")
     ax.set_ylabel("Rendimiento Diario")
     st.pyplot(fig)
     
     # Histograma de rendimientos
     st.subheader("Distribución de Rendimientos")
     fig, ax = plt.subplots(figsize=(10, 5))
     ax.hist(df_rendimientos[activo_escogido], bins=30, alpha=0.7, color='blue', edgecolor='black')
     ax.axvline(rendimiento_medio, color='red', linestyle='dashed', linewidth=2, label=f"Promedio: {rendimiento_medio:.4%}")
     ax.legend()
     ax.set_title("Histograma de Rendimientos")
     ax.set_xlabel("Rendimiento Diario")
     ax.set_ylabel("Frecuencia")
     st.pyplot(fig)

     st.subheader('Test de Normalidad (Shapiro-Wilk)')

     stat , p = shapiro(df_rendimientos[activo_escogido[0]])
     st.write(f'Shapirop-Wilk Test : {stat:.4}' )
     st.write(f'P_value : {p:.6}')


     
     #VaR paramétrico
     
     st.header("VaR (Asuminedo una distribucion Normal)")

     
     st.subheader("Aproximación paramétrica")
     promedio = np.mean(df_rendimientos[activo_escogido[0]])
     stdev = np.std(df_rendimientos[activo_escogido[0]])


     VaR_95 = norm.ppf(1-0.95,promedio,stdev)
     VaR_975 = norm.ppf(1-0.975,promedio,stdev)
     VaR_99 = norm.ppf(1-0.99,promedio,stdev)

     col4, col5, col6 = st.columns(3)
     col4.metric("VaR con 95% de confianza", f"{VaR_95:.4}")
     col5.metric("VaR con 97.5% de confianza", f"{VaR_975:.4}")
     col6.metric("VaR con 99% de confianza", f"{VaR_99:.4}")

     
     
     st.subheader("Aproximación Histórica")
     # Historical VaR
     hVaR_95 = (df_rendimientos[activo_escogido[0]].quantile(0.05))
     hVaR_975 = (df_rendimientos[activo_escogido[0]].quantile(0.025))
     hVaR_99 = (df_rendimientos[activo_escogido[0]].quantile(0.01))

     col7, col8, col9 = st.columns(3)
     col7.metric("hVaR con 95% de confianza", f"{hVaR_95:.4}")
     col8.metric("hVaR con 97.5% de confianza", f"{hVaR_975:.4}")
     col9.metric("hVaR con 99% de confianza", f"{hVaR_99:.4}")


     st.subheader("Monte Carlo")
    # Monte Carlo
    # Number of simulations
     n_sims = 100000

      # Simulate returns and sort
     sim_returns = np.random.normal(promedio, stdev, n_sims)

     MCVaR_95 = np.percentile(sim_returns, 5)
     MCVaR_975 = np.percentile(sim_returns, 2.5)
     MCVaR_99 = np.percentile(sim_returns, 1)

     col10, col11, col12 = st.columns(3)
     col10.metric("MCVaR con 95% de confianza", f"{MCVaR_95:.4}")
     col11.metric("MCVaR con 97.5% de confianza", f"{MCVaR_975:.4}")
     col12.metric("MCVaR con 99% de confianza", f"{MCVaR_99:.4}")


     st.subheader("CVaR (ES)")
     CVaR_95 = (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR_95].mean())
     CVaR_975 = (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR_975].mean())
     CVaR_99= (df_rendimientos[activo_escogido[0]][df_rendimientos[activo_escogido[0]] <= hVaR_99].mean())


     col13, col14, col15 = st.columns(3)
     col13.metric("CVaR con 95% de confianza", f"{CVaR_95:.4}")
     col14.metric("CVaR con 97.5% de confianza", f"{CVaR_975:.4}")
     col15.metric("CVaR con 99% de confianza", f"{CVaR_99:.4}")




     fig_2, ax_2 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR_95:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_2.axvline(x=VaR_95, color='black', linestyle='--', label='VaR 95% (Paramétrico)')
     ax_2.axvline(x=MCVaR_95, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
     ax_2.axvline(x=hVaR_95, color='green', linestyle='--', label='VaR 95% (Aprox.Histórico)')
     ax_2.axvline(x=CVaR_95, color='purple', linestyle='-.', label='CVaR 95%')

# Add a legend and labels to make the chart more informative
     ax_2.set_title('Histograma de los Retornos con VaR y CVaR al 95%')
     ax_2.set_xlabel('Retornos')
     ax_2.set_ylabel('Frequencia')
     ax_2.legend()
     st.pyplot(fig_2)

       # Create a DataFrame with various VaR and CVaR calculations
     out = pd.DataFrame({'VaR (Normal)': [VaR_95 * 100],
                    'VaR (Historical)': [hVaR_95 * 100],
                    'VaR (Monte Carlo)': [MCVaR_95 * 100],
                    'CVaR': [CVaR_95 * 100]},
                   index=['95% Confidence'])

# Display the DataFrame
     out





     fig_3, ax_3 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR_975:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_3.axvline(x=VaR_975, color='black', linestyle='--', label='VaR 97.5% (Paramétrico)')
     ax_3.axvline(x=MCVaR_975, color='grey', linestyle='--', label='VaR 97.5% (Monte Carlo)')
     ax_3.axvline(x=hVaR_975, color='green', linestyle='--', label='VaR 97.5% (Aprox. Histórico)')
     ax_3.axvline(x=CVaR_975, color='purple', linestyle='-.', label='CVaR 97.5%')



# Add a legend and labels to make the chart more informative
     ax_3.set_title('Histograma de los Retornos con VaR y CVaR al 97.5%')
     ax_3.set_xlabel('Retornos')
     ax_3.set_ylabel('Frequencia')
     ax_3.legend()
     st.pyplot(fig_3)

     # Create a DataFrame with various VaR and CVaR calculations
     out = pd.DataFrame({'VaR (Normal)': [VaR_975 * 100],
                    'VaR (Historical)': [hVaR_975 * 100],
                    'VaR (Monte Carlo)': [MCVaR_975 * 100],
                    'CVaR': [CVaR_975 * 100]},
                   index=['97.5% Confidence'])

# Display the DataFrame
     out

     
     
     
     fig_4, ax_4 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR_99:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_4.axvline(x=VaR_99, color='black', linestyle='--', label='VaR 99% (Paramétrico)')
     ax_4.axvline(x=MCVaR_99, color='grey', linestyle='--', label='VaR 99% (Monte Carlo)')
     ax_4.axvline(x=hVaR_99, color='green', linestyle='--', label='VaR 99% (Aprox. Histórico)')
     ax_4.axvline(x=CVaR_99, color='purple', linestyle='-.', label='CVaR 99%')

# Add a legend and labels to make the chart more informative
     ax_4.set_title('Histograma de los Retornos con VaR y CVaR al 99%')
     ax_4.set_xlabel('Retornos')
     ax_4.set_ylabel('Frequencia')
     ax_4.legend()
     st.pyplot(fig_4)


     # Create a DataFrame with various VaR and CVaR calculations
     out = pd.DataFrame({'VaR (Normal)': [VaR_99 * 100],
                    'VaR (Historical)': [hVaR_99 * 100],
                    'VaR (Monte Carlo)': [MCVaR_99 * 100],
                    'CVaR': [CVaR_99 * 100]},
                   index=['99% Confidence'])

# Display the DataFrame
     out





     st.header("VaR (Asuminedo una distribucion t de Student)")

     df_grados_de_libertad = len(df_rendimientos)-1

     st.subheader("VaR Paramétrico")

     t_cuantil_95 = t.ppf(1-0.95, df_grados_de_libertad )
     VaR_t_95 = promedio + t_cuantil_95 * stdev
     t_cuantil_975 = t.ppf(1-0.975, df_grados_de_libertad )
     VaR_t_975 = promedio + t_cuantil_975 * stdev
     t_cuantil_99 = t.ppf(1-0.99, df_grados_de_libertad )
     VaR_t_99 = promedio + t_cuantil_99 * stdev

     col16, col17, col18 = st.columns(3)
     col16.metric("VaR con 95% de confianza (t-Student)", f"{VaR_t_95:.4}")
     col17.metric("VaR con 97.5% de confianza (t-Student)", f"{VaR_t_975:.4}")
     col18.metric("VaR con 99% de confianza (t-Student)", f"{VaR_t_99:.4}")




     st.subheader("Monte Carlo")
    # Monte Carlo
    # Number of simulations
     n_sims = 100000

      # Simulate returns and sort
     sim_returns = promedio + stdev * np.random.standard_t(df_grados_de_libertad, size=n_sims)

     MCVaR_t_95 = np.percentile(sim_returns, 5)
     MCVaR_t_975 = np.percentile(sim_returns, 2.5)
     MCVaR_t_99 = np.percentile(sim_returns, 1)

     col19, col20, col21 = st.columns(3)
     col19.metric("MCVaR con 95% de confianza", f"{MCVaR_t_95:.4}")
     col20.metric("MCVaR con 97.5% de confianza", f"{MCVaR_t_975:.4}")
     col21.metric("MCVaR con 99% de confianza", f"{MCVaR_t_99:.4}")

    
    
     fig_5, ax_5 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR_95:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_5.axvline(x=VaR_t_95, color='black', linestyle='--', label='VaR (t de student) 95% (Paramétrico)')
     ax_5.axvline(x=MCVaR_t_95, color='grey', linestyle='--', label='VaR 95% (Monte Carlo)')
     ax_5.axvline(x=hVaR_95, color='green', linestyle='--', label='VaR 95% (Aprox. Histórico)')
     ax_5.axvline(x=CVaR_95, color='purple', linestyle='-.', label='CVaR 95%')

# Add a legend and labels to make the chart more informative
     ax_5.set_title('Histograma de los Retornos con VaR y CVaR al 95%')
     ax_5.set_xlabel('Retornos')
     ax_5.set_ylabel('Frequencia')
     ax_5.legend()
     st.pyplot(fig_5)



     fig_6, ax_6 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR_975:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_6.axvline(x=VaR_t_975, color='black', linestyle='--', label='VaR (t de student) 97.5% (Paramétrico)')
     ax_6.axvline(x=MCVaR_t_975, color='grey', linestyle='--', label='VaR 97.5% (Monte Carlo)')
     ax_6.axvline(x=hVaR_975, color='green', linestyle='--', label='VaR 97.5% (Aprox. Histórico)')
     ax_6.axvline(x=CVaR_975, color='purple', linestyle='-.', label='CVaR 97.5%')

# Add a legend and labels to make the chart more informative
     ax_6.set_title('Histograma de los Retornos con VaR y CVaR al 97.5%')
     ax_6.set_xlabel('Retornos')
     ax_6.set_ylabel('Frequencia')
     ax_6.legend()
     st.pyplot(fig_6)


     fig_7, ax_7 = plt.subplots(figsize=(10, 5))
     n, bins, patches = plt.hist(df_rendimientos[activo_escogido[0]], bins=50, color='blue', alpha=0.7, label='Retornos')

# Identify bins to the left of hVaR_95 and color them differently
     for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
          if bin_left < hVaR_99:
              patch.set_facecolor('red')

# Mark the different VaR and CVaR values on the histogram
     ax_7.axvline(x=VaR_t_99, color='black', linestyle='--', label='VaR (t de student) 99% (Paramétrico)')
     ax_7.axvline(x=MCVaR_t_99, color='grey', linestyle='--', label='VaR 99% (Monte Carlo)')
     ax_7.axvline(x=hVaR_99, color='green', linestyle='--', label='VaR 99% (Aprox. Histórico)')
     ax_7.axvline(x=CVaR_99, color='purple', linestyle='-.', label='CVaR 99%')

# Add a legend and labels to make the chart more informative
     ax_7.set_title('Histograma de los Retornos con VaR y CVaR al 99%')
     ax_7.set_xlabel('Retornos')
     ax_7.set_ylabel('Frequencia')
     ax_7.legend()
     st.pyplot(fig_7)




     

     



