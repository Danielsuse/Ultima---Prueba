'''
 Este es un Scrit generado para almacenar todas las funciones,
 generadas a alolargo de la primera parte del curso de MCF
 
 '''
 
 # Paqueterias necesarias 
 
 # Manejo de datos
import pandas as pd
import numpy as np
  
 # Visualizacion de datos
import matplotlib.pyplot as plt
 
 # Api de Yahoo Finanzas
import yfinance as yf
 
def obtener_datos(activo):
     '''
     El objetivo de esta funcion es descargar el precio
     de cierre de un o varios activos en una ventana de un año
 
     Input = Ticker del activo en string 
     Output = DataFrame del precio del activo
 
     '''
     start_date = "2010-01-01" #Fecha de Inicio
     end_date = "2025-03-30"  #Fecha al día de hoy

     # Descargar los datos
     df = yf.download(activo, start = start_date, end = end_date)['Close']
     return df
 
def calcular_rendimientos(df):
     '''
     Funcion de calcula los rendimientos de un activo
 
     Input = Data Frame de precios por activo
 
     Output = Data Frame de  rendimientos
 
     '''
     return df.pct_change().dropna()
