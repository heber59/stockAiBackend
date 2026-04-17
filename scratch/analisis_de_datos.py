import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Configuración de Estilo Visual (Apariencia Académica Premium)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

# 1. Carga y Preparación de Datos
# Usamos el dataset completo desde el archivo raw parquet
print("Cargando datos desde data/raw/AAPL.parquet...")
df = pd.read_parquet('data/raw/AAPL.parquet')

# Procesamiento de fechas y limpieza
df.index = pd.to_datetime(df.index).tz_localize(None)
if df.index.name != 'Date':
    df.index.name = 'Date'
df = df.sort_index()

# Cálculo de Retornos Diarios
# EXPLICACIÓN ACADÉMICA: El retorno diario es el cambio porcentual del precio de cierre.
# Es la unidad básica para analizar el riesgo y la volatilidad.
df['returns_1d'] = df['Close'].pct_change()
returns = df['returns_1d'].dropna()
prices = df['Close']

# Preparación de datos por décadas para análisis comparativo
df['Decada'] = (df.index.year // 10) * 10
decades = sorted(df['Decada'].unique())
decade_data = [df[df['Decada'] == d]['returns_1d'].dropna().values for d in decades]

# 2. Estadísticas Descriptivas (Resumen Académico)
mean_ret = returns.mean()
std_ret = returns.std()
skew_ret = stats.skew(returns)
kurt_ret = stats.kurtosis(returns)

# 3. Generación de Visualizaciones (Layout de 7 Gráficos)
fig = plt.figure(figsize=(16, 22))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3) 

# --- 1. HISTOGRAMA Y CURVA NORMAL ---
ax1 = fig.add_subplot(gs[0, 0])
n, bins, patches = ax1.hist(returns, bins=100, density=True, alpha=0.6, color='royalblue', label='Retornos Reales')
mu, std = stats.norm.fit(returns)
x = np.linspace(returns.min(), returns.max(), 100)
p = stats.norm.pdf(x, mu, std)
ax1.plot(x, p, 'k', linewidth=2, label='Normal Teórica')
ax1.set_title('1. Distribución vs. Modelo Normal')
ax1.set_xlabel('Retorno Diario (Variación Porcentual %)')
ax1.set_ylabel('Densidad de Probabilidad')
ax1.legend()

# --- 2. PRUEBA Q-Q (NORMALIDAD) ---
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(returns, dist="norm", plot=ax2)
ax2.get_lines()[0].set_markersize(2)
ax2.get_lines()[0].set_markerfacecolor('blue')
ax2.set_title('2. Prueba Q-Q (Efecto Colas Pesadas)')
ax2.set_xlabel('Cuantiles Teóricos')
ax2.set_ylabel('Cuantiles de la Muestra')

# --- 3. BOX PLOTS POR DÉCADAS ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.boxplot(decade_data, labels=[f"{int(d)}s" for d in decades], patch_artist=True,
            boxprops=dict(facecolor='lightgreen'))
ax3.set_title('3. Comparativa de Riesgo por Épocas')
ax3.set_xlabel('Década de Observación')
ax3.set_ylabel('Retorno Diario (Variación %)')

# --- 4. ANÁLISIS DE DRAWDOWN (MÁXIMA CAÍDA) ---
ax4 = fig.add_subplot(gs[1, 1])
roll_max = prices.cummax()
drawdown = (prices - roll_max) / roll_max
ax4.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
ax4.set_title('4. Drawdown Histórico (Caída desde Máximos)')
ax4.set_xlabel('Tiempo (Años)')
ax4.set_ylabel('% Pérdida desde el Máximo Anterior')

# --- 5. VOLATILIDAD RODANTE (ANUALIZADA) ---
ax5 = fig.add_subplot(gs[2, :])
vol_roll = returns.rolling(252).std() * np.sqrt(252)
ax5.plot(returns.index, vol_roll, color='orange', linewidth=1)
ax5.set_title('5. Evolución de la Volatilidad (1 Año Rodante)')
ax5.set_xlabel('Tiempo (Años)')
ax5.set_ylabel(r'Riesgo Anualizado ($\sigma$ en %)')
ax5.fill_between(returns.index, 0, vol_roll, color='orange', alpha=0.1)

# --- 6. SESGO RODANTE (MARKET BIAS) ---
ax6 = fig.add_subplot(gs[3, 0])
skew_roll = returns.rolling(252).apply(lambda x: stats.skew(x) if len(x.dropna()) > 100 else np.nan)
ax6.plot(returns.index, skew_roll, color='purple', linewidth=1)
ax6.axhline(0, color='black', linestyle='--')
ax6.set_title('6. Sesgo Rodante (Direccionalidad del Riesgo)')
ax6.set_xlabel('Tiempo (Años)')
ax6.set_ylabel('Coeficiente de Sesgo (Skewness)')
ax6.fill_between(returns.index, 0, skew_roll, where=(skew_roll < 0), color='grey', alpha=0.2)

# --- 7. CURTOSIS RODANTE (RIESGO EXTREMO) ---
ax7 = fig.add_subplot(gs[3, 1])
kurt_roll = returns.rolling(252).apply(lambda x: stats.kurtosis(x) if len(x.dropna()) > 100 else np.nan)
ax7.plot(returns.index, kurt_roll, color='brown', linewidth=1)
ax7.set_title('7. Curtosis Rodante (Frecuencia de Eventos Extremos)')
ax7.set_xlabel('Tiempo (Años)')
ax7.set_ylabel('Exceso de Curtosis')

plt.tight_layout()
plt.savefig('analisis_de_datos.png')
print("\nGráficos actualizados guardados como 'analisis_de_datos.png'")

# 4. Reporte Estadístico Final en Consola
print("\n" + "="*50)
print("REPORTE ESTADÍSTICO ACADÉMICO (ANÁLISIS DE DATOS)")
print("="*50)
print(f"Periodo de Análisis:  {df.index.min().date()} — {df.index.max().date()}")
print(f"Total de Días:        {len(returns)} días de trading")
print("-" * 50)
print(f"Retorno Promedio:     {mean_ret:.6f}")
print(f"Volatilidad (σ):       {std_ret:.6f}")
print(f"Sesgo (Skewness):     {skew_ret:.4f} " + ("(Sesgo Negativo: Riesgo de caídas)" if skew_ret < 0 else "(Sesgo Positivo)"))
print(f"Curtosis:             {kurt_ret:.4f} " + ("(Colas Pesadas: Riesgo Extremo)" if kurt_ret > 0 else "(Colas Ligeras)"))
print("-" * 50)

# Ley de Probabilidad Empírica
m = returns.mean()
s = returns.std()
for sig in [0.5, 1.0, 2.0]:
    dentro = ((returns >= m - sig*s) & (returns <= m + sig*s)).sum()
    pct = (dentro / len(returns)) * 100
    print(f"Datos dentro de {sig}σ: {pct:.1f}% " + (f"(Esperado ~{68 if sig==1 else 95}%)" if sig >= 1 else ""))

print("\nInterpretación: AAPL muestra leptocurtosis significativa, lo que significa que")
print("aunque la mayoría de los días son estables, los eventos extremos son mucho")
print("más frecuentes de lo que predeciría una distribución normal estándar.")
print("="*50)
