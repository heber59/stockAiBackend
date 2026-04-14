# Explicación del Proyecto de Predicción de Acciones: XGBoost

Este documento proporciona una explicación detallada del sistema de predicción de acciones desarrollado en este proyecto. Está diseñado para ser compartido con un profesor o evaluador.

---

## 1. Cómo funciona XGBoost (Versión Simple)
**XGBoost** (Extreme Gradient Boosting) es un potente algoritmo de aprendizaje automático basado en "equipos". Imagina que tienes un equipo de 150 estudiantes (árboles) intentando predecir si una acción subirá o bajará:

1.  **El primer estudiante:** Hace una suposición simple. No será perfecta.
2.  **El segundo estudiante:** No empieza de cero. En su lugar, observa dónde cometió errores el primer estudiante e intenta corregirlos.
3.  **Los siguientes estudiantes:** Cada uno se enfoca en corregir los errores del equipo que le precedió.

Al final, las suposiciones individuales "débiles" se combinan en una predicción final muy fuerte y precisa.

---

## 2. Parámetros del Modelo
En el código del proyecto (`stock_model.py`), configuramos XGBoost con estos ajustes específicos:

| Parámetro | Valor | Qué hace |
| :--- | :--- | :--- |
| `n_estimators` | 150 | El número de árboles (estudiantes) en el equipo. Más árboles pueden aprender más, pero demasiados pueden causar sobreajuste (overfitting). |
| `max_depth` | 5 | Qué tan "alto" puede crecer cada árbol. Una profundidad de 5 permite al modelo encontrar patrones complejos sin perderse en el ruido. |
| `learning_rate` | 0.08 | Qué tan rápido aprende el modelo. Usamos un número pequeño para que el modelo se acerque a la respuesta correcta con cuidado. |
| `subsample` | 0.8 | Cada árbol solo ve el 80% de los datos. Esto mantiene a los árboles diferentes entre sí (diversidad). |
| `colsample_bytree` | 0.8 | Cada árbol solo ve el 80% de las características (indicadores). Esto evita que el modelo dependa demasiado de un solo indicador como el RSI. |
| `objective` | `multi:softprob`| Le dice a XGBoost que tenemos 3 categorías (Bajista, Neutral, Alcista) y queremos la probabilidad de cada una. |

---

## 3. Muestra de Datos Reales (AAPL)
El modelo analiza datos históricos de acciones como **AAPL**. Aquí hay una muestra de los datos reales utilizados para el entrenamiento (después del procesamiento):

| Fecha | Precio de Cierre | Retorno 1d | RSI | ATR | Objetivo (0=Bajista, 1=Neutral, 2=Alcista) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-02-18 | 264.35 | -1.61% | 51.63 | 6.55 | 0 |
| 2026-02-19 | 260.57 | -1.42% | 45.45 | 6.50 | 0 |
| 2026-02-20 | 264.57 | +1.53% | 49.70 | 6.50 | 0 |
| 2026-02-23 | 266.17 | +0.60% | 51.33 | 6.47 | 0 |
| 2026-02-24 | 272.14 | +2.23% | 56.94 | 6.63 | 0 |

### Explicación de Columnas (Características)
Transformamos los precios brutos en "Características" (Features) que el modelo puede entender:

1.  **Retornos (1d, 3d, 7d, etc.):** El cambio porcentual en el precio. Esto le dice al modelo si la acción está actualmente en una fase de impulso.
2.  **RSI (Índice de Fuerza Relativa):** Mide si una acción está "sobrecomprada" (demasiado cara) o "sobrevendida" (demasiado barata).
3.  **MACD:** Un indicador de seguimiento de tendencia. Cuando la línea MACD cruza la línea de señal, a menudo indica un cambio de dirección.
4.  **ATR (Rango Verdadero Promedio):** Mide la **Volatilidad**. Le dice al modelo qué tan "nerviosa" está la acción.
5.  **Ratio de Volumen:** Compara el volumen de negociación de hoy con el promedio de 30 días. Un volumen alto suele confirmar un movimiento de precios.
6.  **Soporte y Resistencia:** Niveles de precios clave donde la acción históricamente deja de caer (Soporte) o deja de subir (Resistencia).
7.  **Características del Mercado (SPY y VIX):** El modelo también observa el S&P 500 general (SPY) y el "Índice de Miedo" (VIX) para entender el ánimo económico global.

---

## 4. Estrategia de Predicción
El modelo no solo adivina un precio; clasifica los próximos 7 días en tres categorías:
- **Clase 0 (Bajista):** Esperando una caída de más del 2%.
- **Clase 1 (Neutral):** Esperando que el precio se mantenga mayormente plano (+/- 2%).
- **Clase 2 (Alcista):** Esperando una ganancia de más del 2%.

Utilizamos **Validación Progresiva (Walk-Forward Validation)**, lo que significa que el modelo siempre se prueba con datos que nunca ha visto antes para asegurar que funcione en el mundo real.
