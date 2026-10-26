# Piloto: L-BFGS interno frente a Adam

Fecha: 2026-10-26.

## Conclusion

**L-BFGS interno fue la mejor variante aprendida en tiempo a igual calidad
en esta ejecucion**, pero su ventaja frente al L-BFGS convencional fue
pequena: **3,5%**. Es una senal favorable, no una aceleracion robusta
demostrada: se probo solo el escenario 2, una vez por metodo, sin
repeticiones ni balanceo del orden.

Frente al acotado con Adam-200, L-BFGS interno:

- redujo el tiempo hasta igualar simultaneamente misfit y error de modelo
  de 51,02 a **45,30 segundos: 11,2% menos**;
- redujo el entrenamiento total de 20,19 a **5,23 segundos: 74,1% menos**;
- necesito **79 solves**, frente a 71 de Adam-200, para alcanzar ambos
  umbrales.

El resultado muestra el intercambio buscado: entrenar menos eficientemente
en terminos de solves, pero con menor tiempo total. No cambia la recomendacion
por defecto sin una verificacion mas amplia.

## Prueba realizada

Se ejecutaron cuatro inversiones nuevas, una por metodo, sobre el escenario
2 estandar: acustica 2D, adquisicion superficie/pozos, inicio suave y sin
ruido. Cada inversion tuvo exactamente **100 solves equivalentes**,
incluyendo forwards rechazados y adjuntos.

Orden fijo: L-BFGS convencional, Adam-200, Adam-50, L-BFGS interno.
Misma NVIDIA RTX 2000 Ada Generation Laptop GPU, PyTorch 2.8.0+cu128,
un hilo CPU y semilla restablecida antes de cada inversion.
No se impuso determinismo estricto de kernels CUDA.

**No se ejecuto SPECFEM3D.** No se seleccionaron parametros despues de
observar resultados.

## Que cambia y que permanece igual

En las tres variantes aprendidas se conservaron:

- el operador definido positivo, kernel 9 x 9 y mapa espacial 6 x 10;
- el limite espacial `illum_bound=3`, piso positivo y escala gamma;
- la perdida secante ponderada, regularizacion y todos los pares admitidos;
- los parametros aprendidos de la actualizacion anterior;
- el optimizador externo L-BFGS y la busqueda de linea fisica.

Solo cambia el optimizador interno o su presupuesto:

| Variante | Configuracion interna |
|---|---|
| Adam-200 | Adam, lr=0,03, 200 pasos por entrenamiento |
| Adam-50 | Adam, lr=0,03, 50 pasos por entrenamiento |
| Inner-LBFGS-20 | L-BFGS, lr=1, hasta 20 iteraciones, strong Wolfe, memoria 10 |

L-BFGS interno usa `max_eval=40` nominal, `tolerance_grad=1e-7` y
`tolerance_change=1e-9`. El presupuesto nominal de evaluaciones no es un
limite duro dentro de cada busqueda de linea de PyTorch; se registraron
las evaluaciones reales. Se crea un optimizador nuevo en cada entrenamiento,
sin reutilizar su memoria de curvatura. Adam ya reiniciaba sus momentos
de la misma forma.

La implementacion comparte la misma construccion de perdida del metodo
original mediante un hook de entrenamiento. Adam sigue siendo el
comportamiento predeterminado. Los historiales incorporan contadores
adicionales; la perdida historica de Adam conserva su convencion anterior
(antes del ultimo paso), mientras que L-BFGS registra una evaluacion final.
Por ello no se comparan esos campos como si fueran identicos.

## Tiempo a igual calidad

La referencia es el resultado final del L-BFGS convencional a 100 solves:

- misfit J <= **0,00184119**;
- error relativo RMS enmascarado de velocidad <= **0,33512%**.

Se exige cumplir ambos en el mismo modelo aceptado. El error de modelo
es un diagnostico sintetico que requiere conocer la verdad; no es un
criterio de parada directamente disponible en campo.

| Metodo | Solves al cumplir ambos | Tiempo (s) | Ahorro de tiempo vs convencional | Error modelo al cruce (%) |
|---|---:|---:|---:|---:|
| L-BFGS convencional | 100 | 46,96 | referencia | 0,33512 |
| Adam-200 | 71 | 51,02 | -8,6% | 0,33100 |
| Adam-50 | 77 | 48,11 | -2,4% | 0,33064 |
| **L-BFGS interno** | **79** | **45,30** | **3,5%** | 0,33001 |

Ahorro negativo significa mayor tiempo. Adam-50 reduce 5,7% el tiempo
al cruce respecto a Adam-200. L-BFGS interno lo reduce 11,2%.

El ahorro de solves frente al convencional es 29%, 23% y 21%,
respectivamente. No equivale al ahorro de tiempo.

Si se exige solamente el umbral de misfit:

| Metodo | Solves | Tiempo (s) |
|---|---:|---:|
| L-BFGS convencional | 100 | 46,96 |
| Adam-200 | 51 | 35,03 |
| Adam-50 | 57 | 35,82 |
| L-BFGS interno | 53 | 31,83 |

En esos cruces, los errores de modelo de las variantes aprendidas
todavia exceden el objetivo; no se presentan como igualdad de calidad
conjunta.

## Costo del entrenamiento y calidad final

Son totales hasta consumir 100 solves, no costos hasta el primer cruce.

| Metodo | Entrenamientos | Iteraciones internas | Evaluaciones gradiente | Tiempo entrenamiento (s) |
|---|---:|---:|---:|---:|
| Adam-200 | 49 | 9.800 | 9.800 | 20,19 |
| Adam-50 | 49 | 2.450 | 2.450 | 4,95 |
| L-BFGS interno | 49 | 980 | 1.171 | 5,23 |

L-BFGS realizo entre **22 y 29 evaluaciones de gradiente por entrenamiento**
(23,9 en promedio). Las 20 iteraciones no equivalen a 20 evaluaciones.
Ademas hizo una evaluacion final sin gradiente por entrenamiento:
**1.220 evaluaciones de perdida** en total.

| Metodo | J final | Error modelo final (%) | Tiempo total (s) |
|---|---:|---:|---:|
| L-BFGS convencional | 0,00184119 | 0,33512 | 46,98 |
| Adam-200 | **0,00029396** | **0,28724** | 80,76 |
| Adam-50 | 0,00043568 | 0,29605 | 65,13 |
| L-BFGS interno | 0,00031665 | 0,29063 | 57,70 |

Adam-200 sigue obteniendo el mejor resultado final a igual presupuesto de
solves, pero no el mejor tiempo al umbral de calidad usado en esta prueba.
L-BFGS interno no fue el de entrenamiento mas barato (Adam-50 lo fue);
su ventaja observada es en la trayectoria completa hasta el umbral.

## Medicion, auditoria y limites

Se reutilizo la medicion sincronizada por forward y por componente.
Los tiempos al cruce incluyen construccion del precondicionador,
entrenamiento, aplicaciones, forwards, adjuntos y busquedas previas.
No incluyen adjunto/entrenamiento posteriores al forward que ya cumple
la calidad. Setup de observaciones, un forward de calentamiento y
serializacion quedan fuera del tiempo de inversion.

La instrumentacion introduce overhead de sincronizacion y copias para
auditoria en todos los metodos. El orden fijo y la falta de repeticiones
permiten variacion termica, de reloj y de carga: **una ventaja de 1,66
segundos frente al convencional no establece significancia estadistica**.

Se auditaron presupuestos exactos, hashes de fuentes, tiempos crecientes,
contadores de evaluaciones y recalculo de los cruces. **11 forwards
adicionales** comprobaron los modelos finales y modelos de cruce guardados;
los objetivos y errores coinciden dentro de las tolerancias verificadas.
Esos forwards de auditoria no influyeron en el entrenamiento.

Las pruebas verifican equivalencia del Adam refactorizado con su bucle
original dentro de tolerancias float32, positividad y conteo de closures
del L-BFGS interno, e integracion con el solver real. La comprobacion
inicial de igualdad bit a bit de Adam detecto una diferencia maxima
de 7,45e-9 en el mapa, compatible con variacion numerica CUDA; se usa
atol=1e-7, rtol=1e-5, no una exigencia de igualdad binaria.

## Reproducir y siguiente decision

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_inner_optimizers.py --output results/inner-optimizers-replica
python -m unittest -v test_inner_lbfgs test_quality_cost test_direct_precond test_elm_surrogate
```

Se utilizo `/home/orincon/miniconda3/envs/ml-seismic-waves-env/bin/python`.
El runner rechaza sobrescribir resultados.

Los modelos, historial interno, evaluaciones, tiempos y auditoria quedan
en [results/inner-optimizers/s2](./results/inner-optimizers/s2/).

**Recomendacion:** mantener L-BFGS interno como candidato prometedor,
sin convertirlo todavia en predeterminado. La siguiente validacion seria
repetir tiempos con orden balanceado y comprobar el escenario 1, solo
si se desea ampliar este piloto.
