# Ajuste directo no negativo: resultado del experimento

Fecha: 2026-10-26.

## Conclusion

**El ajuste directo abarato mucho el entrenamiento, pero no mejoro el tiempo
para alcanzar la misma calidad.** No alcanzo la calidad final de L-BFGS
convencional dentro de 100 simulaciones en ninguna de las seis inversiones
evaluadas. Por tanto, no hay un factor de aceleracion a igual calidad que
reportar para esta variante.

La hipotesis tenia dos partes, con resultados distintos:

1. **Entrenar mas barato:** confirmado. El ajuste consumio aproximadamente
   0,44-0,46 segundos por inversion completa, frente a unos 20 segundos del
   precondicionador acotado con Adam: **97,7%-97,8% menos tiempo de ajuste**.
2. **Conservar suficiente calidad de las actualizaciones:** no confirmado.
   El misfit final fue mayor que el de L-BFGS convencional y mucho mayor
   que el del precondicionador acotado.

No debe confundirse reducir el entrenamiento con acelerar la inversion.

## Metodo y controles

Se implemento [direct_precond.py](./direct_precond.py), sin cambiar el
optimizador ni los metodos existentes:

```text
P(q) = gamma [0.1 q + sum_l theta_l B_l B_l^T q], theta_l >= 0.
```

Los cuatro operadores son identidad y convoluciones gaussianas a escalas
1, 2 y 4 celdas. Las convoluciones y sus transpuestas usan el mismo dominio
y tratamiento de borde. La identidad positiva garantiza que P sea definida
positiva cuando gamma es positivo.

Se ajustan cuatro pesos, con el mismo tipo de error secante relativo,
normalizacion y ponderacion temporal del metodo aprendido existente:

```text
min_theta sum_i w_i ||P(y_i) - s_i||^2 / ||s_i||^2
          + 0.01 ||theta - [0.9, 0, 0, 0]||^2,
subject to theta >= 0.
```

Los pesos temporales se normalizan para sumar uno; los pares recientes
reciben hasta cuatro veces el peso de los antiguos. Gamma se calcula con
el ultimo par, como en el metodo existente. Se utiliza todo el historial
disponible de pares admitidos por L-BFGS.

Se forma una matriz de Gram 4 x 4 y un RHS de cuatro elementos; se resuelve
NNLS sobre un factor de Cholesky con ridge. **Es un ajuste directo pequeno,
no una solucion analitica cerrada**: NNLS sigue siendo un algoritmo
restringido iterativo, pero no entrena una red durante 200 epocas.
El calculo del Gram se hace en float64 y solo los arreglos pequenos pasan
a CPU para el ajuste.

Los cuatro metodos comparados fueron:

- **L-BFGS:** convencional.
- **Bounded:** precondicionador existente, `illum_bound=3`, con Adam.
- **Direct:** filtros fijos y cuatro pesos aprendidos con NNLS.
- **Fixed:** los mismos filtros con pesos iguales que suman 0,9, sin
  aprendizaje de pesos. Conserva la actualizacion secante de gamma.

El metodo directo cambia la familia de operadores, no solamente el
optimizador de entrenamiento: no aprende mapa espacial de iluminacion ni
kernel. El experimento no demuestra que un ajuste directo de una familia
mas expresiva vaya a fallar.

## Protocolo

Se ejecutaron **24 inversiones nuevas**: cuatro metodos, dos escenarios
estandar y tres repeticiones. Cada inversion recibio exactamente
100 simulaciones equivalentes, contando forwards rechazados y adjuntos.
No se usaron Born, datos offline ni simulaciones gruesas.

Se rotaron los ordenes de ejecucion entre repeticiones. Los parametros se
fijaron antes del benchmark y no se modificaron tras observar resultados.

Las ejecuciones fueron secuenciales en la misma NVIDIA RTX 2000 Ada
Generation Laptop GPU, con PyTorch 2.8.0+cu128 y un hilo CPU.
Se restablecieron semillas antes de cada inversion, sin imponer kernels
CUDA estrictamente deterministas. Las repeticiones miden variacion temporal
y numerica del mismo problema, no generalizacion a nuevas geologias.

Se mantuvo el problema acustico 2D, inicio suave, adquisicion con superficie
y pozos y ausencia de ruido. **No se ejecuto SPECFEM3D.**

## Tiempo para igualar simultaneamente misfit y error de modelo

En cada repeticion los umbrales son el misfit y el error de velocidad finales
del L-BFGS convencional a 100 solves. Se exige que ambos se cumplan en el
mismo modelo aceptado. Se compara el primer cruce de cada metodo, no una
interpolacion ni un ensayo rechazado.

Los tiempos siguientes son medias de las tres repeticiones.

| Escenario | Metodo | Cumple ambos | Solves hasta el cruce | Tiempo al cruce (s) |
|---|---|---:|---:|---:|
| 1 | L-BFGS | 3/3 | 98 | 49,39 |
| 1 | Bounded | 3/3 | 33 | **21,98** |
| 1 | Direct | **0/3** | No alcanza en 100 | No disponible |
| 1 | Fixed | 0/3 | No alcanza en 100 | No disponible |
| 2 | L-BFGS | 3/3 | 100 | **57,19** |
| 2 | Bounded | 3/3 | 71-73 | 58,36 |
| 2 | Direct | **0/3** | No alcanza en 100 | No disponible |
| 2 | Fixed | 0/3 | No alcanza en 100 | No disponible |

Direct y Fixed tampoco alcanzaron el umbral de misfit por separado en
ninguna repeticion. Sus tiempos hasta el umbral estan censurados por el
presupuesto: no se afirma que nunca vayan a alcanzarlo, solo que no lo
hicieron dentro de 100 solves.

En esta nueva medicion Bounded ahorro 55,1% de tiempo en el escenario 1,
y fue 2,0% mas lento en promedio en el escenario 2. En este ultimo el rango
fue de 5,0% mas rapido a 5,6% mas lento. La variacion refuerza que no hay una
aceleracion temporal robusta de Bounded en el escenario 2 a igual calidad
conjunta. No se sustituyen las mediciones anteriores por el resultado mas
favorable: esta comparacion incluye nuevas ejecuciones y sincronizacion
adicional para medir componentes.

## Resultados finales a 100 simulaciones

El error de velocidad es 100 veces el RMS relativo enmascarado.
Los tiempos totales no representan tiempos a igual calidad.

| Escenario | Metodo | Misfit J medio | Error velocidad (%) | Tiempo total (s) | Ajuste online (s) |
|---|---|---:|---:|---:|---:|
| 1 | L-BFGS | 0,00254849 | 0,69051 | 50,46 | 0 |
| 1 | Bounded | **0,00004552** | **0,50258** | 81,23 | 20,068 |
| 1 | Direct | 0,00339404 | 0,71435 | 53,48 | **0,435** |
| 1 | Fixed | 0,00355017 | 0,72279 | 53,80 | 0,013 |
| 2 | L-BFGS | 0,00183104 | 0,33443 | 57,20 | 0 |
| 2 | Bounded | **0,00030526** | **0,28852** | 83,64 | 20,236 |
| 2 | Direct | 0,00335297 | 0,38821 | 57,75 | **0,456** |
| 2 | Fixed | 0,00422884 | 0,40280 | 57,90 | 0,013 |

Direct mejoro el misfit medio frente a Fixed en 4,4% y 20,7%,
respectivamente. Pero su misfit fue **33,2% y 83,1% mayor que L-BFGS**.
El aprendizaje aporta algo frente a este control, no una mejora practica
frente a los baselines mas fuertes.

## Diagnostico observado

**Los pesos de las tres componentes gaussianas quedaron en cero en todas
las actualizaciones Direct: 147/147 por escenario.** El ajuste eligio
efectivamente una identidad reescalada.

Esto explica la forma del operador aprendido, pero no prueba por si solo
la causa completa del peor rendimiento. Los operadores suaves disponibles
no fueron seleccionados por este objetivo y estos pares. No hay evidencia
para concluir que bastaria cambiar el ridge o agregar mas escalas.

Una limitacion importante de la propuesta es que una suma positiva de
filtros gaussianos no reproduce todos los kernels o mapas espaciales
permitidos por el precondicionador anterior. La menor expresividad y el
ajuste sobre pares historicos son posibles factores, no causas aisladas
demostradas experimentalmente.

## Medicion y auditoria

[run_direct_experiments.py](./run_direct_experiments.py) reutiliza el
solver temporizado y la funcion de primer cruce del experimento previo.
Los tiempos de cruce se toman al terminar el forward verificador, con
sincronizacion CUDA, e incluyen toda la construccion, aplicacion,
entrenamiento, adjuntos y busquedas precedentes.

Los componentes de entrenamiento y aplicacion tambien se sincronizan
para medirlos. Esto introduce overhead; las comparaciones deben hacerse
dentro de este benchmark, no mezclando sus tiempos con otros archivos.
El tiempo de ajuste incluye construccion del problema, transferencia CPU,
NNLS y actualizacion de pesos, no solo la llamada al optimizador.

Se excluyen del tiempo la generacion de observaciones, un forward de
calentamiento por escenario y la serializacion. El adjunto y entrenamiento
posteriores al punto de cruce no son necesarios para terminar ahi y no se
atribuyen a ese punto. El overhead de guardar copias de modelos para
auditoria se aplica a todos los metodos.

Se auditaron las 24 ejecuciones: presupuesto exacto, hashes de codigo,
coherencia temporal, descenso de objetivos aceptados, no negatividad de
pesos y recalculo de los resultados de primer cruce.
**39 forwards adicionales** verificaron los modelos finales y todos los
modelos distintos guardados en los cruces, reproduciendo objetivos y
errores dentro de las tolerancias. Esos forwards no influyeron en la
optimizacion ni cuentan como entrenamiento.

Las pruebas focalizadas verifican equivalencia con NNLS sobre el diseno
completo, positividad y simetria, mejora del objetivo de ajuste, errores
explicitos, integracion con el solver y tratamiento de umbrales no alcanzados.
Pylance no encontro errores de sintaxis en los tres archivos nuevos.

## Recomendacion

No reemplazar el precondicionador acotado con esta variante. El siguiente
experimento mas conservador seria **abaratar el entrenamiento de la misma
familia de operadores**: menos epocas o actualizaciones menos frecuentes,
con la misma validacion a igual calidad. Eso preservaria la expresividad
que aqui se elimino. Esta recomendacion sigue sin probarse.

Los archivos y modelos estan en [results/direct-fit](./results/direct-fit/);
`summary.json` agrega resultados, los `comparisons.json` conservan los
cruces y los `audit.json` registran la verificacion.

Reproducir en el entorno CUDA existente:

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_direct_experiments.py --repeats 3 --output results/direct-fit-replica
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_direct_experiments.py --repeats 3 --output results/direct-fit-replica
python -m unittest -v test_direct_precond test_quality_cost test_elm_surrogate
```

El ejecutable usado fue
`/home/orincon/miniconda3/envs/ml-seismic-waves-env/bin/python`.
El runner rechaza sobrescribir un directorio de escenario con resultados.
