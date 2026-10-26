# Ahorro a igual calidad: L-BFGS con precondicionador aprendido acotado

Fecha: 2026-10-26.

## Resultado principal

**Si exigimos igualar tanto el ajuste de datos como el error del modelo:**

- **Escenario 1:** el acotado necesita 33 simulaciones frente a 98 de
  L-BFGS: **66,3% menos simulaciones y 51,9% menos tiempo**, aproximadamente
  **2,08 veces mas rapido**.
- **Escenario 2:** necesita 71-73 frente a 99-100: **27,8% menos simulaciones**,
  pero **no reduce el tiempo** en esta GPU y problema pequeno: resulta
  **4,8% mas lento en promedio**.

Si solo exigimos el mismo misfit de los sismogramas, el escenario 2 si
mejora: **48,0% menos simulaciones y 29,5% menos tiempo**, aproximadamente
1,42 veces mas rapido. Sin embargo, en ese punto su modelo de velocidad
todavia es menos preciso que el de L-BFGS.

Los porcentajes son medias de tres comparaciones pareadas, usando el primer
modelo aceptado que cumple el umbral para ambos metodos. No son estimaciones
a partir del misfit final ni extrapolaciones a SPECFEM3D.

## Que se comparo

Se uso el `L-BFGS+P-b3` ya existente, con `illum_bound=3`, sin cambiar sus
hiperparametros, estructura, entrenamiento o algoritmo de optimizacion.
Es el metodo con menor misfit archivado a 100 solves en ambos escenarios
estandar; no es necesariamente el de menor error de modelo en todas las
variantes y escenarios.

Cada metodo se ejecuto hasta 100 simulaciones equivalentes:

- un forward cuesta una unidad;
- un adjunto cuesta una unidad;
- los forwards rechazados por la busqueda de linea tambien cuentan;
- no hay productos Born, simulaciones gruesas ni entrenamiento offline.

Se hicieron **12 inversiones nuevas**: dos metodos por dos escenarios por
tres repeticiones. El orden fue convencional/acotado, acotado/convencional,
convencional/acotado. Todas corrieron secuencialmente en la misma NVIDIA
RTX 2000 Ada Generation Laptop GPU, con PyTorch 2.8.0+cu128.

El problema es el acustico 2D de 60 x 100 celdas, seis fuentes y 700 pasos
temporales. Se conservaron adquisicion con superficie y pozos, inicializacion
suavizada, mascara y ausencia de ruido. **No se ejecuto SPECFEM3D.**

## Definicion de igual calidad

En cada repeticion se fijaron dos umbrales a partir del resultado final
de L-BFGS a 100 simulaciones:

1. **Misfit:** J menor o igual al J final del L-BFGS pareado.
2. **Modelo:** error relativo RMS enmascarado de velocidad menor o igual al
   error final del L-BFGS pareado.
3. **Ambos:** las dos condiciones en el mismo modelo aceptado.

Solo se utilizaron modelos aceptados, no ensayos rechazados. Se busco el
primer cruce observado, sin interpolar entre iteraciones.

El error de modelo requiere conocer el modelo verdadero y solo sirve como
diagnostico sintetico; no seria un criterio de parada disponible en campo.

Para evitar contar trabajo innecesario del baseline, el resultado principal
compara **primer cruce contra primer cruce**. L-BFGS ya alcanza su calidad
final a 98 solves en el escenario 1 y a 99-100 en el escenario 2, aunque
la ejecucion completa consume 100.

## Costos medidos

Los tiempos son medias de las tres repeticiones. Los ahorros porcentuales
son medias de los cocientes pareados, no cocientes de los tiempos medios.

| Escenario y criterio | Solves L-BFGS | Solves acotado | Tiempo L-BFGS (s) | Tiempo acotado (s) | Ahorro solves | Ahorro tiempo |
|---|---:|---:|---:|---:|---:|---:|
| 1: misfit | 98 | 33 | 49,47 | 23,76 | 66,3% | 51,9% |
| 1: modelo | 98 | 33 | 49,47 | 23,76 | 66,3% | 51,9% |
| 1: ambos | 98 | 33 | 49,47 | 23,76 | 66,3% | 51,9% |
| 2: misfit | 99-100 | 51-53 | 53,19 | 37,51 | 48,0% | 29,5% |
| 2: modelo | 99-100 | 71-73 | 53,19 | 55,79 | 27,8% | **-4,8%** |
| 2: ambos | 99-100 | 71-73 | 53,19 | 55,79 | 27,8% | **-4,8%** |

Un ahorro negativo significa mayor tiempo.

### Variabilidad entre repeticiones

| Escenario y criterio | Ahorro de solves observado | Ahorro de tiempo observado |
|---|---:|---:|
| 1: ambos | 66,3% en las tres | 50,1%-54,2% |
| 2: misfit | 46,5%-49,0% | 27,4%-31,3% |
| 2: ambos | 26,3%-29,0% | -11,2% a -0,1% |

Las tres repeticiones del escenario 2 fueron mas lentas para igualar ambos
criterios, aunque una quedo practicamente empatada. Son repeticiones
temporales del mismo problema, no tres realizaciones geologicas
independientes ni intervalos de confianza.

### Si se compara contra las 100 simulaciones completas

Para conectar con la pregunta original, que tomaba como referencia todo
el presupuesto de L-BFGS:

| Escenario y criterio | Ahorro solves frente a 100 | Ahorro tiempo frente a ejecucion completa |
|---|---:|---:|
| 1: ambos | 67,0% | 52,8% |
| 2: misfit | 48,3% | 30,2% |
| 2: ambos | 28,3% | -3,8% |

La diferencia con la tabla principal es que aqui se incluye el trabajo
restante de L-BFGS despues de haber alcanzado su calidad final.

## Calidad efectiva en los cruces

Valores medios; el error de modelo se expresa en porcentaje.

| Escenario y punto | Misfit J | Error de velocidad (%) |
|---|---:|---:|
| 1: objetivo L-BFGS | 0,00254648 | 0,69041 |
| 1: acotado al cumplir ambos | 0,00213625 | 0,67586 |
| 2: objetivo L-BFGS | 0,00185641 | 0,33555 |
| 2: acotado al cumplir misfit | 0,00176117 | **0,44205** |
| 2: acotado al cumplir ambos | 0,00066333 | 0,33333 |

El escenario 2 ilustra por que no basta reportar solamente misfit:
al cruzar ese umbral el error de velocidad del acotado todavia es mayor.
Los cruces sobrepasan ligeramente los umbrales porque las iteraciones son
discretas; no se fuerza una igualdad numerica exacta.

Los umbrales finales de L-BFGS variaron ligeramente entre repeticiones:

- Escenario 1: J entre 0,00254258 y 0,00254863.
- Escenario 2: J entre 0,00184119 y 0,00186402.

Se restablecio la semilla antes de cada inversion, pero no se impuso
determinismo estricto de kernels CUDA. Las diferencias numericas observadas
no se ocultan ni se reemplazan por el resultado mas favorable.

## Medicion temporal y validacion

[run_quality_cost.py](./run_quality_cost.py) usa el argumento `solver`
existente de L-BFGS para registrar tiempos; no modifica el optimizador.
Cada tiempo de cruce se toma al terminar el forward correspondiente, con
sincronizacion CUDA. Incluye:

- construccion del precondicionador;
- todos los forwards y adjuntos precedentes;
- entrenamiento online y aplicacion del precondicionador;
- trabajo previo de la busqueda de linea, incluidos ensayos rechazados.

No incluye el adjunto ni el entrenamiento posteriores al modelo que ya
alcanzo la calidad objetivo: no hacen falta si la inversion termina ahi.
Tampoco incluye la pequena comprobacion CPU de aceptacion posterior al
forward. Se mantienen copias de los modelos para auditoria; ese pequeno
overhead de instrumentacion afecta a ambos metodos.

La generacion de observaciones y un forward de calentamiento por proceso
quedan fuera del presupuesto de inversion y del tiempo. La serializacion
tambien queda fuera del tiempo medido.

Se verificaron:

- los contadores exactos de 100 solves en las 12 inversiones;
- la correspondencia de eventos temporales con los modelos aceptados;
- tiempos crecientes y consistentes con el tiempo total;
- hashes de las fuentes usadas;
- el recalculo exacto de las comparaciones guardadas;
- los modelos finales y todos los modelos distintos de cruce guardados,
  mediante **27 forwards de auditoria adicionales**, sin intervenir en la
  inversion.

Los forwards de auditoria reprodujeron los objetivos y errores almacenados
dentro de las tolerancias verificadas. Sus costos no se atribuyen a la
optimizacion. Cuatro pruebas focalizadas cubren la primera llegada, el
criterio conjunto, los casos sin cruce y que la instrumentacion no cambie
el resultado del solver. Las comprobaciones de sintaxis Pylance pasaron.

Los archivos en [results/quality-cost](./results/quality-cost/) contienen
modelos finales y de cruce, historiales aceptados, tiempos por forward,
comparaciones por repeticion, resumen y auditorias.

## Interpretacion

**La evidencia mas favorable es el escenario 1:** hay una reduccion real
de aproximadamente la mitad del tiempo para igualar simultaneamente ajuste
de datos y calidad del modelo.

**La evidencia no es uniforme:** en el escenario 2 se ahorran solves, pero
el costo del metodo aprendido compensa ese ahorro cuando exigimos la misma
precision del modelo. La medicion temporal incluye el entrenamiento; no
debe anunciarse una aceleracion temporal en ese criterio.

En una inversion 3D con solves mucho mas caros, reducir su numero podria
ser mas valioso, pero estos datos no cuantifican el ahorro neto en
SPECFEM3D. El costo de entrenar y aplicar un precondicionador 3D tambien
cambia. El siguiente paso razonable es reducir el overhead de entrenamiento
y probar un caso 3D controlado, no extrapolar directamente el factor 2,08.

## Reproducir

Se utilizo el interprete existente
`/home/orincon/miniconda3/envs/ml-seismic-waves-env/bin/python`.

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_quality_cost.py --repeats 3 --output results/quality-cost-replica
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_quality_cost.py --repeats 3 --output results/quality-cost-replica
python -m unittest -v test_quality_cost
```

Los comandos se ejecutan secuencialmente para no competir por la GPU.
El runner rechaza directorios de escenario que ya contengan resultados.
