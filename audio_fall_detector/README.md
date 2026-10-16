# Detector acústico migrado

Procede de `C:\proyects\Modulo-imagen-vid\audio_fall_detector`. El código de
origen y el modelo se conservaron allí. Aquí se reutilizan su disparador RMS,
características espectrales de 2048 dimensiones, MLP calibrado de seis clases,
evaluación posterior de voz/silencio y reglas de decisión.

`streaming.py` es el adaptador usado por Flask. Espera audio posterior real,
usa la entrada cruda y la alineación con las que se entrenó la cabeza original,
aplica las reglas originales y publica únicamente posibles caídas. No guarda
grabaciones ni descarga PANNs. El clasificador heurístico heredado no se utiliza:
si el modelo falta o falla, el análisis informa un error y el video continúa.

El artefacto registra scikit-learn 1.9.1 y serialización NumPy 2.
`model_compat.py` permite cargarlo con NumPy 1.26 sin modificar sus pesos ni
parchear NumPy globalmente. Consulte las pruebas de compatibilidad.

SHA-256 de `models/fall_classifier.joblib`:
`68527b88f75142693c56ee798f981611febb5d25fd46316e6c799aca7e2eb9da`.

Las pruebas son sintéticas. No establecen precisión clínica, calibración en
viviendas reales ni una tasa de falsas alarmas por hora. Los índices internos
no son una confianza válida de que una persona haya caído.

Consulte `../INFORME_INTEGRACION_DETECCION_CAIDAS_AUDIO.md` para la arquitectura,
los comandos de ejecución, los cambios de parámetros y las limitaciones.
