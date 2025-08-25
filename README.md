# 🎵 AnalisisDeOndas - Registro y Análisis de Ondas Sonoras
<img src="\img\ParaPDF.png "/>
Una aplicación de escritorio desarrollada en Python para el registro, análisis y visualización de ondas sonoras. Diseñada especialmente para el estudio de física de ondas y acústica.

## Características

###  Grabación de Audio
- Grabación de alta calidad (44.1 kHz, 32-bit float)
- Duración configurable de grabación (1-10 segundos)
- Reproducción del audio grabado
- Interfaz intuitiva con controles de grabación

###  Análisis Temporal
- **Detección automática del período** mediante autocorrelación
- Cálculo de frecuencia temporal (f = 1/T)
- Cálculo de frecuencia angular (ω = 2πf)
- Visualización de la oscilación completa
- Vista ampliada para medición precisa del período

###  Análisis Espectral
- **Transformada Rápida de Fourier (FFT)** para análisis de frecuencias
- Detección de frecuencia fundamental
- Identificación de armónicos principales
- Reconocimiento automático de notas musicales
- Visualización del espectro de frecuencias

###  Reconocimiento Musical
- Detección automática de notas musicales en español (Do, Re, Mi, etc.)
- Análisis de 8 octavas por nota musical
- Identificación de armónicos y su relación con la fundamental
- Precisión de ±50 cents para detección de notas

###  Visualización
- **Tres gráficas simultáneas**:
  1. Oscilación completa en el tiempo
  2. Vista ampliada para medición del período
  3. Espectro de frecuencias (FFT)
- Interfaz oscura optimizada para análisis
- Marcadores automáticos de períodos y frecuencias
- Exportación de gráficas en alta resolución


##  Instalación
**Importante:** Asegurate de tener git instalado en tu sistema o descarga el archivo .zip desde github.

1. **Clona o descarga el repositorio**:
```bash
git clone https://github.com/tu-usuario/audio-spectrometer.git
cd audio-spectrometer
```

2. **Instala las dependencias**:
```bash
pip install -r requirements.txt
```

3. **Ejecuta la aplicación**:
```bash
python audio_spectrometer.py
```

## Uso de la Aplicación

### Grabación y Análisis Básico

1. **Configura la duración** de grabación (1-10 segundos)
2. **Presiona "Grabar Sonido"** y produce el sonido que deseas analizar
3. **Presiona "Analizar Grabación"** para procesar los datos
4. **Observa los resultados** en las gráficas y paneles de información

### Funcionalidades Avanzadas

- **Reproducir Audio**: Escucha la grabación realizada
- **Guardar Gráficas**: Exporta las visualizaciones en PNG o PDF
- **Análisis Temporal**: Observa el período y frecuencia de la onda
- **Análisis Espectral**: Identifica frecuencias dominantes y armónicos

## Interpretación de Resultados

### Panel de Análisis Temporal
- **Período (T)**: Tiempo que tarda la onda en completar un ciclo
- **Frecuencia (f)**: Número de ciclos por segundo (Hz)
- **Frecuencia angular (ω)**: Frecuencia en radianes por segundo

### Panel de Análisis Espectral
- **Frecuencia fundamental**: Frecuencia principal del sonido
- **Armónicos principales**: Múltiplos de la frecuencia fundamental
- **Nota musical**: Nota musical más cercana a la fundamental

### Gráficas
- **Gráfica 1**: Visualiza toda la grabación para observar patrones
- **Gráfica 2**: Vista ampliada con marcadores de período
- **Gráfica 3**: Espectro de frecuencias con picos marcados


##  Desarrollo y Personalización

### Estructura del Código
```
audio_spectrometer.py
├── AudioSpectrometer (clase principal)
├── setup_gui() (interfaz gráfica)
├── analyze_temporal_properties() (análisis temporal)
├── analyze_spectral_properties() (análisis espectral)
└── frequency_to_note() (reconocimiento musical)
```

### Parámetros Configurables
- **CHUNK**: Tamaño del buffer de audio (4096)
- **RATE**: Frecuencia de muestreo (44100 Hz)
- **FORMAT**: Formato de audio (paFloat32)
- Rango de frecuencias musicales (80-2000 Hz)

## ⚠️ Solución de Problemas

### Error de PyAudio
```
OSError: [Errno -9996] Invalid input device
```
**Solución**: Verifica que tu micrófono esté conectado y funcionando.

### Error de Dependencias
```
ModuleNotFoundError: No module named 'pyaudio'
```
**Solución**: Instala PyAudio siguiendo las instrucciones específicas de tu sistema operativo.

### Audio Distorsionado
- Reduce el volumen de entrada del micrófono
- Asegúrate de que no haya ruido de fondo excesivo
- Verifica que el micrófono esté funcionando correctamente

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

##  Contacto

Para preguntas, sugerencias o reportes de bugs, puedes contactar a través de:
- Email: tu-email@ejemplo.com
- Issues: [GitHub Issues](https://github.com/tu-usuario/audio-spectrometer/issues)

##  Agradecimientos

- **NumPy** - Computación científica
- **Matplotlib** - Visualización de datos
- **PyAudio** - Interfaz de audio
- **Tkinter** - Interfaz gráfica de usuario

---

*Desarrollado con ❤️ para el estudio de la física de ondas y la acústica*
