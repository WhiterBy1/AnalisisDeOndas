import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import pyaudio
import threading
from collections import deque
import math
import wave
from datetime import datetime

class AudioSpectrometer:
    def __init__(self):
        # Configuración de audio
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Buffer para datos de audio
        self.audio_buffer = deque(maxlen=self.CHUNK * 4)
        self.recorded_data = []  # Para guardar la grabación completa
        
        # Variables de control
        self.is_recording = False
        self.recording_duration = 3.0  # Duración de grabación en segundos
        self.current_note = "---"
        self.current_frequency = 0.0
        self.current_amplitude = 0.0
        
        # Datos para análisis
        self.time_data = None
        self.freq_data = None
        self.magnitude_data = None
        self.sample_times = None
        
        # Variables para zoom interactivo
        self.zoom_start = 0.0
        self.zoom_end = 0.05
        self.rectangle_selector = None
        self.zoom_rectangle = None
        self.interactive_zoom_enabled = False
        self.is_zoom_analysis = False
        self.selection_start_x = None
        self.selection_active = False
        
        # Variables para análisis original
        self.original_time_data = None
        self.original_sample_times = None
        self.original_analysis = {}
        
        # Diccionario de notas musicales con nombres en español
        self.notes = {
            'C': [16.35, 32.70, 65.41, 130.81, 261.63, 523.25, 1046.50, 2093.00],
            'C#': [17.32, 34.65, 69.30, 138.59, 277.18, 554.37, 1108.73, 2217.46],
            'D': [18.35, 36.71, 73.42, 146.83, 293.66, 587.33, 1174.66, 2349.32],
            'D#': [19.45, 38.89, 77.78, 155.56, 311.13, 622.25, 1244.51, 2489.02],
            'E': [20.60, 41.20, 82.41, 164.81, 329.63, 659.25, 1318.51, 2637.02],
            'F': [21.83, 43.65, 87.31, 174.61, 349.23, 698.46, 1396.91, 2793.83],
            'F#': [23.12, 46.25, 92.50, 185.00, 369.99, 739.99, 1479.98, 2959.96],
            'G': [24.50, 49.00, 98.00, 196.00, 392.00, 783.99, 1567.98, 3135.96],
            'G#': [25.96, 51.91, 103.83, 207.65, 415.30, 830.61, 1661.22, 3322.44],
            'A': [27.50, 55.00, 110.00, 220.00, 440.00, 880.00, 1760.00, 3520.00],
            'A#': [29.14, 58.27, 116.54, 233.08, 466.16, 932.33, 1864.66, 3729.31],
            'B': [30.87, 61.74, 123.47, 246.94, 493.88, 987.77, 1975.53, 3951.07]
        }
        
        # Mapeo de notas inglesas a españolas
        self.note_names_spanish = {
            'C': 'Do', 'C#': 'Do#', 'D': 'Re', 'D#': 'Re#',
            'E': 'Mi', 'F': 'Fa', 'F#': 'Fa#', 'G': 'Sol',
            'G#': 'Sol#', 'A': 'La', 'A#': 'La#', 'B': 'Si'
        }
        
        # Crear la interfaz
        self.setup_gui()
        
        # Configurar PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Registro de Ondas Sonoras y Análisis de Espectro - Física Calor y Ondas")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de control
        control_frame = ttk.LabelFrame(main_frame, text="Control de Grabación", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Primera fila de controles
        control_row1 = ttk.Frame(control_frame)
        control_row1.pack(fill=tk.X, pady=(0, 5))
        
        self.record_button = ttk.Button(control_row1, text="Grabar Sonido", command=self.start_recording)
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_row1, text="Detener", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_button = ttk.Button(control_row1, text="Analizar Grabación", command=self.analyze_recording, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.play_button = ttk.Button(control_row1, text="▶ Reproducir Audio", command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Segunda fila de controles
        control_row2 = ttk.Frame(control_frame)
        control_row2.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(control_row2, text="Duración de grabación (s):").pack(side=tk.LEFT, padx=(0, 5))
        self.duration_var = tk.DoubleVar(value=3.0)
        duration_spinbox = ttk.Spinbox(control_row2, from_=1.0, to=10.0, increment=0.5, 
                                      textvariable=self.duration_var, width=10)
        duration_spinbox.pack(side=tk.LEFT, padx=(0, 20))
        
        self.save_button = ttk.Button(control_row2, text="Guardar Gráficas", command=self.save_plots, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Tercera fila - Controles de zoom interactivo
        control_row3 = ttk.Frame(control_frame)
        control_row3.pack(fill=tk.X, pady=(5, 0))
        
        self.enable_zoom_button = ttk.Button(control_row3, text="🔍 Activar Zoom Interactivo", 
                                            command=self.toggle_interactive_zoom, state=tk.DISABLED)
        self.enable_zoom_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_zoom_button = ttk.Button(control_row3, text="↻ Reset a Audio Completo", 
                                           command=self.reset_zoom_view, state=tk.DISABLED)
        self.reset_zoom_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Mostrar información de la selección
        self.zoom_info_label = tk.Label(control_row3, text="Selección: No activa", 
                                       font=("Arial", 10), bg='#95a5a6', fg='black')
        self.zoom_info_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Frame de información y resultados
        info_frame = ttk.LabelFrame(main_frame, text="Resultados del Análisis", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Crear dos columnas para la información
        info_left = ttk.Frame(info_frame)
        info_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        info_right = ttk.Frame(info_frame)
        info_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Información de mediciones temporales
        time_frame = ttk.LabelFrame(info_left, text="Análisis Temporal", padding=5)
        time_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.period_label = tk.Label(time_frame, text="Período (T): -- s", 
                                    font=("Arial", 10), bg='#ecf0f1')
        self.period_label.pack(anchor='w')
        
        self.frequency_label = tk.Label(time_frame, text="Frecuencia (f): -- Hz", 
                                       font=("Arial", 10), bg='#ecf0f1')
        self.frequency_label.pack(anchor='w')
        
        self.angular_freq_label = tk.Label(time_frame, text="Frecuencia angular (ω): -- rad/s", 
                                          font=("Arial", 10), bg='#ecf0f1')
        self.angular_freq_label.pack(anchor='w')
        
        # Información de análisis espectral
        freq_frame = ttk.LabelFrame(info_right, text="Análisis Espectral", padding=5)
        freq_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.fundamental_label = tk.Label(freq_frame, text="Frecuencia fundamental: -- Hz", 
                                         font=("Arial", 10), bg='#ecf0f1')
        self.fundamental_label.pack(anchor='w')
        
        self.harmonics_label = tk.Label(freq_frame, text="Armónicos principales: --", 
                                       font=("Arial", 10), bg='#ecf0f1')
        self.harmonics_label.pack(anchor='w')
        
        self.note_detected_label = tk.Label(freq_frame, text="Nota musical: --", 
                                           font=("Arial", 12, "bold"), bg='#ecf0f1')
        self.note_detected_label.pack(anchor='w')
        
        # Status bar
        self.status_label = tk.Label(main_frame, text="Listo para grabar", 
                                    font=("Arial", 10), bg='#34495e', fg='white')
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Crear las gráficas
        self.setup_plots(main_frame)
        
    def setup_plots(self, parent):
        # Frame para las gráficas
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear figura con 3 subplots con mayor tamaño
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(16, 14))
        self.fig.patch.set_facecolor('#2c3e50')
        
        # Configurar subplot 1 - Oscilación completa en el tiempo
        self.ax1.set_facecolor('#34495e')
        self.ax1.set_title('Oscilación de la Onda Sonora en el Tiempo - Selecciona región para zoom', color='white', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Tiempo (s)', color='white', fontsize=12)
        self.ax1.set_ylabel('Amplitud', color='white', fontsize=12)
        self.ax1.tick_params(colors='white', labelsize=10)
        self.ax1.grid(True, alpha=0.4, linewidth=0.8)
        
        # Configurar subplot 2 - Zoom para medir período
        self.ax2.set_facecolor('#34495e')
        self.ax2.set_title('Zoom - Medición del Período', color='white', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Tiempo (s)', color='white', fontsize=12)
        self.ax2.set_ylabel('Amplitud', color='white', fontsize=12)
        self.ax2.tick_params(colors='white', labelsize=10)
        self.ax2.grid(True, alpha=0.4, linewidth=0.8)
        
        # Configurar subplot 3 - Espectro de frecuencias
        self.ax3.set_facecolor('#34495e')
        self.ax3.set_title('Espectro de Frecuencias (FFT)', color='white', fontsize=14, fontweight='bold')
        self.ax3.set_xlabel('Frecuencia (Hz)', color='white', fontsize=12)
        self.ax3.set_ylabel('Magnitud', color='white', fontsize=12)
        self.ax3.tick_params(colors='white', labelsize=10)
        self.ax3.grid(True, alpha=0.4, linewidth=0.8)
        
        plt.tight_layout()
        
        # Integrar matplotlib con tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def frequency_to_note(self, frequency):
        """Convierte una frecuencia a nota musical en español"""
        if frequency < 80 or frequency > 4000:
            return "---", -1
            
        min_diff = float('inf')
        closest_note = "---"
        octave = -1
        
        for note_name, frequencies in self.notes.items():
            for i, note_freq in enumerate(frequencies):
                diff = abs(frequency - note_freq)
                if diff < min_diff:
                    min_diff = diff
                    closest_note = note_name
                    octave = i
                    
        # Solo retornar la nota si está dentro de un margen razonable
        if octave >= 0:
            expected_freq = self.notes[closest_note][octave]
            cents_diff = 1200 * math.log2(frequency / expected_freq)
            
            if abs(cents_diff) < 50:
                spanish_note = self.note_names_spanish[closest_note]
                return f"{spanish_note}{octave}", octave
                
        return "---", -1
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback para grabar datos de audio"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.recorded_data.extend(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Inicia la grabación de audio"""
        try:
            self.recorded_data = []
            self.recording_duration = self.duration_var.get()
            
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            self.is_recording = True
            
            # Actualizar interfaz
            self.record_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.analyze_button.config(state=tk.DISABLED)
            self.status_label.config(text=f"Grabando... ({self.recording_duration}s)")
            
            # Programar detención automática
            self.root.after(int(self.recording_duration * 1000), self.stop_recording)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar la grabación: {str(e)}")
    
    def stop_recording(self):
        """Detiene la grabación"""
        if self.is_recording:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Actualizar interfaz
            self.record_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Grabación completada - {len(self.recorded_data)} samples")
    
    def analyze_recording(self):
        """Analiza la grabación completa"""
        if not self.recorded_data:
            messagebox.showwarning("Advertencia", "No hay datos grabados para analizar")
            return
        
        self.status_label.config(text="Analizando datos...")
        
        # Convertir a numpy array
        self.time_data = np.array(self.recorded_data)
        self.sample_times = np.linspace(0, len(self.time_data)/self.RATE, len(self.time_data))
        
        # Resetear estado de zoom
        self.is_zoom_analysis = False
        self.original_time_data = None
        
        # Análisis temporal - encontrar período
        self.analyze_temporal_properties()
        
        # Análisis espectral - FFT
        self.analyze_spectral_properties()
        
        # Actualizar gráficas
        self.update_plots()
        
        # Habilitar controles
        self.save_button.config(state=tk.NORMAL)
        self.enable_zoom_button.config(state=tk.NORMAL)
        self.reset_zoom_button.config(state=tk.NORMAL)
        
        self.status_label.config(text="Análisis completado - Zoom interactivo disponible")
    
    def analyze_temporal_properties(self):
        """Analiza las propiedades temporales con mejor robustez"""
        signal = self.time_data
        
        # Preprocesamiento mejorado
        if len(signal) > 100:
            # Filtro simple: restar media móvil
            window_size = min(100, len(signal) // 10)
            moving_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
            signal = signal - moving_avg
        
        # Normalizar solo si hay señal significativa
        signal_power = np.var(signal)
        if signal_power > 1e-8:  # Umbral de potencia mínima
            signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            # Señal muy débil, marcar como no detectada
            self.period = 0
            self.frequency_temporal = 0
            self.angular_frequency = 0
            return
        
        # Autocorrelación con rango ampliado
        correlation = np.correlate(signal, signal, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Rango ampliado para cubrir más frecuencias musicales
        min_period_samples = max(10, int(0.0005 * self.RATE))  # ~2000 Hz máximo
        max_period_samples = min(len(correlation)-1, int(0.1 * self.RATE))  # ~10 Hz mínimo
        
        if max_period_samples <= min_period_samples:
            self.period = 0
            self.frequency_temporal = 0
            self.angular_frequency = 0
            return
        
        # Búsqueda mejorada del período
        search_range = correlation[min_period_samples:max_period_samples]
        
        # Encontrar múltiples picos y elegir el mejor
        peaks = []
        threshold = 0.3 * np.max(search_range)  # Umbral dinámico
        
        for i in range(1, len(search_range) - 1):
            if (search_range[i] > search_range[i-1] and 
                search_range[i] > search_range[i+1] and 
                search_range[i] > threshold):
                peaks.append((i + min_period_samples, search_range[i]))
        
        if peaks:
            # Elegir el pico más prominente
            best_peak = max(peaks, key=lambda x: x[1])
            period_samples = best_peak[0]
            
            # Verificar que el período sea consistente
            confidence = best_peak[1] / np.max(search_range) if np.max(search_range) > 0 else 0
            
            if confidence > 0.5:  # Solo aceptar si la confianza es alta
                self.period = period_samples / self.RATE
                self.frequency_temporal = 1.0 / self.period
                self.angular_frequency = 2 * np.pi * self.frequency_temporal
                self.period_confidence = confidence
            else:
                self.period = 0
                self.frequency_temporal = 0
                self.angular_frequency = 0
        else:
            self.period = 0
            self.frequency_temporal = 0
            self.angular_frequency = 0
    
    def analyze_spectral_properties(self):
        """Analiza las propiedades espectrales usando FFT"""
        # Aplicar ventana para reducir efectos de borde
        windowed_data = self.time_data * np.hanning(len(self.time_data))
        
        # Calcular FFT
        fft = np.fft.rfft(windowed_data)
        self.magnitude_data = np.abs(fft)
        self.freq_data = np.fft.rfftfreq(len(windowed_data), 1/self.RATE)
        
        # Encontrar frecuencia fundamental (pico más alto en rango musical)
        music_range_mask = (self.freq_data >= 80) & (self.freq_data <= 2000)
        music_freqs = self.freq_data[music_range_mask]
        music_magnitude = self.magnitude_data[music_range_mask]
        
        if len(music_magnitude) > 0:
            fundamental_idx = np.argmax(music_magnitude)
            self.fundamental_frequency = music_freqs[fundamental_idx]
            
            # Encontrar armónicos (picos significativos múltiplos de la fundamental)
            self.harmonics = self.find_harmonics()
            
            # Determinar nota musical
            note, octave = self.frequency_to_note(self.fundamental_frequency)
            self.detected_note = note
        else:
            self.fundamental_frequency = 0
            self.harmonics = []
            self.detected_note = "---"
    
    def find_harmonics(self):
        """Encuentra los armónicos principales"""
        harmonics = []
        fundamental = self.fundamental_frequency
        
        if fundamental > 0:
            # Buscar hasta el 5to armónico
            for n in range(2, 6):
                harmonic_freq = n * fundamental
                if harmonic_freq > self.freq_data[-1]:
                    break
                
                # Buscar pico cerca de la frecuencia armónica
                tolerance = fundamental * 0.1  # 10% de tolerancia
                mask = (self.freq_data >= harmonic_freq - tolerance) & \
                       (self.freq_data <= harmonic_freq + tolerance)
                
                if np.any(mask):
                    harmonic_magnitudes = self.magnitude_data[mask]
                    harmonic_freqs = self.freq_data[mask]
                    
                    if len(harmonic_magnitudes) > 0:
                        max_idx = np.argmax(harmonic_magnitudes)
                        actual_freq = harmonic_freqs[max_idx]
                        magnitude = harmonic_magnitudes[max_idx]
                        
                        # Solo incluir si la magnitud es significativa
                        if magnitude > 0.1 * np.max(self.magnitude_data):
                            harmonics.append((n, actual_freq, magnitude))
        
        return harmonics
    
    # FUNCIONES DE ZOOM INTERACTIVO
    def toggle_interactive_zoom(self):
        """Activa o desactiva el zoom interactivo"""
        if not self.interactive_zoom_enabled:
            self.enable_interactive_selection()
            self.enable_zoom_button.config(text="❌ Desactivar Zoom")
            self.zoom_info_label.config(text="Zoom activo - Arrastra en la gráfica superior", bg='#2ecc71')
            self.interactive_zoom_enabled = True
        else:
            self.disable_interactive_selection()
            self.enable_zoom_button.config(text="🔍 Activar Zoom Interactivo")
            self.zoom_info_label.config(text="Selección: No activa", bg='#95a5a6')
            self.interactive_zoom_enabled = False
    
    def enable_interactive_selection(self):
        """Habilita la selección interactiva usando eventos de mouse directos"""
        # Variables para el drag
        self.selection_start_x = None
        self.selection_active = False
        
        # Conectar eventos del mouse
        self.press_cid = self.canvas.mpl_connect('button_press_event', self.on_selection_press)
        self.release_cid = self.canvas.mpl_connect('button_release_event', self.on_selection_release)
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_selection_motion)
    
    def disable_interactive_selection(self):
        """Deshabilita la selección interactiva"""
        # Desconectar eventos
        if hasattr(self, 'press_cid'):
            self.canvas.mpl_disconnect(self.press_cid)
        if hasattr(self, 'release_cid'):
            self.canvas.mpl_disconnect(self.release_cid)
        if hasattr(self, 'motion_cid'):
            self.canvas.mpl_disconnect(self.motion_cid)
        
        # Limpiar rectángulo de selección visual
        if self.zoom_rectangle:
            self.zoom_rectangle.remove()
            self.zoom_rectangle = None
            self.canvas.draw()
    
    def on_selection_press(self, event):
        """Inicia la selección"""
        if event.inaxes == self.ax1 and event.button == 1:  # Botón izquierdo en ax1
            self.selection_start_x = event.xdata
            self.selection_active = True
            self.zoom_info_label.config(text="Iniciando selección...", bg='#f39c12')
    
    def on_selection_motion(self, event):
        """Muestra la selección en progreso"""
        if (self.selection_active and event.inaxes == self.ax1 and 
            self.selection_start_x is not None and event.xdata is not None):
            
            # Limpiar rectángulo anterior si existe
            if self.zoom_rectangle:
                self.zoom_rectangle.remove()
            
            # Crear nuevo rectángulo
            x1, x2 = sorted([self.selection_start_x, event.xdata])
            y_min, y_max = self.ax1.get_ylim()
            
            self.zoom_rectangle = patches.Rectangle(
                (x1, y_min), x2 - x1, y_max - y_min,
                linewidth=2, edgecolor='red', facecolor='yellow', alpha=0.3
            )
            self.ax1.add_patch(self.zoom_rectangle)
            
            # Actualizar info
            duration = abs(x2 - x1)
            self.zoom_info_label.config(
                text=f"Seleccionando: {x1:.3f}s - {x2:.3f}s ({duration:.3f}s)",
                bg='#e67e22'
            )
            
            self.canvas.draw()
    
    def on_selection_release(self, event):
        """Completa la selección"""
        if (self.selection_active and event.inaxes == self.ax1 and 
            self.selection_start_x is not None and event.xdata is not None):
            
            # Finalizar selección
            x1, x2 = sorted([self.selection_start_x, event.xdata])
            
            # Validar selección mínima
            if abs(x2 - x1) < 0.001:  # Mínimo 1ms
                x2 = x1 + 0.001
            
            # Actualizar zoom
            self.zoom_start = x1
            self.zoom_end = x2
            
            duration = x2 - x1
            self.zoom_info_label.config(
                text=f"Analizando región: {x1:.3f}s - {x2:.3f}s ({duration:.3f}s)",
                bg='#3498db'
            )
            
            # Analizar solo la región seleccionada
            self.analyze_zoom_region(x1, x2)
            
            # Actualizar gráfica de zoom
            self.update_zoom_plot_interactive()
            
            self.selection_active = False
            self.selection_start_x = None
    
    def analyze_zoom_region(self, start_time, end_time):
        """Analiza solo la región seleccionada y actualiza todos los indicadores"""
        if self.time_data is None:
            return
        
        # Extraer datos de la región
        start_idx = int(start_time * self.RATE)
        end_idx = int(end_time * self.RATE)
        start_idx = max(0, start_idx)
        end_idx = min(len(self.time_data), end_idx)
        
        if end_idx <= start_idx:
            return
        
        # Datos de la región seleccionada
        region_data = self.time_data[start_idx:end_idx]
        region_times = self.sample_times[start_idx:end_idx]
        
        # Guardar datos originales completos si no lo hemos hecho
        if not hasattr(self, 'original_time_data') or self.original_time_data is None:
            self.original_time_data = self.time_data.copy()
            self.original_sample_times = self.sample_times.copy()
            self.original_analysis = {
                'period': getattr(self, 'period', 0),
                'frequency_temporal': getattr(self, 'frequency_temporal', 0),
                'angular_frequency': getattr(self, 'angular_frequency', 0),
                'fundamental_frequency': getattr(self, 'fundamental_frequency', 0),
                'harmonics': getattr(self, 'harmonics', []),
                'detected_note': getattr(self, 'detected_note', '---'),
                'freq_data': getattr(self, 'freq_data', None),
                'magnitude_data': getattr(self, 'magnitude_data', None)
            }
        
        # Analizar la región seleccionada
        self.analyze_region_temporal_properties(region_data)
        self.analyze_region_spectral_properties(region_data)
        
        # Actualizar información mostrada
        self.update_info_labels()
        
        # Marcar que estamos en modo zoom
        self.is_zoom_analysis = True
        self.zoom_region_duration = end_time - start_time
    
    def analyze_region_temporal_properties(self, region_data):
        """Analiza propiedades temporales de la región seleccionada"""
        if len(region_data) < 100:  # Región muy pequeña
            self.period = 0
            self.frequency_temporal = 0
            self.angular_frequency = 0
            return
        
        # Normalizar la región
        signal = region_data
        signal_power = np.var(signal)
        
        if signal_power > 1e-8:
            signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            self.period = 0
            self.frequency_temporal = 0
            self.angular_frequency = 0
            return
        
        # Autocorrelación para la región
        correlation = np.correlate(signal, signal, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Adaptar rango de búsqueda al tamaño de la región
        region_duration = len(region_data) / self.RATE
        min_period_samples = max(10, int(0.002 * self.RATE))
        max_period_samples = min(len(correlation)-1, int(region_duration * 0.25 * self.RATE))
        
        if max_period_samples <= min_period_samples:
            self.period = 0
            self.frequency_temporal = 0
            self.angular_frequency = 0
            return
        
        search_range = correlation[min_period_samples:max_period_samples]
        
        if len(search_range) > 0:
            # Buscar picos con umbral adaptativo
            peaks = []
            threshold = 0.2 * np.max(search_range)
            
            for i in range(1, len(search_range) - 1):
                if (search_range[i] > search_range[i-1] and 
                    search_range[i] > search_range[i+1] and 
                    search_range[i] > threshold):
                    peaks.append((i + min_period_samples, search_range[i]))
            
            if peaks:
                best_peak = max(peaks, key=lambda x: x[1])
                period_samples = best_peak[0]
                confidence = best_peak[1] / np.max(search_range) if np.max(search_range) > 0 else 0
                
                if confidence > 0.3:  # Umbral más bajo para regiones pequeñas
                    self.period = period_samples / self.RATE
                    self.frequency_temporal = 1.0 / self.period
                    self.angular_frequency = 2 * np.pi * self.frequency_temporal
                    self.period_confidence = confidence
                else:
                    self.period = 0
                    self.frequency_temporal = 0
                    self.angular_frequency = 0
            else:
                self.period = 0
                self.frequency_temporal = 0
                self.angular_frequency = 0
    
    def analyze_region_spectral_properties(self, region_data):
        """Analiza propiedades espectrales de la región seleccionada"""
        # Aplicar ventana para reducir efectos de borde
        windowed_data = region_data * np.hanning(len(region_data))
        
        # Calcular FFT para la región
        fft = np.fft.rfft(windowed_data)
        magnitude_data = np.abs(fft)
        freq_data = np.fft.rfftfreq(len(windowed_data), 1/self.RATE)
        
        # Actualizar datos espectrales para la región
        self.freq_data_region = freq_data
        self.magnitude_data_region = magnitude_data
        
        # Para la gráfica, usar datos de la región
        self.freq_data = freq_data
        self.magnitude_data = magnitude_data
        
        # Encontrar frecuencia fundamental en la región
        music_range_mask = (freq_data >= 80) & (freq_data <= 2000)
        music_freqs = freq_data[music_range_mask]
        music_magnitude = magnitude_data[music_range_mask]
        
        if len(music_magnitude) > 0 and np.max(music_magnitude) > np.max(magnitude_data) * 0.1:
            fundamental_idx = np.argmax(music_magnitude)
            self.fundamental_frequency = music_freqs[fundamental_idx]
            
            # Encontrar armónicos en la región
            self.harmonics = self.find_harmonics_in_region(freq_data, magnitude_data)
            
            # Determinar nota musical
            note, octave = self.frequency_to_note(self.fundamental_frequency)
            self.detected_note = note
        else:
            self.fundamental_frequency = 0
            self.harmonics = []
            self.detected_note = "---"
    
    def find_harmonics_in_region(self, freq_data, magnitude_data):
        """Encuentra armónicos en la región seleccionada"""
        harmonics = []
        fundamental = self.fundamental_frequency
        
        if fundamental > 0:
            for n in range(2, 6):
                harmonic_freq = n * fundamental
                if harmonic_freq > freq_data[-1]:
                    break
                
                tolerance = fundamental * 0.1
                mask = (freq_data >= harmonic_freq - tolerance) & \
                       (freq_data <= harmonic_freq + tolerance)
                
                if np.any(mask):
                    harmonic_magnitudes = magnitude_data[mask]
                    harmonic_freqs = freq_data[mask]
                    
                    if len(harmonic_magnitudes) > 0:
                        max_idx = np.argmax(harmonic_magnitudes)
                        actual_freq = harmonic_freqs[max_idx]
                        magnitude = harmonic_magnitudes[max_idx]
                        
                        # Umbral más flexible para regiones pequeñas
                        if magnitude > 0.05 * np.max(magnitude_data):
                            harmonics.append((n, actual_freq, magnitude))
        
        return harmonics
    
    def reset_to_full_analysis(self):
        """Restaura el análisis del audio completo"""
        if hasattr(self, 'original_time_data') and self.original_time_data is not None:
            # Restaurar datos originales
            self.time_data = self.original_time_data
            self.sample_times = self.original_sample_times
            
            # Restaurar análisis original
            for key, value in self.original_analysis.items():
                setattr(self, key, value)
            
            self.is_zoom_analysis = False
            
            # Actualizar información
            self.update_info_labels()
            self.zoom_info_label.config(text="Análisis completo restaurado", bg='#27ae60')
    
    def reset_zoom_view(self):
        """Resetea el zoom Y restaura análisis completo"""
        if self.time_data is None:
            return
        
        # Restaurar análisis completo
        self.reset_to_full_analysis()
        
        # Resetear valores de zoom visual
        self.zoom_start = 0.0
        total_duration = len(self.time_data) / self.RATE if hasattr(self, 'original_time_data') and self.original_time_data is not None else len(self.time_data) / self.RATE
        self.zoom_end = min(0.05, total_duration)
        
        # Actualizar gráfica con datos completos
        self.update_plots()
        
        # Limpiar selector visual
        if self.zoom_rectangle:
            self.zoom_rectangle.remove()
            self.zoom_rectangle = None
            self.canvas.draw()
    
    def update_zoom_plot_interactive(self):
        """Actualiza la segunda gráfica con la selección interactiva"""
        if self.time_data is None:
            return
        
        # Limpiar la segunda gráfica
        self.ax2.clear()
        self.ax2.set_facecolor('#34495e')
        self.ax2.tick_params(colors='white', labelsize=10)
        self.ax2.grid(True, alpha=0.4, linewidth=0.8)
        for spine in self.ax2.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1.2)
        
        # Usar datos originales para extraer la región visual
        data_source = self.original_time_data if hasattr(self, 'original_time_data') and self.original_time_data is not None else self.time_data
        times_source = self.original_sample_times if hasattr(self, 'original_sample_times') and self.original_sample_times is not None else self.sample_times
        
        # Calcular índices para la selección
        start_idx = int(self.zoom_start * self.RATE)
        end_idx = int(self.zoom_end * self.RATE)
        
        # Validar índices
        start_idx = max(0, start_idx)
        end_idx = min(len(data_source), end_idx)
        
        if end_idx <= start_idx:
            end_idx = start_idx + int(0.01 * self.RATE)  # Mínimo 10ms
        
        # Extraer datos del zoom
        zoom_times = times_source[start_idx:end_idx]
        zoom_data = data_source[start_idx:end_idx]
        
        if len(zoom_data) > 0:
            # Graficar la selección
            self.ax2.plot(zoom_times, zoom_data, 'lime', linewidth=2.5)
            region_indicator = "[REGIÓN]" if self.is_zoom_analysis else "[ZOOM]"
            self.ax2.set_title(
                f'{region_indicator} {self.zoom_start:.3f}s a {self.zoom_end:.3f}s', 
                color='white', fontsize=14, fontweight='bold'
            )
            self.ax2.set_xlabel('Tiempo (s)', color='white', fontsize=12)
            self.ax2.set_ylabel('Amplitud', color='white', fontsize=12)
            
            # Marcar períodos en la selección
            if hasattr(self, 'period') and self.period > 0:
                current_time = self.zoom_start
                period_count = 0
                
                while current_time < self.zoom_end and period_count < 20:  # Máximo 20 marcas
                    period_time = self.zoom_start + (period_count * self.period)
                    if period_time <= self.zoom_end:
                        self.ax2.axvline(x=period_time, color='red', linestyle='--', 
                                       linewidth=2, alpha=0.8)
                        
                        # Añadir etiqueta cada pocos períodos para no saturar
                        if period_count % 2 == 0:
                            self.ax2.text(period_time, np.max(zoom_data) * 0.9, 
                                        f'T={self.period:.4f}s', rotation=90, 
                                        color='red', fontsize=9, fontweight='bold')
                        
                        period_count += 1
                    else:
                        break
        
        # Actualizar también la tercera gráfica con el espectro de la región
        self.update_spectrum_plot()
        
        # Redibujar canvas
        self.canvas.draw()
    
    def update_spectrum_plot(self):
        """Actualiza la gráfica del espectro"""
        # Limpiar la tercera gráfica
        self.ax3.clear()
        self.ax3.set_facecolor('#34495e')
        self.ax3.tick_params(colors='white', labelsize=10)
        self.ax3.grid(True, alpha=0.4, linewidth=0.8)
        for spine in self.ax3.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1.2)
        
        if hasattr(self, 'freq_data') and hasattr(self, 'magnitude_data'):
            # Limitar a frecuencias musicales para mejor visualización
            freq_limit_mask = self.freq_data <= 2000
            display_freqs = self.freq_data[freq_limit_mask]
            display_magnitudes = self.magnitude_data[freq_limit_mask]
            
            self.ax3.plot(display_freqs, display_magnitudes, 'yellow', linewidth=1.5)
            
            region_indicator = "[REGIÓN]" if self.is_zoom_analysis else "[COMPLETO]"
            self.ax3.set_title(f'Espectro de Frecuencias (FFT) {region_indicator}', 
                             color='white', fontsize=14, fontweight='bold')
            self.ax3.set_xlabel('Frecuencia (Hz)', color='white', fontsize=12)
            self.ax3.set_ylabel('Magnitud', color='white', fontsize=12)
            
            # Marcar frecuencia fundamental con mejor visibilidad
            if hasattr(self, 'fundamental_frequency') and self.fundamental_frequency > 0:
                self.ax3.axvline(x=self.fundamental_frequency, color='red', linestyle='-', 
                                linewidth=3, alpha=0.9, label=f'Fundamental: {self.fundamental_frequency:.1f} Hz')
                
                # Añadir texto sobre el pico fundamental
                max_mag_idx = np.argmax(display_magnitudes)
                max_magnitude = display_magnitudes[max_mag_idx]
                self.ax3.text(self.fundamental_frequency + 50, max_magnitude * 0.9, 
                             f'{self.fundamental_frequency:.1f} Hz\n{getattr(self, "detected_note", "---")}', 
                             color='red', fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Marcar armónicos con mejor visibilidad
                if hasattr(self, 'harmonics'):
                    for n, freq, mag in self.harmonics:
                        self.ax3.axvline(x=freq, color='orange', linestyle='--', 
                                        linewidth=2, alpha=0.8, label=f'{n}º armónico: {freq:.1f} Hz')
            
            # Agregar leyenda si hay elementos
            if self.fundamental_frequency > 0 or (hasattr(self, 'harmonics') and self.harmonics):
                self.ax3.legend(loc='upper right', fontsize=10, facecolor='white', framealpha=0.9)
    
    def update_plots(self):
        """Actualiza todas las gráficas con los datos analizados"""
        # Limpiar gráficas anteriores
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Reconfigurar estilos con mejor visualización
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white', labelsize=10)
            ax.grid(True, alpha=0.4, linewidth=0.8)
            # Mejorar el contraste de los ejes
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(1.2)
        
        # Gráfica 1: Oscilación completa en el tiempo
        self.ax1.plot(self.sample_times, self.time_data, 'cyan', linewidth=1.2)
        self.ax1.set_title('Oscilación de la Onda Sonora - Selecciona región para zoom', color='white', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Tiempo (s)', color='white', fontsize=12)
        self.ax1.set_ylabel('Amplitud', color='white', fontsize=12)
        
        # Gráfica 2: Zoom para medir período (región por defecto)
        zoom_duration = min(0.05, 3 * self.period if hasattr(self, 'period') and self.period > 0 else 0.05)
        zoom_samples = int(zoom_duration * self.RATE)
        zoom_times = self.sample_times[:zoom_samples]
        zoom_data = self.time_data[:zoom_samples]
        
        self.ax2.plot(zoom_times, zoom_data, 'lime', linewidth=2.5)
        self.ax2.set_title('Zoom - Medición del Período', color='white', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Tiempo (s)', color='white', fontsize=12)
        self.ax2.set_ylabel('Amplitud', color='white', fontsize=12)
        
        # Marcar períodos si se detectaron con líneas más visibles
        if hasattr(self, 'period') and self.period > 0:
            for i in range(1, int(zoom_duration / self.period) + 1):
                period_time = i * self.period
                if period_time <= zoom_duration:
                    self.ax2.axvline(x=period_time, color='red', linestyle='--', linewidth=2.5, alpha=0.9)
                    # Añadir etiquetas de período
                    self.ax2.text(period_time, np.max(zoom_data) * 0.8, f'T={self.period:.4f}s', 
                                rotation=90, color='red', fontsize=10, fontweight='bold')
        
        # Gráfica 3: Espectro de frecuencias con mejor visualización
        if hasattr(self, 'freq_data') and hasattr(self, 'magnitude_data'):
            # Limitar a frecuencias musicales para mejor visualización
            freq_limit_mask = self.freq_data <= 2000
            display_freqs = self.freq_data[freq_limit_mask]
            display_magnitudes = self.magnitude_data[freq_limit_mask]
            
            self.ax3.plot(display_freqs, display_magnitudes, 'yellow', linewidth=1.5)
            self.ax3.set_title('Espectro de Frecuencias (FFT)', color='white', fontsize=14, fontweight='bold')
            self.ax3.set_xlabel('Frecuencia (Hz)', color='white', fontsize=12)
            self.ax3.set_ylabel('Magnitud', color='white', fontsize=12)
            
            # Marcar frecuencia fundamental con mejor visibilidad
            if hasattr(self, 'fundamental_frequency') and self.fundamental_frequency > 0:
                self.ax3.axvline(x=self.fundamental_frequency, color='red', linestyle='-', 
                                linewidth=3, alpha=0.9, label=f'Fundamental: {self.fundamental_frequency:.1f} Hz')
                
                # Añadir texto sobre el pico fundamental
                max_mag_idx = np.argmax(display_magnitudes)
                max_magnitude = display_magnitudes[max_mag_idx]
                self.ax3.text(self.fundamental_frequency + 50, max_magnitude * 0.9, 
                             f'{self.fundamental_frequency:.1f} Hz\n{getattr(self, "detected_note", "---")}', 
                             color='red', fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Marcar armónicos con mejor visibilidad
                if hasattr(self, 'harmonics'):
                    for n, freq, mag in self.harmonics:
                        self.ax3.axvline(x=freq, color='orange', linestyle='--', 
                                        linewidth=2, alpha=0.8, label=f'{n}º armónico: {freq:.1f} Hz')
            
            # Agregar leyenda si hay elementos
            if (hasattr(self, 'fundamental_frequency') and self.fundamental_frequency > 0) or \
               (hasattr(self, 'harmonics') and self.harmonics):
                self.ax3.legend(loc='upper right', fontsize=10, facecolor='white', framealpha=0.9)
        
        # Ajustar espaciado entre gráficas
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout()
        self.canvas.draw()
        
        # Actualizar etiquetas de información
        self.update_info_labels()
    
    def update_info_labels(self):
        """Actualiza las etiquetas con contexto de región/completo"""
        # Prefijo para indicar si es análisis de región o completo
        if hasattr(self, 'is_zoom_analysis') and self.is_zoom_analysis:
            prefix = f"[REGIÓN {getattr(self, 'zoom_region_duration', 0):.3f}s] "
            label_color = '#3498db'  # Azul para región
        else:
            prefix = "[COMPLETO] "
            label_color = '#2ecc71'  # Verde para completo
        
        # Información temporal
        if hasattr(self, 'period') and self.period > 0:
            self.period_label.config(
                text=f"{prefix}Período (T): {self.period:.6f} s",
                bg=label_color, fg='white'
            )
            self.frequency_label.config(
                text=f"{prefix}Frecuencia (f): {getattr(self, 'frequency_temporal', 0):.2f} Hz",
                bg=label_color, fg='white'
            )
            self.angular_freq_label.config(
                text=f"{prefix}Frecuencia angular (ω): {getattr(self, 'angular_frequency', 0):.2f} rad/s",
                bg=label_color, fg='white'
            )
        else:
            self.period_label.config(text=f"{prefix}Período (T): No detectado", bg='#e74c3c', fg='white')
            self.frequency_label.config(text=f"{prefix}Frecuencia (f): No detectada", bg='#e74c3c', fg='white')
            self.angular_freq_label.config(text=f"{prefix}Frecuencia angular (ω): No detectada", bg='#e74c3c', fg='white')
        
        # Información espectral
        if hasattr(self, 'fundamental_frequency') and self.fundamental_frequency > 0:
            self.fundamental_label.config(
                text=f"{prefix}Frecuencia fundamental: {self.fundamental_frequency:.2f} Hz",
                bg=label_color, fg='white'
            )
            
            if hasattr(self, 'harmonics') and self.harmonics:
                harmonics_text = ", ".join([f"{freq:.1f}Hz" for _, freq, _ in self.harmonics[:3]])
                self.harmonics_label.config(
                    text=f"{prefix}Armónicos: {harmonics_text}",
                    bg=label_color, fg='white'
                )
            else:
                self.harmonics_label.config(text=f"{prefix}Armónicos: No detectados", bg='#f39c12', fg='white')
                
            self.note_detected_label.config(
                text=f"{prefix}Nota musical: {getattr(self, 'detected_note', '---')}",
                bg=label_color, fg='white'
            )
        else:
            self.fundamental_label.config(text=f"{prefix}Frecuencia fundamental: No detectada", bg='#e74c3c', fg='white')
            self.harmonics_label.config(text=f"{prefix}Armónicos: --", bg='#e74c3c', fg='white')
            self.note_detected_label.config(text=f"{prefix}Nota musical: --", bg='#e74c3c', fg='white')
    
    def get_signal_quality_info(self):
        """Función para diagnosticar la calidad de la señal"""
        if self.time_data is None:
            return "No hay datos"
        
        signal_power = np.var(self.time_data)
        max_amplitude = np.max(np.abs(self.time_data))
        
        if signal_power < 1e-8:
            return "Señal muy débil - Verificar micrófono"
        elif max_amplitude < 0.01:
            return "Señal débil - Aumentar volumen"
        elif max_amplitude > 0.9:
            return "Señal saturada - Reducir volumen"
        elif hasattr(self, 'period_confidence'):
            if self.period_confidence > 0.8:
                return f"Excelente detección (confianza: {self.period_confidence:.1%})"
            elif self.period_confidence > 0.5:
                return f"Buena detección (confianza: {self.period_confidence:.1%})"
            else:
                return f"Detección incierta (confianza: {self.period_confidence:.1%})"
        else:
            return "Análisis completado"
    
    def play_audio(self):
        """Reproduce el audio grabado"""
        if not self.recorded_data:
            messagebox.showwarning("Advertencia", "No hay audio grabado para reproducir")
            return
        
        try:
            # Cambiar el texto del botón mientras reproduce
            self.play_button.config(text="⏸ Reproduciendo...", state=tk.DISABLED)
            self.status_label.config(text="Reproduciendo audio...")
            
            # Crear un thread para la reproducción
            play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
            play_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al reproducir audio: {str(e)}")
            self.play_button.config(text="▶ Reproducir Audio", state=tk.NORMAL)
    
    def _play_audio_thread(self):
        """Thread para reproducir audio sin bloquear la interfaz"""
        try:
            # Configurar stream de salida
            output_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True,
                frames_per_buffer=self.CHUNK
            )
            
            # Convertir datos a bytes
            audio_data = np.array(self.recorded_data, dtype=np.float32)
            
            # Reproducir en chunks
            for i in range(0, len(audio_data), self.CHUNK):
                chunk = audio_data[i:i+self.CHUNK]
                if len(chunk) < self.CHUNK:
                    # Rellenar el último chunk con ceros si es necesario
                    chunk = np.pad(chunk, (0, self.CHUNK - len(chunk)), 'constant')
                
                output_stream.write(chunk.tobytes())
            
            output_stream.stop_stream()
            output_stream.close()
            
            # Restaurar botón después de la reproducción
            self.root.after(100, self._restore_play_button)
            
        except Exception as e:
            self.root.after(100, lambda: messagebox.showerror("Error", f"Error durante reproducción: {str(e)}"))
            self.root.after(100, self._restore_play_button)
    
    def _restore_play_button(self):
        """Restaura el estado del botón de reproducción"""
        self.play_button.config(text="▶ Reproducir Audio", state=tk.NORMAL)
        self.status_label.config(text="Reproducción completada")
    
    def save_plots(self):
        """Guarda las gráficas como imágenes con alta calidad"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_type = "region" if (hasattr(self, 'is_zoom_analysis') and self.is_zoom_analysis) else "completo"
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"analisis_sonido_{analysis_type}_{timestamp}.png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Guardar con alta resolución y mejor calidad
                self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                               facecolor='#2c3e50', edgecolor='none',
                               pad_inches=0.2)
                messagebox.showinfo("Éxito", f"Gráficas guardadas en: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar: {str(e)}")
    
    def run(self):
        """Ejecuta la aplicación"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        if self.is_recording:
            self.stop_recording()
        
        # Limpiar selector interactivo
        if hasattr(self, 'rectangle_selector') and self.rectangle_selector:
            self.rectangle_selector.set_active(False)
        
        # Desconectar eventos de mouse
        self.disable_interactive_selection()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        self.root.destroy()

if __name__ == "__main__":
    try:
        import pyaudio
        app = AudioSpectrometer()
        app.run()
    except ImportError as e:
        print("Error: Faltan dependencias.")
        print("Instala las librerías necesarias con:")
        print("pip install pyaudio numpy matplotlib")
        print(f"Error específico: {e}")
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
        input("Presiona Enter para salir...")