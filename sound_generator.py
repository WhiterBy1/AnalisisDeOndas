import numpy as np
import wave
import os
from datetime import datetime
import math

class SoundGenerator:
    def __init__(self):
        """Inicializa el generador de sonidos"""
        self.sample_rate = 44100  # Frecuencia de muestreo (Hz)
        self.bit_depth = 16       # Profundidad de bits
        self.channels = 1         # Mono
        
        # Diccionario de notas musicales (frecuencias en Hz)
        self.musical_notes = {
            'Do3': 130.81, 'Do#3': 138.59, 'Re3': 146.83, 'Re#3': 155.56,
            'Mi3': 164.81, 'Fa3': 174.61, 'Fa#3': 185.00, 'Sol3': 196.00,
            'Sol#3': 207.65, 'La3': 220.00, 'La#3': 233.08, 'Si3': 246.94,
            
            'Do4': 261.63, 'Do#4': 277.18, 'Re4': 293.66, 'Re#4': 311.13,
            'Mi4': 329.63, 'Fa4': 349.23, 'Fa#4': 369.99, 'Sol4': 392.00,
            'Sol#4': 415.30, 'La4': 440.00, 'La#4': 466.16, 'Si4': 493.88,
            
            'Do5': 523.25, 'Do#5': 554.37, 'Re5': 587.33, 'Re#5': 622.25,
            'Mi5': 659.25, 'Fa5': 698.46, 'Fa#5': 739.99, 'Sol5': 783.99,
            'Sol#5': 830.61, 'La5': 880.00, 'La#5': 932.33, 'Si5': 987.77
        }
    
    def generate_sine_wave(self, frequency, duration, amplitude=0.5):
        """
        Genera una onda sinusoidal pura
        
        Args:
            frequency: Frecuencia en Hz
            duration: Duración en segundos
            amplitude: Amplitud (0.0 a 1.0)
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
        return wave_data
    
    def generate_complex_wave(self, fundamental_freq, duration, harmonics=None, amplitude=0.5):
        """
        Genera una onda compleja con armónicos
        
        Args:
            fundamental_freq: Frecuencia fundamental en Hz
            duration: Duración en segundos
            harmonics: Lista de tuplas (armónico, amplitud_relativa)
                      Ej: [(2, 0.5), (3, 0.3)] = 2do armónico al 50%, 3ro al 30%
            amplitude: Amplitud total
        """
        if harmonics is None:
            harmonics = [(2, 0.3), (3, 0.2), (4, 0.1)]  # Armónicos por defecto
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Comenzar con la fundamental
        wave_data = np.sin(2 * np.pi * fundamental_freq * t)
        
        # Agregar armónicos
        for harmonic_num, harmonic_amplitude in harmonics:
            harmonic_freq = fundamental_freq * harmonic_num
            harmonic_wave = harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * t)
            wave_data += harmonic_wave
        
        # Normalizar y aplicar amplitud
        wave_data = wave_data / np.max(np.abs(wave_data))  # Normalizar
        wave_data *= amplitude
        
        return wave_data
    
    def generate_chord(self, frequencies, duration, amplitude=0.5):
        """
        Genera un acorde (múltiples frecuencias simultáneas)
        
        Args:
            frequencies: Lista de frecuencias en Hz
            duration: Duración en segundos
            amplitude: Amplitud total
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave_data = np.zeros_like(t)
        
        # Sumar todas las frecuencias
        for freq in frequencies:
            wave_data += np.sin(2 * np.pi * freq * t)
        
        # Normalizar y aplicar amplitud
        wave_data = wave_data / len(frequencies)  # Promedio para evitar saturación
        wave_data *= amplitude
        
        return wave_data
    
    def generate_sweep(self, start_freq, end_freq, duration, amplitude=0.5):
        """
        Genera un sweep (barrido de frecuencias)
        
        Args:
            start_freq: Frecuencia inicial en Hz
            end_freq: Frecuencia final en Hz
            duration: Duración en segundos
            amplitude: Amplitud
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Frecuencia que cambia linealmente con el tiempo
        frequency_func = start_freq + (end_freq - start_freq) * t / duration
        
        # Integrar para obtener la fase
        phase = 2 * np.pi * np.cumsum(frequency_func) / self.sample_rate
        
        wave_data = amplitude * np.sin(phase)
        
        return wave_data
    
    def generate_beat_frequency(self, freq1, freq2, duration, amplitude=0.5):
        """
        Genera batimientos (dos frecuencias muy cercanas)
        
        Args:
            freq1, freq2: Dos frecuencias cercanas en Hz
            duration: Duración en segundos
            amplitude: Amplitud
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        wave1 = np.sin(2 * np.pi * freq1 * t)
        wave2 = np.sin(2 * np.pi * freq2 * t)
        
        # La suma crea el efecto de batimiento
        wave_data = amplitude * (wave1 + wave2) / 2
        
        return wave_data
    
    def add_noise(self, wave_data, noise_level=0.05):
        """
        Añade ruido blanco a una señal
        
        Args:
            wave_data: Datos de onda
            noise_level: Nivel de ruido (0.0 a 1.0)
        """
        noise = np.random.normal(0, noise_level, len(wave_data))
        return wave_data + noise
    
    def apply_envelope(self, wave_data, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
        """
        Aplica un envolvente ADSR a la señal
        
        Args:
            wave_data: Datos de onda
            attack, decay, sustain, release: Parámetros del envolvente (0.0 a 1.0)
        """
        total_samples = len(wave_data)
        envelope = np.ones(total_samples)
        
        # Calcular puntos de transición
        attack_samples = int(attack * total_samples)
        decay_samples = int(decay * total_samples)
        release_samples = int(release * total_samples)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        decay_start = attack_samples
        decay_end = decay_start + decay_samples
        envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_samples)
        
        # Sustain
        sustain_start = decay_end
        sustain_end = sustain_start + sustain_samples
        envelope[sustain_start:sustain_end] = sustain
        
        # Release
        release_start = sustain_end
        envelope[release_start:] = np.linspace(sustain, 0, release_samples)
        
        return wave_data * envelope
    
    def save_wave_file(self, wave_data, filename):
        """
        Guarda los datos de onda como archivo WAV
        
        Args:
            wave_data: Datos de onda (numpy array)
            filename: Nombre del archivo
        """
        # Convertir a enteros de 16 bits
        wave_data_int = np.int16(wave_data * 32767)
        
        # Crear directorio si no existe
        os.makedirs('generated_sounds', exist_ok=True)
        filepath = os.path.join('generated_sounds', filename)
        
        # Guardar archivo WAV
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(wave_data_int.tobytes())
        
        print(f"✅ Sonido guardado: {filepath}")
        return filepath
    
    def create_test_suite(self):
        """Crea una suite completa de sonidos de prueba"""
        print("🎵 Generando suite de sonidos de prueba...")
        print("=" * 50)
        
        # 1. Ondas sinusoidales puras (frecuencias específicas)
        test_frequencies = [220, 440, 880, 1000, 1760]  # La3, La4, La5, 1kHz, La6
        for freq in test_frequencies:
            wave_data = self.generate_sine_wave(freq, 3.0, 0.7)
            filename = f"sine_{freq}Hz_3s.wav"
            self.save_wave_file(wave_data, filename)
        
        # 2. Notas musicales específicas
        musical_tests = ['La4', 'Do4', 'Mi4', 'Sol4', 'Do5']
        for note in musical_tests:
            freq = self.musical_notes[note]
            wave_data = self.generate_sine_wave(freq, 2.5, 0.6)
            filename = f"nota_{note}_{freq:.1f}Hz.wav"
            self.save_wave_file(wave_data, filename)
        
        # 3. Ondas complejas con armónicos
        complex_tests = [
            (440, [(2, 0.5), (3, 0.3), (4, 0.2)], "violin_like"),
            (220, [(2, 0.8), (3, 0.6), (4, 0.4), (5, 0.2)], "brass_like"),
            (880, [(2, 0.3), (3, 0.1)], "simple_harmonic")
        ]
        
        for freq, harmonics, name in complex_tests:
            wave_data = self.generate_complex_wave(freq, 4.0, harmonics, 0.6)
            wave_data = self.apply_envelope(wave_data, 0.1, 0.2, 0.6, 0.1)
            filename = f"complex_{name}_{freq}Hz.wav"
            self.save_wave_file(wave_data, filename)
        
        # 4. Acordes
        chord_tests = [
            ([261.63, 329.63, 392.00], "Do_Mayor"),  # Do-Mi-Sol
            ([220.00, 277.18, 329.63], "La_menor"),  # La-Do-Mi
            ([440.00, 554.37, 659.25], "La_Mayor")   # La-Do#-Mi
        ]
        
        for frequencies, chord_name in chord_tests:
            wave_data = self.generate_chord(frequencies, 3.5, 0.5)
            wave_data = self.apply_envelope(wave_data, 0.2, 0.1, 0.7, 0.0)
            filename = f"acorde_{chord_name}.wav"
            self.save_wave_file(wave_data, filename)
        
        # 5. Barridos de frecuencia
        sweep_tests = [
            (100, 1000, "grave_a_agudo"),
            (1000, 100, "agudo_a_grave"),
            (200, 800, "rango_medio")
        ]
        
        for start, end, name in sweep_tests:
            wave_data = self.generate_sweep(start, end, 5.0, 0.6)
            filename = f"sweep_{name}_{start}to{end}Hz.wav"
            self.save_wave_file(wave_data, filename)
        
        # 6. Batimientos
        beat_tests = [
            (440, 445, "batimiento_5Hz"),
            (880, 885, "batimiento_5Hz_agudo"),
            (220, 222, "batimiento_2Hz")
        ]
        
        for freq1, freq2, name in beat_tests:
            wave_data = self.generate_beat_frequency(freq1, freq2, 4.0, 0.6)
            filename = f"beat_{name}.wav"
            self.save_wave_file(wave_data, filename)
        
        # 7. Sonidos con ruido
        wave_clean = self.generate_sine_wave(440, 3.0, 0.6)
        wave_noisy = self.add_noise(wave_clean, 0.1)
        self.save_wave_file(wave_noisy, "sine_440Hz_con_ruido.wav")
        
        # 8. Diferentes duraciones para probar el análisis temporal
        durations = [1.0, 2.0, 5.0]
        for duration in durations:
            wave_data = self.generate_sine_wave(440, duration, 0.6)
            filename = f"sine_440Hz_{duration:.0f}s.wav"
            self.save_wave_file(wave_data, filename)
        
        print("=" * 50)
        print("🎉 ¡Suite de prueba completada!")
        print(f"📁 Todos los archivos guardados en: generated_sounds/")
        print("\n📋 Resumen de archivos creados:")
        print("   • Ondas sinusoidales puras (varias frecuencias)")
        print("   • Notas musicales específicas")
        print("   • Ondas complejas con armónicos")
        print("   • Acordes musicales")
        print("   • Barridos de frecuencia")
        print("   • Batimientos")
        print("   • Sonidos con ruido")
        print("   • Diferentes duraciones")
        
    def create_custom_sound(self):
        """Interfaz interactiva para crear sonidos personalizados"""
        print("\n🎼 Creador de Sonidos Personalizados")
        print("=" * 40)
        
        while True:
            print("\nOpciones disponibles:")
            print("1. Onda sinusoidal pura")
            print("2. Onda con armónicos")
            print("3. Acorde")
            print("4. Barrido de frecuencias")
            print("5. Batimientos")
            print("6. Salir")
            
            choice = input("\nSelecciona una opción (1-6): ").strip()
            
            if choice == '1':
                freq = float(input("Frecuencia (Hz): "))
                duration = float(input("Duración (segundos): "))
                wave_data = self.generate_sine_wave(freq, duration, 0.7)
                filename = f"custom_sine_{freq}Hz_{duration}s.wav"
                self.save_wave_file(wave_data, filename)
                
            elif choice == '2':
                freq = float(input("Frecuencia fundamental (Hz): "))
                duration = float(input("Duración (segundos): "))
                harmonics = [(2, 0.5), (3, 0.3), (4, 0.2)]  # Por defecto
                wave_data = self.generate_complex_wave(freq, duration, harmonics, 0.7)
                wave_data = self.apply_envelope(wave_data)
                filename = f"custom_complex_{freq}Hz_{duration}s.wav"
                self.save_wave_file(wave_data, filename)
                
            elif choice == '3':
                freqs_input = input("Frecuencias separadas por comas (ej: 440,554,659): ")
                frequencies = [float(f.strip()) for f in freqs_input.split(',')]
                duration = float(input("Duración (segundos): "))
                wave_data = self.generate_chord(frequencies, duration, 0.6)
                filename = f"custom_chord_{len(frequencies)}notas_{duration}s.wav"
                self.save_wave_file(wave_data, filename)
                
            elif choice == '4':
                start_freq = float(input("Frecuencia inicial (Hz): "))
                end_freq = float(input("Frecuencia final (Hz): "))
                duration = float(input("Duración (segundos): "))
                wave_data = self.generate_sweep(start_freq, end_freq, duration, 0.6)
                filename = f"custom_sweep_{start_freq}to{end_freq}Hz_{duration}s.wav"
                self.save_wave_file(wave_data, filename)
                
            elif choice == '5':
                freq1 = float(input("Primera frecuencia (Hz): "))
                freq2 = float(input("Segunda frecuencia (Hz): "))
                duration = float(input("Duración (segundos): "))
                wave_data = self.generate_beat_frequency(freq1, freq2, duration, 0.6)
                filename = f"custom_beat_{freq1}_{freq2}Hz_{duration}s.wav"
                self.save_wave_file(wave_data, filename)
                
            elif choice == '6':
                break
                
            else:
                print("❌ Opción no válida")

def main():
    """Función principal"""
    print("🎵 Generador de Sonidos de Prueba para Análisis de Ondas")
    print("=" * 60)
    
    generator = SoundGenerator()
    
    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Crear suite completa de pruebas")
        print("2. Crear sonidos personalizados")
        print("3. Salir")
        
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == '1':
            generator.create_test_suite()
        elif choice == '2':
            generator.create_custom_sound()
        elif choice == '3':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()