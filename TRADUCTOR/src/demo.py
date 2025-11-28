# demo.py
import os
import json
import tensorflow as tf
import numpy as np
from datetime import datetime

class ODamDataManager:
    def __init__(self):
        self.vocab_odam = set()
        self.vocab_spanish = set()
        self.training_pairs = []
        self.word_translations = {}
        self.grammar_rules = {
            'plural_rules': {},
            'verb_conjugation': {},
            'word_order': 'VSO'
        }
        self.load_data()  # Cargar datos inmediatamente al inicializar
    
    def add_word(self, odam_word, spanish_word, word_type='sustantivo'):
        """Agrega una palabra al vocabulario y guarda inmediatamente"""
        odam_clean = odam_word.strip()
        spanish_clean = spanish_word.strip()
        
        self.vocab_odam.add(odam_clean)
        self.vocab_spanish.add(spanish_clean)
        self.word_translations[odam_clean] = spanish_clean
        self.word_translations[spanish_clean] = odam_clean
        
        print(f"✓ Palabra agregada: '{odam_clean}' -> '{spanish_clean}'")
        self.save_data()  # Guardar inmediatamente después de agregar
    
    def add_sentence_pair(self, odam_sentence, spanish_sentence):
        """Agrega un par de oraciones y guarda inmediatamente"""
        odam_clean = odam_sentence.strip()
        spanish_clean = spanish_sentence.strip()
        
        self.training_pairs.append({
            'odam': odam_clean,
            'spanish': spanish_clean,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✓ Oración agregada: '{odam_clean}' -> '{spanish_clean}'")
        self.save_data()  # Guardar inmediatamente después de agregar
    
    def initialize_base_vocabulary(self):
        """Inicializa con el vocabulario base"""
        base_words = {
            'tai': 'fuego',
            'juuk': 'pino', 
            'jujuk': 'pinos',
            'kubh+x': 'humo',
            "b+p+'": 'primera',
            "oidha'": 'cerro',
            "sai'": 'zacate',
            "judai'": 'piedra',
            'd+b+r': 'tierra',
            "sudai'": 'agua',
            'jatkam': 'personas',
            'tua': 'encino',
            "ba'bhak": 'casas',
            "sasoi'": 'animales',
            'tanolh': 'sol',
            'j+b+lh': 'viento',
            'mubalh': 'mosca',
            'gabhar': 'parcela',
            'jun': 'maíz',
            'gagox': 'perro',
            'ubil': 'mujer',
            'chioñ': 'hombre',
            "tuka'": 'noche',
            "jix b+'": 'rojo',
            'jix ch+do': 'azul', 
            'jix uam': 'amarillo',
            "jix koma'": 'gris',
            'mistuiñ': 'gato',
            'tai tussadham': 'brigada'
        }
        
        for odam, spanish in base_words.items():
            self.vocab_odam.add(odam)
            self.vocab_spanish.add(spanish)
            self.word_translations[odam] = spanish
            self.word_translations[spanish] = odam
            
        # Agregar algunas oraciones de ejemplo
        base_sentences = [
            {"odam": "tai kubh+x", "spanish": "el fuego hace humo"},
            {"odam": "ubil jun", "spanish": "la mujer siembra maíz"},
            {"odam": "chioñ gabhar", "spanish": "el hombre cultiva la parcela"},
            {"odam": "gagox sasoi", "spanish": "el perro cuida animales"},
            {"odam": "tanolh jix uam", "spanish": "el sol es amarillo"},
            {"odam": "tuka' jatkam", "spanish": "las personas en la noche"},
            {"odam": "ba'bhak tua", "spanish": "casas de encino"},
            {"odam": "sudai' judai'", "spanish": "agua sobre piedra"},
            {"odam": "mubalh j+b+lh", "spanish": "mosca en el viento"},
            {"odam": "juuk jix b+'", "spanish": "pino rojo"}
        ]
        
        for sentence in base_sentences:
            self.training_pairs.append({
                'odam': sentence['odam'],
                'spanish': sentence['spanish'],
                'timestamp': datetime.now().isoformat()
            })
        
        self.save_data()
        print("✓ Vocabulario base inicializado y guardado")
    
    def translate_word(self, word):
        """Traduce una palabra individual"""
        if not word:
            return None
            
        word_clean = word.strip().lower()
        
        # Buscar coincidencia exacta
        if word_clean in self.word_translations:
            return self.word_translations[word_clean]
        
        # Buscar coincidencia insensible a mayúsculas
        for key, value in self.word_translations.items():
            if key.lower() == word_clean:
                return value
        
        return None
    
    def find_similar_words(self, word):
        """Encuentra palabras similares en el vocabulario"""
        if not word:
            return []
            
        word_clean = word.strip().lower()
        similar = []
        
        for vocab_word in self.vocab_odam.union(self.vocab_spanish):
            vocab_lower = vocab_word.lower()
            if (word_clean in vocab_lower or 
                vocab_lower in word_clean or 
                word_clean == vocab_lower):
                
                translation = self.word_translations.get(vocab_word, "?")
                similar.append((vocab_word, translation))
        
        return similar
    
    def translate_sentence_word_by_word(self, sentence, source_lang='odam'):
        """Traduce una oración palabra por palabra"""
        if not sentence:
            return ""
            
        words = sentence.split()
        translated_words = []
        
        for word in words:
            translation = self.translate_word(word)
            if translation:
                translated_words.append(translation)
            else:
                # Si no encuentra traducción, mostrar la palabra original entre corchetes
                translated_words.append(f"[{word}]")
        
        return ' '.join(translated_words)
    
    def get_vocabulary_table(self):
        """Retorna el vocabulario en formato de tabla"""
        table = []
        for odam_word in sorted(self.vocab_odam):
            spanish_word = self.word_translations.get(odam_word, "?")
            table.append({"O'dam": odam_word, "Español": spanish_word})
        return table
    
    def get_sentences_table(self):
        """Retorna las oraciones en formato de tabla"""
        return self.training_pairs
    
    def save_data(self, filename='data/odam_data.json'):
        """Guarda todos los datos"""
        try:
            os.makedirs('data', exist_ok=True)
            data = {
                'vocab_odam': list(self.vocab_odam),
                'vocab_spanish': list(self.vocab_spanish),
                'training_pairs': self.training_pairs,
                'word_translations': self.word_translations,
                'grammar_rules': self.grammar_rules
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"✘ Error guardando datos: {e}")
            return False
    
    def load_data(self, filename='data/odam_data.json'):
        """Carga datos existentes"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.vocab_odam = set(data['vocab_odam'])
                    self.vocab_spanish = set(data['vocab_spanish'])
                    self.training_pairs = data['training_pairs']
                    self.word_translations = data.get('word_translations', {})
                    self.grammar_rules = data.get('grammar_rules', {})
                print(f"✓ Datos cargados: {len(self.vocab_odam)} palabras, {len(self.training_pairs)} oraciones")
                return True
            else:
                print("❗ No se encontraron datos previos, creando vocabulario base...")
                self.initialize_base_vocabulary()
                return True
        except Exception as e:
            print(f"✘ Error cargando datos: {e}")
            print("❗ Inicializando con vocabulario base...")
            self.initialize_base_vocabulary()
            return False


class TranslationSystem:
    def __init__(self):
        self.data_manager = ODamDataManager()  # Se auto-inicializa con datos
    
    def get_data_manager(self):
        """Retorna el gestor de datos para uso externo"""
        return self.data_manager


class ODamTranslator:
    def __init__(self):
        self.system = TranslationSystem()
        self.data_manager = self.system.get_data_manager()
        self.model = None
    
    def translate_word_interactive(self, word):
        """Traduce una palabra de manera interactiva"""
        if not word or not word.strip():
            print("✘ Por favor ingresa una palabra válida")
            return
            
        word_clean = word.strip()
        translation = self.data_manager.translate_word(word_clean)
        
        if translation:
            # Determinar dirección de la traducción
            if word_clean in self.data_manager.vocab_odam:
                print(f"✓ O'dam -> Español: '{word_clean}' -> '{translation}'")
            else:
                print(f"✓ Español -> O'dam: '{word_clean}' -> '{translation}'")
        else:
            print(f"✘ Palabra '{word_clean}' no encontrada en el vocabulario")
            
            # Mostrar palabras similares
            similar = self.data_manager.find_similar_words(word_clean)
            if similar:
                print("🔎 Palabras similares encontradas:")
                for original, trad in similar[:5]:
                    print(f"   '{original}' -> '{trad}'")
            else:
                print("💡 Sugerencia: Agrega esta palabra al vocabulario usando la opción 4 del menú principal")
    
    def translate_sentence_interactive(self, sentence, source_lang='auto'):
        """Traduce una oración de manera interactiva"""
        if not sentence or not sentence.strip():
            print("✘ Por favor ingresa una oración válida")
            return
            
        sentence_clean = sentence.strip()
        
        # Detectar idioma automáticamente si no se especifica
        if source_lang == 'auto':
            # Si contiene caracteres típicos del O'dam, asumir que es O'dam
            if any(char in sentence_clean for char in ["'", "+", "ñ", "x"]):
                source_lang = 'odam'
                target_lang = 'español'
            else:
                source_lang = 'español'
                target_lang = 'odam'
        
        translation = self.data_manager.translate_sentence_word_by_word(sentence_clean, source_lang)
        
        print(f"   Traducción ({source_lang} -> {target_lang}):")
        print(f"   Original: '{sentence_clean}'")
        print(f"   Traducción: '{translation}'")
        
        # Mostrar palabras no encontradas
        unknown_words = [word for word in sentence_clean.split() 
                        if f"[{word}]" in translation]
        if unknown_words:
            print(f"⚠️  Palabras no encontradas: {', '.join(unknown_words)}")
            print("💡 Sugerencia: Agrega estas palabras al vocabulario")


def display_vocabulary_table(data_manager):
    """Muestra el vocabulario en formato de tabla"""
    vocab_table = data_manager.get_vocabulary_table()
    
    print("\n" + "="*60)
    print("VOCABULARIO O'DAM - ESPAÑOL")
    print("="*60)
    print(f"{'O\'dam':<25} {'Español':<25}")
    print("-" * 50)
    
    for item in vocab_table:
        print(f"{item['O\'dam']:<25} {item['Español']:<25}")
    
    print(f"\nTotal: {len(vocab_table)} palabras")

def display_sentences_table(data_manager):
    """Muestra las oraciones en formato de tabla"""
    sentences = data_manager.get_sentences_table()
    
    print("\n" + "="*90)
    print("ORACIONES DE ENTRENAMIENTO")
    print("="*90)
    print(f"{'#':<3} {'O\'dam':<40} {'Español':<40}")
    print("-" * 85)
    
    for i, sentence in enumerate(sentences, 1):
        odam = sentence['odam'][:38] + "..." if len(sentence['odam']) > 38 else sentence['odam']
        spanish = sentence['spanish'][:38] + "..." if len(sentence['spanish']) > 38 else sentence['spanish']
        print(f"{i:<3} {odam:<40} {spanish:<40}")
    
    print(f"\n Total: {len(sentences)} oraciones")

def translation_demo(translator):
    """Demo interactiva de traducción"""
    print("\n" + "="*50)
    print("🗣️ MODO TRADUCCIÓN")
    print("="*50)
    
    while True:
        print("\n Opciones de traducción:")
        print("1. Traducir palabra (automático O'dam↔Español)")
        print("2. Traducir oración (automático O'dam↔Español)")
        print("3. Traducir palabra de O'dam a Español")
        print("4. Traducir palabra de Español a O'dam")
        print("5. Buscar palabras similares")
        print("6. Ver estadísticas del vocabulario")
        print("7. Volver al menú principal")
        
        choice = input("\nSelecciona opción (1-7): ").strip()
        
        if choice == '1':
            word = input("Palabra a traducir: ").strip()
            if word:
                translator.translate_word_interactive(word)
            else:
                print("✘ Por favor ingresa una palabra")
                
        elif choice == '2':
            sentence = input("Oración a traducir: ").strip()
            if sentence:
                translator.translate_sentence_interactive(sentence)
            else:
                print("✘ Por favor ingresa una oración")
                
        elif choice == '3':
            word = input("Palabra en O'dam a traducir: ").strip()
            if word:
                translator.translate_word_interactive(word)
            else:
                print("✘ Por favor ingresa una palabra")
                
        elif choice == '4':
            word = input("Palabra en Español a traducir: ").strip()
            if word:
                translator.translate_word_interactive(word)
            else:
                print("✘ Por favor ingresa una palabra")
                
        elif choice == '5':
            word = input("Palabra a buscar: ").strip()
            if word:
                similar = translator.data_manager.find_similar_words(word)
                if similar:
                    print(f"🔍 Palabras similares a '{word}':")
                    for original, translation in similar:
                        print(f"   '{original}' -> '{translation}'")
                else:
                    print(f"✘ No se encontraron palabras similares a '{word}'")
            else:
                print("✘ Por favor ingresa una palabra")
                
        elif choice == '6':
            dm = translator.data_manager
            print("\n ESTADÍSTICAS DEL VOCABULARIO:")
            print(f"    Palabras O'dam: {len(dm.vocab_odam)}")
            print(f"    Palabras Español: {len(dm.vocab_spanish)}")
            print(f"    Oraciones de entrenamiento: {len(dm.training_pairs)}")
            print(f"    Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
        elif choice == '7':
            break
        else:
            print("✘ Opción no válida")

def main_demo():
    """Demo principal del sistema"""
    print("SISTEMA DE TRADUCCIÓN O'DAM")
    
    # Inicializar sistema (se auto-carga con datos)
    translator = ODamTranslator()
    data_manager = translator.data_manager
    
    print(" ✓ Sistema inicializado y datos cargados")
    
    # Menú principal mejorado
    while True:
        #print("\n" + "="*50)
        print("MENÚ")
        #Sprint("="*50)
        print("1.  Ver vocabulario completo")
        print("2.  Ver oraciones de entrenamiento")
        print("3.  Modo traducción")
        print("4.  Agregar nueva palabra")
        print("5.  Agregar nueva oración")
        print("6.  Guardar y salir")
        
        choice = input("\nSelecciona opción (1-6): ").strip()
        
        if choice == '1':
            display_vocabulary_table(data_manager)
            
        elif choice == '2':
            display_sentences_table(data_manager)
            
        elif choice == '3':
            translation_demo(translator)
            
        elif choice == '4':
            print("\n AGREGAR NUEVA PALABRA")
            odam_word = input("Palabra en O'dam: ").strip()
            spanish_word = input("Traducción al español: ").strip()
            if odam_word and spanish_word:
                data_manager.add_word(odam_word, spanish_word)
                print("✓ ¡Palabra agregada y guardada! Puedes usarla inmediatamente en las traducciones.")
            else:
                print("✘ Ambas palabras son requeridas")
                
        elif choice == '5':
            print("\n AGREGAR NUEVA ORACIÓN")
            odam_sentence = input("Oración en O'dam: ").strip()
            spanish_sentence = input("Traducción al español: ").strip()
            if odam_sentence and spanish_sentence:
                data_manager.add_sentence_pair(odam_sentence, spanish_sentence)
                print(" ✓ ¡Oración agregada y guardada!")
            else:
                print("✘ Ambas oraciones son requeridas")
                
        elif choice == '6':
            # Confirmar guardado
            if data_manager.save_data():
                print("\n✓ Todos los datos han sido guardados exitosamente")
            else:
                print("\nHubo un problema al guardar los datos")
            
            # Mostrar resumen final
            print("\n--- RESUMEN FINAL ---")
            print(f"Palabras O'dam: {len(data_manager.vocab_odam)}")
            print(f"Palabras Español: {len(data_manager.vocab_spanish)}")
            print(f"Oraciones de entrenamiento: {len(data_manager.training_pairs)}")
            print(f"Archivo de datos: data/odam_data.json")
            break
            
        else:
            print("✘ Opción no válida")

if __name__ == "__main__":
    main_demo()