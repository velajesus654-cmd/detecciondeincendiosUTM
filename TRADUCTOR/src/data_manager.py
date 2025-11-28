# src/data_manager.py
import json
import os
from datetime import datetime

class ODamDataManager:
    def __init__(self):
        self.vocab_odam = set()
        self.vocab_spanish = set()
        self.training_pairs = []
        self.grammar_rules = {
            'plural_rules': {},
            'verb_conjugation': {},
            'word_order': 'VSO'
        }
    
    def add_word(self, odam_word, spanish_word, word_type='sustantivo'):
        """Agrega una palabra al vocabulario"""
        self.vocab_odam.add(odam_word)
        self.vocab_spanish.add(spanish_word)
        
    def add_sentence_pair(self, odam_sentence, spanish_sentence):
        """Agrega un par de oraciones para entrenamiento"""
        self.training_pairs.append({
            'odam': odam_sentence,
            'spanish': spanish_sentence,
            'timestamp': datetime.now().isoformat()
        })
        
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
            self.add_word(odam, spanish)
            
        # Agregar algunas oraciones de ejemplo
        base_sentences = [
            {"odam": "tai kubh+x", "spanish": "el fuego hace humo"},
            {"odam": "ubil jun", "spanish": "mujer maíz"},
            {"odam": "chioñ gabhar", "spanish": "hombre parcela"},
            {"odam": "gagox sasoi", "spanish": "perro animales"},
            {"odam": "tanolh jix uam", "spanish": "sol amarillo"}
        ]
        
        for sentence in base_sentences:
            self.add_sentence_pair(sentence['odam'], sentence['spanish'])
    
    def save_data(self, filename='data/odam_data.json'):
        """Guarda todos los datos"""
        os.makedirs('data', exist_ok=True)
        data = {
            'vocab_odam': list(self.vocab_odam),
            'vocab_spanish': list(self.vocab_spanish),
            'training_pairs': self.training_pairs,
            'grammar_rules': self.grammar_rules
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_data(self, filename='data/odam_data.json'):
        """Carga datos existentes"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocab_odam = set(data['vocab_odam'])
                self.vocab_spanish = set(data['vocab_spanish'])
                self.training_pairs = data['training_pairs']
                self.grammar_rules = data['grammar_rules']
                return True
        except FileNotFoundError:
            return False