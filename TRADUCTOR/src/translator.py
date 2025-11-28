# src/translator.py
import tensorflow as tf
import numpy as np
import pickle

class ODamTranslator:
    def __init__(self):
        self.model = None
        self.vectorizer_odam = None
        self.vectorizer_spanish = None
        self.spanish_vocab = None
        
    def load_model(self):
        """Carga el modelo y vectorizadores entrenados"""
        try:
            self.model = tf.keras.models.load_model('models/odam_translator.h5')
            
            with open('models/vectorizers/odam_vectorizer.pkl', 'rb') as f:
                self.vectorizer_odam = pickle.load(f)
                
            with open('models/vectorizers/spanish_vectorizer.pkl', 'rb') as f:
                self.vectorizer_spanish = pickle.load(f)
                
            self.spanish_vocab = self.vectorizer_spanish.get_vocabulary()
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def translate_odam_to_spanish(self, odam_sentence):
        """Traduce de O'dam a Español"""
        if not self.model:
            return "Modelo no cargado. Entrena primero el sistema."
            
        # Vectorizar entrada
        input_seq = self.vectorizer_odam([odam_sentence])
        
        # Token de inicio
        start_token = [self.vectorizer_spanish(["[start]"])[0][0]]
        
        # Decodificación greedy
        decoded_tokens = []
        current_token = start_token[0]
        
        for _ in range(15):  # Longitud máxima
            predictions = self.model.predict([input_seq, [decoded_tokens]], verbose=0)
            predicted_id = np.argmax(predictions[0, -1, :])
            
            if predicted_id == self.vectorizer_spanish(["[end]"])[0][0]:
                break
                
            decoded_tokens.append(predicted_id)
            current_token = predicted_id
        
        # Convertir a texto
        decoded_text = ' '.join([self.spanish_vocab[token] for token in decoded_tokens])
        return decoded_text.replace(' [end]', '').capitalize()
    
    def translate_spanish_to_odam(self, spanish_sentence):
        """Traduce de Español a O'dam"""
        return "Funcionalidad en desarrollo - necesitas más datos de entrenamiento"