# src/trainer.py
import os
import pickle

class ModelTrainer:
    def __init__(self, translation_system):
        self.system = translation_system
        
    def train(self, epochs=100, validation_split=0.2):
        """Entrena el modelo"""
        odam_seq, spanish_seq = self.system.prepare_data()
        
        if odam_seq is None:
            return None
            
        # Para seq2seq, necesitamos los datos de entrada y objetivo desplazado
        decoder_input = spanish_seq[:, :-1]
        decoder_target = spanish_seq[:, 1:]
        
        history = self.system.model.fit(
            [odam_seq, decoder_input],
            decoder_target,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
            batch_size=32
        )
        
        # Guardar modelo
        os.makedirs('models', exist_ok=True)
        self.system.model.save('models/odam_translator.h5')
        
        # Guardar vectorizadores
        self.save_vectorizers()
        
        return history
    
    def save_vectorizers(self):
        """Guarda los vectorizadores para uso futuro"""
        os.makedirs('models/vectorizers', exist_ok=True)
        
        with open('models/vectorizers/odam_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.system.vectorizer_odam, f)
            
        with open('models/vectorizers/spanish_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.system.vectorizer_spanish, f)