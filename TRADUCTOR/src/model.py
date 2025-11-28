# src/model.py
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

class ODamTranslator(tf.keras.Model):
    def __init__(self, vocab_size_src, vocab_size_tgt, embedding_dim=128, units=256):
        super(ODamTranslator, self).__init__()
        
        # Encoder
        self.encoder_embedding = tf.keras.layers.Embedding(vocab_size_src, embedding_dim)
        self.encoder_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        )
        
        # Decoder
        self.decoder_embedding = tf.keras.layers.Embedding(vocab_size_tgt, embedding_dim)
        self.decoder_gru = tf.keras.layers.GRU(
            units * 2, return_sequences=True, return_state=True
        )
        self.attention = tf.keras.layers.Attention()
        self.output_layer = tf.keras.layers.Dense(vocab_size_tgt, activation='softmax')
        
    def call(self, inputs):
        source, target = inputs
        
        # Encoder
        enc_embed = self.encoder_embedding(source)
        enc_output, forward_state, backward_state = self.encoder_gru(enc_embed)
        enc_state = tf.concat([forward_state, backward_state], axis=-1)
        
        # Decoder
        dec_embed = self.decoder_embedding(target)
        dec_output, dec_state = self.decoder_gru(dec_embed, initial_state=enc_state)
        
        # Attention
        context_vector = self.attention([dec_output, enc_output])
        
        # Output
        combined = tf.concat([dec_output, context_vector], axis=-1)
        output = self.output_layer(combined)
        
        return output

class TranslationSystem:
    def __init__(self):
        from .data_manager import ODamDataManager
        self.data_manager = ODamDataManager()
        self.model = None
        self.vectorizer_odam = TextVectorization(
            max_tokens=10000,
            output_mode='int',
            output_sequence_length=15,
            standardize=None
        )
        self.vectorizer_spanish = TextVectorization(
            max_tokens=10000,
            output_mode='int', 
            output_sequence_length=15,
            standardize=None
        )
    
    def prepare_data(self):
        """Prepara los datos para entrenamiento"""
        if len(self.data_manager.training_pairs) < 5:
            print("Se necesitan al menos 5 pares de oraciones para entrenar")
            return None, None
            
        odam_sentences = [pair['odam'] for pair in self.data_manager.training_pairs]
        spanish_sentences = [pair['spanish'] for pair in self.data_manager.training_pairs]
        
        # Adaptar vectorizadores
        self.vectorizer_odam.adapt(odam_sentences)
        self.vectorizer_spanish.adapt(spanish_sentences)
        
        # Convertir a secuencias
        odam_sequences = self.vectorizer_odam(odam_sentences)
        spanish_sequences = self.vectorizer_spanish(spanish_sentences)
        
        return odam_sequences, spanish_sequences
    
    def build_model(self):
        """Construye el modelo neuronal"""
        vocab_odam_size = len(self.vectorizer_odam.get_vocabulary())
        vocab_spanish_size = len(self.vectorizer_spanish.get_vocabulary())
        
        self.model = ODamTranslator(vocab_odam_size, vocab_spanish_size)
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model