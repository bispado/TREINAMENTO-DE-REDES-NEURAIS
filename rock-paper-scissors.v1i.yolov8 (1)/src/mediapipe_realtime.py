"""
Aplicação de Reconhecimento de Rock Paper Scissors em Tempo Real com MediaPipe

Este script usa MediaPipe Hands para detectar mãos e classificar gestos de
Pedra, Papel e Tesoura baseado na geometria dos landmarks detectados.

Autor: Projeto de Visão Computacional
Data: 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import random


class RockPaperScissorsDetector:
    """
    Detector de gestos Rock, Paper, Scissors usando MediaPipe Hands.
    """
    
    def __init__(self, 
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 max_num_hands=2):
        """
        Inicializa o detector de gestos.
        
        Args:
            min_detection_confidence: Confiança mínima para detecção de mãos
            min_tracking_confidence: Confiança mínima para rastreamento
            max_num_hands: Número máximo de mãos a detectar
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Histórico de predições para suavização
        self.prediction_history = deque(maxlen=5)
        
        # Estatísticas
        self.total_detections = 0
        self.gesture_counts = {'Rock': 0, 'Paper': 0, 'Scissors': 0}
        
        # Cores para visualização (BGR)
        self.colors = {
            'Rock': (0, 0, 255),      # Vermelho
            'Paper': (0, 255, 0),     # Verde
            'Scissors': (255, 0, 0),  # Azul
            'Unknown': (128, 128, 128) # Cinza
        }
        
        # Para FPS
        self.prev_time = 0
        
    def count_extended_fingers(self, hand_landmarks, handedness):
        """
        Conta quantos dedos estão estendidos.
        
        Args:
            hand_landmarks: Landmarks da mão detectada
            handedness: Se é mão esquerda ou direita
            
        Returns:
            Número de dedos estendidos (0-5)
        """
        fingers_extended = []
        
        # Landmarks dos dedos
        # Polegar: 4 (tip), 3 (ip), 2 (mcp)
        # Indicador: 8 (tip), 6 (mcp)
        # Médio: 12 (tip), 10 (mcp)
        # Anelar: 16 (tip), 14 (mcp)
        # Mindinho: 20 (tip), 18 (mcp)
        
        # Verificar polegar (lógica diferente por ser horizontal)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Determinar se é mão esquerda ou direita
        is_right_hand = handedness.classification[0].label == 'Right'
        
        # Para o polegar, comparamos a coordenada X
        if is_right_hand:
            thumb_extended = thumb_tip.x < thumb_ip.x
        else:
            thumb_extended = thumb_tip.x > thumb_ip.x
        fingers_extended.append(thumb_extended)
        
        # Verificar outros dedos (comparação vertical Y)
        finger_tips = [8, 12, 16, 20]  # Indicador, Médio, Anelar, Mindinho
        finger_mcps = [6, 10, 14, 18]  # Base dos dedos
        
        for tip, mcp in zip(finger_tips, finger_mcps):
            # Dedo estendido se a ponta está acima da base (Y menor)
            tip_y = hand_landmarks.landmark[tip].y
            mcp_y = hand_landmarks.landmark[mcp].y
            fingers_extended.append(tip_y < mcp_y)
        
        return sum(fingers_extended)
    
    def get_finger_states(self, hand_landmarks, handedness):
        """
        Obtém o estado de cada dedo (estendido ou dobrado).
        
        Returns:
            Lista de booleans [polegar, indicador, médio, anelar, mindinho]
        """
        fingers = []
        
        # Polegar
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        is_right_hand = handedness.classification[0].label == 'Right'
        
        if is_right_hand:
            thumb_extended = thumb_tip.x < thumb_ip.x
        else:
            thumb_extended = thumb_tip.x > thumb_ip.x
        fingers.append(thumb_extended)
        
        # Outros dedos
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
        
        return fingers
    
    def classify_gesture(self, hand_landmarks, handedness):
        """
        Classifica o gesto como Rock, Paper ou Scissors.
        
        Args:
            hand_landmarks: Landmarks da mão
            handedness: Se é mão esquerda ou direita
            
        Returns:
            Tupla (gesto, confiança)
        """
        num_fingers = self.count_extended_fingers(hand_landmarks, handedness)
        fingers = self.get_finger_states(hand_landmarks, handedness)
        
        # Lógica de classificação
        confidence = 0.0
        gesture = 'Unknown'
        
        # ROCK: 0 ou 1 dedo estendido (punho fechado)
        if num_fingers <= 1:
            gesture = 'Rock'
            confidence = 0.95 if num_fingers == 0 else 0.85
            
        # PAPER: 4 ou 5 dedos estendidos (mão aberta)
        elif num_fingers >= 4:
            gesture = 'Paper'
            confidence = 0.95 if num_fingers == 5 else 0.85
            
        # SCISSORS: 2 ou 3 dedos estendidos
        elif num_fingers in [2, 3]:
            # Verificar se é realmente tesoura (indicador e médio estendidos)
            # fingers = [polegar, indicador, médio, anelar, mindinho]
            if fingers[1] and fingers[2]:  # Indicador e médio estendidos
                gesture = 'Scissors'
                confidence = 0.90
            else:
                # Pode ser um gesto ambíguo
                gesture = 'Scissors'
                confidence = 0.70
        
        return gesture, confidence
    
    def smooth_prediction(self, current_gesture):
        """
        Suaviza as predições usando histórico.
        
        Args:
            current_gesture: Gesto detectado atualmente
            
        Returns:
            Gesto suavizado (mais comum no histórico)
        """
        self.prediction_history.append(current_gesture)
        
        if len(self.prediction_history) < 3:
            return current_gesture
        
        # Retornar o gesto mais comum no histórico recente
        gestures = list(self.prediction_history)
        return max(set(gestures), key=gestures.count)
    
    def process_frame(self, frame):
        """
        Processa um frame de vídeo para detectar e classificar gestos.
        
        Args:
            frame: Frame de vídeo (BGR)
            
        Returns:
            Frame processado com anotações
        """
        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = self.hands.process(frame_rgb)
        
        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        
        # Lista de detecções
        detections = []
        
        # Se detectou mãos
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                  results.multi_handedness):
                # Classificar gesto
                gesture, confidence = self.classify_gesture(hand_landmarks, handedness)
                
                # Suavizar predição
                gesture = self.smooth_prediction(gesture)
                
                # Atualizar estatísticas
                self.total_detections += 1
                if gesture in self.gesture_counts:
                    self.gesture_counts[gesture] += 1
                
                # Desenhar landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Obter posição da mão
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                # Adicionar margem
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                # Desenhar bounding box
                color = self.colors.get(gesture, self.colors['Unknown'])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Adicionar texto com o gesto
                hand_label = handedness.classification[0].label
                label = f"{hand_label}: {gesture} ({confidence:.0%})"
                
                # Fundo para o texto
                (text_width, text_height), _ = cv2.getTextSize(label, 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 
                                                               0.7, 2)
                cv2.rectangle(frame, 
                            (x_min, y_min - text_height - 10),
                            (x_min + text_width, y_min),
                            color, -1)
                
                # Texto
                cv2.putText(frame, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detections.append({
                    'gesture': gesture,
                    'confidence': confidence,
                    'hand': hand_label,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
        
        # Adicionar informações na tela
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Contador de detecções
        y_offset = 60
        cv2.putText(frame, f"Total Detections: {self.total_detections}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Estatísticas de gestos
        y_offset += 30
        for gesture, count in self.gesture_counts.items():
            percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
            text = f"{gesture}: {count} ({percentage:.1f}%)"
            color = self.colors.get(gesture, (255, 255, 255))
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Instruções
        cv2.putText(frame, "Pressione 'q' para sair | 's' para screenshot", 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame, detections
    
    def release(self):
        """Libera recursos do MediaPipe."""
        self.hands.close()


def main():
    """Função principal para executar a aplicação."""
    
    print("=" * 60)
    print("Rock Paper Scissors - Detecção em Tempo Real com MediaPipe")
    print("=" * 60)
    print("\nInstruções:")
    print("  - Mostre sua mão para a câmera")
    print("  - Faça os gestos de Rock (punho), Paper (mão aberta) ou Scissors (tesoura)")
    print("  - Pressione 'q' para sair")
    print("  - Pressione 's' para salvar screenshot")
    print("  - Pressione 'r' para resetar estatísticas")
    print("\n" + "=" * 60)
    
    # Inicializar detector
    detector = RockPaperScissorsDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    
    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a webcam!")
        return
    
    # Configurar resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n✅ Webcam inicializada com sucesso!")
    print("🎥 Iniciando detecção em tempo real...\n")
    
    screenshot_count = 0
    
    try:
        while True:
            # Capturar frame
            success, frame = cap.read()
            
            if not success:
                print("❌ Erro ao capturar frame da webcam")
                break
            
            # Espelhar frame horizontalmente para melhor UX
            frame = cv2.flip(frame, 1)
            
            # Processar frame
            processed_frame, detections = detector.process_frame(frame)
            
            # Mostrar frame
            cv2.imshow('MediaPipe - Rock Paper Scissors', processed_frame)
            
            # Capturar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Encerrando aplicação...")
                break
            elif key == ord('s'):
                # Salvar screenshot
                screenshot_count += 1
                filename = f'results/mediapipe/screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Screenshot salvo: {filename}")
            elif key == ord('r'):
                # Resetar estatísticas
                detector.total_detections = 0
                detector.gesture_counts = {'Rock': 0, 'Paper': 0, 'Scissors': 0}
                print("🔄 Estatísticas resetadas")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrompido pelo usuário")
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        # Mostrar estatísticas finais
        print("\n" + "=" * 60)
        print("📊 ESTATÍSTICAS FINAIS")
        print("=" * 60)
        print(f"Total de detecções: {detector.total_detections}")
        if detector.total_detections > 0:
            for gesture, count in detector.gesture_counts.items():
                percentage = count / detector.total_detections * 100
                print(f"  {gesture}: {count} ({percentage:.1f}%)")
        print("=" * 60)
        print("\n✅ Aplicação encerrada com sucesso!")


if __name__ == "__main__":
    main()

