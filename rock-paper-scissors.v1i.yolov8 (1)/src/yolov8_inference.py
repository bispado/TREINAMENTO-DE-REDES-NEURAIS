"""
Aplicação de Inferência YOLOv8 em Tempo Real para Rock Paper Scissors

Este script carrega um modelo YOLOv8 treinado e realiza inferência em tempo real
usando a webcam para detectar gestos de Pedra, Papel e Tesoura.

Autor: Projeto de Visão Computacional
Data: 2025
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import argparse


class YOLOv8Detector:
    """
    Detector de gestos Rock, Paper, Scissors usando YOLOv8 treinado.
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Inicializa o detector YOLOv8.
        
        Args:
            model_path: Caminho para o modelo treinado (.pt)
            conf_threshold: Threshold de confiança para detecções
            iou_threshold: Threshold de IoU para NMS
        """
        print(f"📦 Carregando modelo YOLOv8: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Cores para cada classe (BGR)
        self.colors = {
            'Paper': (0, 255, 0),      # Verde
            'Rock': (0, 0, 255),       # Vermelho
            'Scissors': (255, 0, 0),   # Azul
        }
        
        # Estatísticas
        self.total_detections = 0
        self.class_counts = {'Paper': 0, 'Rock': 0, 'Scissors': 0}
        
        # Para FPS
        self.prev_time = 0
        self.fps_history = []
        
        print("✅ Modelo carregado com sucesso!")
        
    def process_frame(self, frame):
        """
        Processa um frame de vídeo para detectar gestos.
        
        Args:
            frame: Frame de vídeo (BGR)
            
        Returns:
            Frame processado com anotações e lista de detecções
        """
        # Medir tempo de inferência
        start_time = time.time()
        
        # Realizar predição
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Lista de detecções
        detections = []
        
        # Processar detecções
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            
            for box in boxes:
                # Extrair informações
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                # Atualizar estatísticas
                self.total_detections += 1
                if class_name in self.class_counts:
                    self.class_counts[class_name] += 1
                
                # Cor da classe
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Desenhar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{class_name} {confidence:.2f}"
                
                # Fundo para o texto
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
        
        # Adicionar informações na tela
        # FPS e tempo de inferência
        info_y = 30
        cv2.rectangle(frame, (5, 5), (350, info_y * 2 + 10), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f} | Inference: {inference_time:.1f}ms", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Estatísticas de classes
        y_offset = 90
        cv2.rectangle(frame, (5, 70), (300, y_offset + len(self.class_counts) * 30), 
                     (0, 0, 0), -1)
        cv2.putText(frame, f"Total: {self.total_detections}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for class_name, count in self.class_counts.items():
            percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
            text = f"{class_name}: {count} ({percentage:.1f}%)"
            color = self.colors.get(class_name, (255, 255, 255))
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Instruções
        cv2.putText(frame, "q: Sair | s: Screenshot | r: Reset | +/-: Ajustar confianca", 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame, detections, inference_time
    
    def adjust_confidence(self, delta):
        """Ajusta o threshold de confiança."""
        self.conf_threshold = max(0.1, min(0.9, self.conf_threshold + delta))
        print(f"🎯 Confiança ajustada para: {self.conf_threshold:.2f}")


def main():
    """Função principal para executar a aplicação."""
    
    # Argumentos de linha de comando
    parser = argparse.ArgumentParser(description='YOLOv8 Real-time Inference for Rock Paper Scissors')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Caminho para o modelo YOLOv8 treinado')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Threshold de confiança (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='Threshold de IoU para NMS')
    parser.add_argument('--source', type=int, default=0,
                       help='Índice da webcam (0, 1, 2, ...)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Largura do vídeo')
    parser.add_argument('--height', type=int, default=720,
                       help='Altura do vídeo')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLOv8 - Detecção de Rock Paper Scissors em Tempo Real")
    print("=" * 70)
    print(f"\nConfigurações:")
    print(f"  Modelo: {args.model}")
    print(f"  Confiança: {args.conf}")
    print(f"  IoU: {args.iou}")
    print(f"  Webcam: {args.source}")
    print(f"  Resolução: {args.width}x{args.height}")
    print("\nInstruções:")
    print("  - Mostre sua mão para a câmera fazendo os gestos")
    print("  - Pressione 'q' para sair")
    print("  - Pressione 's' para salvar screenshot")
    print("  - Pressione 'r' para resetar estatísticas")
    print("  - Pressione '+' ou '-' para ajustar confiança")
    print("=" * 70 + "\n")
    
    # Verificar se modelo existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Erro: Modelo não encontrado em {model_path}")
        print("\n💡 Dica: Treine o modelo primeiro usando o notebook ou baixe um modelo pré-treinado.")
        print("   Coloque o arquivo best.pt na pasta 'models/'")
        return
    
    # Inicializar detector
    try:
        detector = YOLOv8Detector(
            model_path=str(model_path),
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Inicializar webcam
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"❌ Erro: Não foi possível abrir a webcam {args.source}!")
        return
    
    # Configurar resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Verificar resolução real
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Webcam inicializada: {actual_width}x{actual_height}")
    print("🎥 Iniciando inferência em tempo real...\n")
    
    screenshot_count = 0
    inference_times = []
    
    try:
        while True:
            # Capturar frame
            success, frame = cap.read()
            
            if not success:
                print("❌ Erro ao capturar frame da webcam")
                break
            
            # Espelhar frame horizontalmente
            frame = cv2.flip(frame, 1)
            
            # Processar frame
            processed_frame, detections, inference_time = detector.process_frame(frame)
            inference_times.append(inference_time)
            
            # Mostrar frame
            cv2.imshow('YOLOv8 - Rock Paper Scissors Detection', processed_frame)
            
            # Capturar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Encerrando aplicação...")
                break
            elif key == ord('s'):
                # Salvar screenshot
                screenshot_count += 1
                filename = f'results/yolov8/screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Screenshot salvo: {filename}")
            elif key == ord('r'):
                # Resetar estatísticas
                detector.total_detections = 0
                detector.class_counts = {'Paper': 0, 'Rock': 0, 'Scissors': 0}
                inference_times = []
                print("🔄 Estatísticas resetadas")
            elif key == ord('+') or key == ord('='):
                detector.adjust_confidence(0.05)
            elif key == ord('-') or key == ord('_'):
                detector.adjust_confidence(-0.05)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrompido pelo usuário")
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar estatísticas finais
        print("\n" + "=" * 70)
        print("📊 ESTATÍSTICAS FINAIS - YOLOv8")
        print("=" * 70)
        print(f"Total de detecções: {detector.total_detections}")
        
        if detector.total_detections > 0:
            print("\nDetecções por classe:")
            for class_name, count in detector.class_counts.items():
                percentage = count / detector.total_detections * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if inference_times:
            avg_inference = sum(inference_times) / len(inference_times)
            min_inference = min(inference_times)
            max_inference = max(inference_times)
            print(f"\nPerformance:")
            print(f"  Tempo médio de inferência: {avg_inference:.2f}ms")
            print(f"  Tempo mínimo: {min_inference:.2f}ms")
            print(f"  Tempo máximo: {max_inference:.2f}ms")
            print(f"  FPS médio: {1000/avg_inference:.1f}")
        
        print("=" * 70)
        print("\n✅ Aplicação encerrada com sucesso!")


if __name__ == "__main__":
    main()

