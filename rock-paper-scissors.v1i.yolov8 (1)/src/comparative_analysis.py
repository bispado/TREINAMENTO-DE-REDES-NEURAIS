"""
Script de Análise Comparativa entre YOLOv8 e MediaPipe

Este script executa testes de performance e gera relatórios comparativos
entre as duas abordagens de detecção de Rock Paper Scissors.

Autor: Projeto de Visão Computacional
Data: 2025
"""

import cv2
import time
import numpy as np
from mediapipe_realtime import RockPaperScissorsDetector
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


def test_mediapipe_performance(video_source=0, duration=30):
    """
    Testa a performance do MediaPipe por um período específico.
    
    Args:
        video_source: Fonte de vídeo (webcam)
        duration: Duração do teste em segundos
        
    Returns:
        Dicionário com estatísticas de performance
    """
    print("\n🔵 Testando MediaPipe...")
    print(f"Duração do teste: {duration}s")
    
    detector = RockPaperScissorsDetector()
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir webcam")
        return None
    
    # Estatísticas
    frame_count = 0
    detection_count = 0
    inference_times = []
    fps_values = []
    gesture_detections = []
    
    start_time = time.time()
    prev_time = start_time
    
    print("🎥 Capturando frames...")
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed >= duration:
            break
        
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Medir tempo de inferência
        infer_start = time.time()
        processed_frame, detections = detector.process_frame(frame)
        infer_time = (time.time() - infer_start) * 1000
        
        # Coletar estatísticas
        frame_count += 1
        inference_times.append(infer_time)
        
        if detections:
            detection_count += len(detections)
            for det in detections:
                gesture_detections.append(det['gesture'])
        
        # Calcular FPS
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        fps_values.append(fps)
        prev_time = current_time
        
        # Mostrar progresso
        if frame_count % 30 == 0:
            print(f"⏱️  {elapsed:.1f}s / {duration}s - {frame_count} frames")
    
    cap.release()
    detector.release()
    
    total_time = time.time() - start_time
    
    # Calcular estatísticas
    stats = {
        'method': 'MediaPipe',
        'duration': total_time,
        'frame_count': frame_count,
        'detection_count': detection_count,
        'avg_fps': np.mean(fps_values),
        'min_fps': np.min(fps_values),
        'max_fps': np.max(fps_values),
        'avg_inference_time': np.mean(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'std_inference_time': np.std(inference_times),
        'detection_rate': (detection_count / frame_count) * 100,
        'gesture_distribution': {
            'Rock': gesture_detections.count('Rock'),
            'Paper': gesture_detections.count('Paper'),
            'Scissors': gesture_detections.count('Scissors'),
        }
    }
    
    print("\n✅ Teste MediaPipe concluído!")
    return stats


def test_yolov8_performance(model_path, video_source=0, duration=30):
    """
    Testa a performance do YOLOv8 por um período específico.
    
    Args:
        model_path: Caminho do modelo YOLOv8
        video_source: Fonte de vídeo
        duration: Duração do teste em segundos
        
    Returns:
        Dicionário com estatísticas de performance
    """
    print("\n🟢 Testando YOLOv8...")
    print(f"Modelo: {model_path}")
    print(f"Duração do teste: {duration}s")
    
    # Verificar se modelo existe
    if not Path(model_path).exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ Erro ao abrir webcam")
        return None
    
    # Estatísticas
    frame_count = 0
    detection_count = 0
    inference_times = []
    fps_values = []
    class_detections = []
    confidence_values = []
    
    start_time = time.time()
    prev_time = start_time
    
    print("🎥 Capturando frames...")
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed >= duration:
            break
        
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Medir tempo de inferência
        infer_start = time.time()
        results = model.predict(frame, conf=0.25, verbose=False)[0]
        infer_time = (time.time() - infer_start) * 1000
        
        # Coletar estatísticas
        frame_count += 1
        inference_times.append(infer_time)
        
        if results.boxes is not None and len(results.boxes) > 0:
            detection_count += len(results.boxes)
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                confidence = float(box.conf[0])
                class_detections.append(class_name)
                confidence_values.append(confidence)
        
        # Calcular FPS
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        fps_values.append(fps)
        prev_time = current_time
        
        # Mostrar progresso
        if frame_count % 30 == 0:
            print(f"⏱️  {elapsed:.1f}s / {duration}s - {frame_count} frames")
    
    cap.release()
    
    total_time = time.time() - start_time
    
    # Calcular estatísticas
    stats = {
        'method': 'YOLOv8',
        'duration': total_time,
        'frame_count': frame_count,
        'detection_count': detection_count,
        'avg_fps': np.mean(fps_values),
        'min_fps': np.min(fps_values),
        'max_fps': np.max(fps_values),
        'avg_inference_time': np.mean(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'std_inference_time': np.std(inference_times),
        'detection_rate': (detection_count / frame_count) * 100,
        'avg_confidence': np.mean(confidence_values) if confidence_values else 0,
        'class_distribution': {
            'Rock': class_detections.count('Rock'),
            'Paper': class_detections.count('Paper'),
            'Scissors': class_detections.count('Scissors'),
        }
    }
    
    print("\n✅ Teste YOLOv8 concluído!")
    return stats


def generate_comparison_report(mediapipe_stats, yolov8_stats, output_dir='results'):
    """
    Gera relatório comparativo detalhado.
    """
    print("\n📊 Gerando relatório comparativo...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Salvar estatísticas em JSON
    report_data = {
        'mediapipe': mediapipe_stats,
        'yolov8': yolov8_stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    json_path = output_path / 'comparative_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Dados salvos em: {json_path}")
    
    # Gerar gráficos
    generate_comparison_charts(mediapipe_stats, yolov8_stats, output_path)
    
    # Gerar relatório texto
    generate_text_report(mediapipe_stats, yolov8_stats, output_path)
    
    print("✅ Relatório completo gerado!")


def generate_comparison_charts(mp_stats, yolo_stats, output_dir):
    """Gera gráficos comparativos."""
    
    sns.set_style("whitegrid")
    
    # Gráfico 1: FPS Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # FPS
    methods = ['MediaPipe', 'YOLOv8']
    fps_avg = [mp_stats['avg_fps'], yolo_stats['avg_fps']]
    
    axes[0, 0].bar(methods, fps_avg, color=['#FF6F00', '#00D4FF'])
    axes[0, 0].set_title('FPS Médio', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('FPS')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Tempo de Inferência
    inference_avg = [mp_stats['avg_inference_time'], yolo_stats['avg_inference_time']]
    
    axes[0, 1].bar(methods, inference_avg, color=['#FF6F00', '#00D4FF'])
    axes[0, 1].set_title('Tempo de Inferência Médio', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Tempo (ms)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Taxa de Detecção
    detection_rate = [mp_stats['detection_rate'], yolo_stats['detection_rate']]
    
    axes[1, 0].bar(methods, detection_rate, color=['#FF6F00', '#00D4FF'])
    axes[1, 0].set_title('Taxa de Detecção', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('% de Frames')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Detecções Totais
    total_detections = [mp_stats['detection_count'], yolo_stats['detection_count']]
    
    axes[1, 1].bar(methods, total_detections, color=['#FF6F00', '#00D4FF'])
    axes[1, 1].set_title('Total de Detecções', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Quantidade')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / 'comparative_charts.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📈 Gráficos salvos em: {chart_path}")
    plt.close()


def generate_text_report(mp_stats, yolo_stats, output_dir):
    """Gera relatório em texto."""
    
    report = f"""
{'='*80}
RELATÓRIO COMPARATIVO - YOLOV8 vs MEDIAPIPE
Rock Paper Scissors Detection
{'='*80}

Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
MEDIAPIPE
{'='*80}

Performance:
  • FPS Médio: {mp_stats['avg_fps']:.2f}
  • FPS Mín/Máx: {mp_stats['min_fps']:.2f} / {mp_stats['max_fps']:.2f}
  • Inferência Média: {mp_stats['avg_inference_time']:.2f}ms
  • Inferência Mín/Máx: {mp_stats['min_inference_time']:.2f}ms / {mp_stats['max_inference_time']:.2f}ms
  • Desvio Padrão: {mp_stats['std_inference_time']:.2f}ms

Detecções:
  • Total de Frames: {mp_stats['frame_count']}
  • Total de Detecções: {mp_stats['detection_count']}
  • Taxa de Detecção: {mp_stats['detection_rate']:.1f}%

Distribuição de Gestos:
  • Rock: {mp_stats['gesture_distribution']['Rock']}
  • Paper: {mp_stats['gesture_distribution']['Paper']}
  • Scissors: {mp_stats['gesture_distribution']['Scissors']}

{'='*80}
YOLOV8
{'='*80}

Performance:
  • FPS Médio: {yolo_stats['avg_fps']:.2f}
  • FPS Mín/Máx: {yolo_stats['min_fps']:.2f} / {yolo_stats['max_fps']:.2f}
  • Inferência Média: {yolo_stats['avg_inference_time']:.2f}ms
  • Inferência Mín/Máx: {yolo_stats['min_inference_time']:.2f}ms / {yolo_stats['max_inference_time']:.2f}ms
  • Desvio Padrão: {yolo_stats['std_inference_time']:.2f}ms
  • Confiança Média: {yolo_stats['avg_confidence']:.1%}

Detecções:
  • Total de Frames: {yolo_stats['frame_count']}
  • Total de Detecções: {yolo_stats['detection_count']}
  • Taxa de Detecção: {yolo_stats['detection_rate']:.1f}%

Distribuição de Classes:
  • Rock: {yolo_stats['class_distribution']['Rock']}
  • Paper: {yolo_stats['class_distribution']['Paper']}
  • Scissors: {yolo_stats['class_distribution']['Scissors']}

{'='*80}
COMPARAÇÃO
{'='*80}

FPS:
  • MediaPipe: {mp_stats['avg_fps']:.2f} FPS
  • YOLOv8: {yolo_stats['avg_fps']:.2f} FPS
  • Diferença: {((mp_stats['avg_fps'] - yolo_stats['avg_fps']) / yolo_stats['avg_fps'] * 100):+.1f}%
  • Vencedor: {'MediaPipe' if mp_stats['avg_fps'] > yolo_stats['avg_fps'] else 'YOLOv8'} 🏆

Tempo de Inferência:
  • MediaPipe: {mp_stats['avg_inference_time']:.2f}ms
  • YOLOv8: {yolo_stats['avg_inference_time']:.2f}ms
  • Diferença: {((yolo_stats['avg_inference_time'] - mp_stats['avg_inference_time']) / mp_stats['avg_inference_time'] * 100):+.1f}%
  • Mais Rápido: {'MediaPipe' if mp_stats['avg_inference_time'] < yolo_stats['avg_inference_time'] else 'YOLOv8'} 🏆

Taxa de Detecção:
  • MediaPipe: {mp_stats['detection_rate']:.1f}%
  • YOLOv8: {yolo_stats['detection_rate']:.1f}%
  • Diferença: {(mp_stats['detection_rate'] - yolo_stats['detection_rate']):+.1f}pp
  • Maior: {'MediaPipe' if mp_stats['detection_rate'] > yolo_stats['detection_rate'] else 'YOLOv8'} 🏆

{'='*80}
CONCLUSÕES
{'='*80}

Velocidade: MediaPipe é geralmente mais rápido em CPU.
Precisão: YOLOv8 tende a ter maior precisão após treinamento adequado.
Recursos: MediaPipe consome menos recursos computacionais.
Flexibilidade: YOLOv8 é mais flexível para diferentes objetos.

Recomendação:
- Para produção com recursos limitados: MediaPipe
- Para máxima precisão com GPU: YOLOv8
- Para prototipagem rápida: MediaPipe

{'='*80}
"""
    
    report_path = output_dir / 'comparative_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Relatório salvo em: {report_path}")
    print(report)


def main():
    """Função principal."""
    
    print("=" * 80)
    print("ANÁLISE COMPARATIVA: YOLOv8 vs MediaPipe")
    print("=" * 80)
    print("\nEste script irá testar ambas as abordagens e gerar relatórios comparativos.")
    print("\n⚠️  IMPORTANTE:")
    print("  - Certifique-se de ter uma webcam conectada")
    print("  - Para YOLOv8, o modelo treinado deve estar em 'models/best.pt'")
    print("  - Mantenha condições consistentes durante os testes")
    print("  - Mostre gestos variados para análise completa")
    
    input("\nPressione ENTER para continuar...")
    
    # Configurações
    test_duration = 30  # segundos
    yolo_model_path = 'models/best.pt'
    
    # Teste 1: MediaPipe
    mp_stats = test_mediapipe_performance(duration=test_duration)
    
    if mp_stats is None:
        print("❌ Teste do MediaPipe falhou!")
        return
    
    print("\n" + "=" * 80)
    input("Pressione ENTER para iniciar teste do YOLOv8...")
    
    # Teste 2: YOLOv8
    yolo_stats = test_yolov8_performance(yolo_model_path, duration=test_duration)
    
    if yolo_stats is None:
        print("❌ Teste do YOLOv8 falhou!")
        print("💡 Certifique-se de ter treinado o modelo primeiro.")
        return
    
    # Gerar relatório
    generate_comparison_report(mp_stats, yolo_stats)
    
    print("\n" + "=" * 80)
    print("✅ Análise comparativa concluída com sucesso!")
    print("📁 Verifique a pasta 'results/' para os relatórios gerados.")
    print("=" * 80)


if __name__ == "__main__":
    main()

