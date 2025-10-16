"""
Script de Verificação de Instalação e Configuração

Este script verifica se todas as dependências estão instaladas corretamente
e se o sistema está pronto para executar o projeto.

Autor: Projeto de Visão Computacional
"""

import sys
import subprocess


def print_header(text):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_python_version():
    """Verifica versão do Python."""
    print("\n🐍 Verificando Python...")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"   Versão instalada: Python {version_str}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Versão compatível (3.8+)")
        return True
    else:
        print("   ❌ Versão incompatível! Requer Python 3.8 ou superior")
        return False


def check_package(package_name, import_name=None):
    """Verifica se um pacote está instalado."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name}: {version}")
        return True, version
    except ImportError:
        print(f"   ❌ {package_name}: NÃO INSTALADO")
        return False, None


def check_packages():
    """Verifica todos os pacotes necessários."""
    print("\n📦 Verificando Pacotes Python...")
    
    packages = [
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
        ('mediapipe', 'mediapipe'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('Pillow', 'PIL'),
    ]
    
    results = {}
    all_installed = True
    
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        results[package_name] = (installed, version)
        if not installed:
            all_installed = False
    
    return all_installed, results


def check_gpu():
    """Verifica disponibilidade de GPU."""
    print("\n🎮 Verificando GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"   ✅ GPU detectada: {gpu_name}")
            print(f"   📊 Quantidade: {gpu_count}")
            print(f"   💾 Memória: {memory:.2f} GB")
            print(f"   🔧 CUDA versão: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️  GPU não detectada - usando CPU")
            print("   💡 YOLOv8 será mais lento, mas MediaPipe funciona bem")
            return False
    except ImportError:
        print("   ❌ PyTorch não instalado")
        return False


def check_webcam():
    """Verifica disponibilidade de webcam."""
    print("\n📷 Verificando Webcam...")
    
    try:
        import cv2
        
        available_cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cams.append(i)
                cap.release()
        
        if available_cams:
            print(f"   ✅ Webcam(s) detectada(s): {available_cams}")
            return True, available_cams
        else:
            print("   ❌ Nenhuma webcam detectada!")
            print("   💡 Conecte uma webcam para usar as aplicações em tempo real")
            return False, []
    except Exception as e:
        print(f"   ❌ Erro ao verificar webcam: {e}")
        return False, []


def check_model():
    """Verifica se modelo YOLOv8 treinado existe."""
    print("\n🤖 Verificando Modelo YOLOv8...")
    
    from pathlib import Path
    
    model_path = Path('models/best.pt')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024**2
        print(f"   ✅ Modelo encontrado: {model_path}")
        print(f"   📊 Tamanho: {size_mb:.2f} MB")
        return True
    else:
        print(f"   ⚠️  Modelo não encontrado em: {model_path}")
        print("   💡 Treine o modelo usando o notebook do Colab primeiro")
        print("   💡 Ou baixe um modelo pré-treinado")
        return False


def check_dataset():
    """Verifica se dataset está presente."""
    print("\n📊 Verificando Dataset...")
    
    from pathlib import Path
    
    train_path = Path('train/images')
    valid_path = Path('valid/images')
    test_path = Path('test/images')
    data_yaml = Path('data.yaml')
    
    checks = {
        'train': train_path.exists(),
        'valid': valid_path.exists(),
        'test': test_path.exists(),
        'data.yaml': data_yaml.exists()
    }
    
    all_present = all(checks.values())
    
    if all_present:
        # Contar imagens
        train_count = len(list(train_path.glob('*.jpg')))
        valid_count = len(list(valid_path.glob('*.jpg')))
        test_count = len(list(test_path.glob('*.jpg')))
        total = train_count + valid_count + test_count
        
        print(f"   ✅ Dataset completo detectado!")
        print(f"   📊 Treino: {train_count} imagens")
        print(f"   📊 Validação: {valid_count} imagens")
        print(f"   📊 Teste: {test_count} imagens")
        print(f"   📊 Total: {total} imagens")
        return True
    else:
        print(f"   ⚠️  Dataset incompleto ou não encontrado")
        for name, present in checks.items():
            status = "✅" if present else "❌"
            print(f"   {status} {name}")
        print("   💡 Baixe o dataset do Roboflow ou Google Drive")
        return False


def check_folders():
    """Verifica estrutura de pastas."""
    print("\n📁 Verificando Estrutura de Pastas...")
    
    from pathlib import Path
    
    folders = ['notebooks', 'src', 'models', 'results', 'results/yolov8', 'results/mediapipe']
    
    all_exist = True
    for folder in folders:
        path = Path(folder)
        if path.exists():
            print(f"   ✅ {folder}/")
        else:
            print(f"   ❌ {folder}/ - FALTANDO")
            all_exist = False
    
    return all_exist


def generate_report(checks):
    """Gera relatório final."""
    print_header("RELATÓRIO FINAL")
    
    status = {
        'Python': checks['python'],
        'Pacotes': checks['packages'],
        'GPU': checks['gpu'],
        'Webcam': checks['webcam'][0],
        'Estrutura': checks['folders'],
        'Dataset': checks['dataset'],
        'Modelo YOLOv8': checks['model']
    }
    
    # Essenciais
    print("\n✨ Componentes Essenciais:")
    essentials = ['Python', 'Pacotes', 'Webcam', 'Estrutura']
    for item in essentials:
        status_icon = "✅" if status[item] else "❌"
        print(f"   {status_icon} {item}")
    
    # Opcionais
    print("\n⚡ Componentes Opcionais:")
    optional = ['GPU', 'Dataset', 'Modelo YOLOv8']
    for item in optional:
        status_icon = "✅" if status[item] else "⚠️ "
        print(f"   {status_icon} {item}")
    
    # Pode executar?
    can_run_mediapipe = all(status[k] for k in ['Python', 'Pacotes', 'Webcam'])
    can_run_yolo = can_run_mediapipe and status['Modelo YOLOv8']
    can_train = status['Dataset']
    
    print("\n" + "=" * 70)
    print("📋 O QUE VOCÊ PODE FAZER:")
    print("=" * 70)
    
    if can_run_mediapipe:
        print("\n✅ PRONTO para executar MediaPipe:")
        print("   python src/mediapipe_realtime.py")
    else:
        print("\n❌ NÃO PRONTO para MediaPipe")
        print("   Corrija os problemas acima primeiro")
    
    if can_run_yolo:
        print("\n✅ PRONTO para executar YOLOv8:")
        print("   python src/yolov8_inference.py")
    else:
        print("\n⚠️  NÃO PRONTO para YOLOv8")
        if not status['Modelo YOLOv8']:
            print("   Treine o modelo primeiro no Google Colab")
    
    if can_train:
        print("\n✅ PRONTO para treinar modelo")
        print("   Use o notebook notebooks/yolov8_training.ipynb no Colab")
    else:
        print("\n⚠️  Dataset não encontrado")
        print("   Baixe o dataset primeiro")
    
    print("\n" + "=" * 70)
    
    # Ações recomendadas
    if not all(status.values()):
        print("\n💡 AÇÕES RECOMENDADAS:")
        print("=" * 70)
        
        if not status['Pacotes']:
            print("1. Instale as dependências:")
            print("   pip install -r requirements.txt")
        
        if not status['Webcam']:
            print("2. Conecte uma webcam ao computador")
        
        if not status['Dataset']:
            print("3. Baixe o dataset:")
            print("   https://universe.roboflow.com/bispado/rock-paper-scissors-sxsw-zbvgm")
        
        if not status['Modelo YOLOv8']:
            print("4. Treine o modelo no Google Colab")
            print("   ou baixe um modelo pré-treinado")
        
        if not status['GPU']:
            print("5. (Opcional) Configure GPU para melhor performance do YOLOv8")
    else:
        print("\n🎉 TUDO PRONTO! Sistema 100% configurado!")
    
    print("\n" + "=" * 70)


def main():
    """Função principal."""
    
    print_header("VERIFICAÇÃO DE SISTEMA - Rock Paper Scissors CV")
    print("\nEste script verifica se seu sistema está pronto para executar o projeto.")
    
    checks = {}
    
    # Python
    checks['python'] = check_python_version()
    
    # Pacotes
    checks['packages'], package_details = check_packages()
    
    # GPU
    checks['gpu'] = check_gpu()
    
    # Webcam
    checks['webcam'] = check_webcam()
    
    # Estrutura
    checks['folders'] = check_folders()
    
    # Dataset
    checks['dataset'] = check_dataset()
    
    # Modelo
    checks['model'] = check_model()
    
    # Relatório final
    generate_report(checks)
    
    # Links úteis
    print("\n📚 LINKS ÚTEIS:")
    print("=" * 70)
    print("• README completo: README.md")
    print("• Guia rápido: QUICKSTART.md")
    print("• Solução de problemas: TROUBLESHOOTING.md")
    print("• Dataset: https://universe.roboflow.com/bispado/rock-paper-scissors-sxsw-zbvgm")
    print("• Documentação YOLOv8: https://docs.ultralytics.com/")
    print("• Documentação MediaPipe: https://developers.google.com/mediapipe")
    print("=" * 70)
    
    print("\n✅ Verificação concluída!\n")


if __name__ == "__main__":
    main()

