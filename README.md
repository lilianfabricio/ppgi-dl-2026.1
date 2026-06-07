# ppgi-dl-2026.1 / dip-2026.1

# Segmentação Semântica — Oxford-IIIT Pet

Pipeline completo: treino UNet → métricas (IoU/acurácia) → export TorchScript FP32 → app Android.

## Estrutura

```
semantic-segmentation-pets/
├── src/                 # dataset, UNet, métricas
├── train/               # scripts de treino e export
├── android/             # app Android (PyTorch Mobile Lite)
├── data/                # download automático do Oxford-IIIT Pet
├── checkpoints/         # pesos treinados
└── outputs/             # visualizações geradas
```

## Ambiente (treino)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Passo a passo

```bash
# 1. Treinar (subset de 500 imagens train / 100 val)
python train/train.py

# 2. Avaliar IoU e acurácia
python train/evaluate.py

# 3. Visualizar predições (Python)
python train/visualize.py

# 4. Exportar modelo FP32 para Android
python train/export_model.py
```

O export copia `model.pt` para `android/app/src/main/assets/`.

## App Android

1. Abra a pasta `android/` no Android Studio.
2. Conecte o smartphone ou use emulador (API 24+).
3. Build & Run.
4. Galeria/Câmera → selecionar imagem → **Inferir** → máscara sobreposta.

Requisitos do enunciado atendidos:
- Seleção via foto/galeria
- ImageView original + máscara segmentada
- Botão de inferência
- Execução em FP32 (TorchScript + `LiteModuleLoader`)

## Entregáveis (Classroom)

1. Print do app com segmentação funcionando
2. Link deste repositório no GitHub

## Hiperparâmetros

Edite `train/config.py` para ajustar subset, épocas, batch size e tamanho da imagem (padrão: 256×256).
