# 2026-1-RetinaScan-AI

Repositório do projeto RetinaScan AI em 2026.1, usando o modelo RETFound‑MAE para fine‑tune em imagens de retina (RFMiD).

---

## Links
Demo (se disponível):
Preview: Não implantado ainda
Dashboard de treino: Verifique a pasta ./output_logs após executar o fine‑tune.

---

## Instalação

Criar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

Instalar dependências

```bash
pip install -r requirements.txt
```

### Baixar o checkpoint RETFound‑MAE

```bash
mkdir -p ./checkpoints
```

Baixe o modelo em: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP

Cole o arquivo `.pth` em `checkpoints`

---

## Configuração do dataset

Organize o dataset RFMiD no formato binário usando o script em `./scripts`:

```bash
python3 ./scripts/organizar_RFMID.py
```

Estrutura esperada:

```
../rfmid_binary/
  train/
    0/
    1/
  val/
    0/
    1/
  test/
    0/
    1/
```

---

## Treinamento (fine‑tune)

Certifique‑se de estar na raiz do repositório

Rode o fine‑tune em 1 GPU:

```bash
sh train.sh
```

Certifique‑se de que `DATA_PATH` no `train.sh` aponta para o seu dataset organizado (por exemplo, `../rfmid_binary` ou um symlink `.rfmid_binary`).

Logs e checkpoints serão salvos em:

- ./output_dir/retfound_mae_RFMiD_binary_finetune/

- ./output_logs/retfound_mae_RFMiD_binary_finetune/

---

## Executar TensorBoard

```bash
tensorboard --logdir ./output_logs
```

Acesse em: `http://localhost:6006`

---

## Avaliação

Após o treino, o melhor modelo será automaticamente salvo como:

```
./output_dir/retfound_mae_RFMiD_binary_finetune/checkpoint-best.pth
```

Esse checkpoint será carregado para avaliação final no final do treino (se `args.eval` estiver habilitado ou se você rodar o script novamente com `--eval`).

Se quiser avaliar manualmente, basta adaptar --task e --resume no main_finetune.py ou criar um pequeno script de avaliação.
