# 2026-1-RetinaScan-AI

Repositório do projeto RetinaScan AI em 2026.1, que tem como objetivo classificação de imagens de retina utilizando *Deep Learning,* com suporte a múltiplos modelos para experimentação e comparação no *dataset* RFMiD.

## Pipeline do projeto

1. Preparar ambiente
2. Organizar dataset
3. Treinar modelo
4. Avaliar resultados
5. Visualizar logs

## Instalação

### 1. Criar ambiente virtual

```bash
python3.12 -m venv venv
source venv/bin/activate
```

**Obs:** **É necessário utilizar o Python 3.12**

Se quiser sair do ambiente virtual utilize o comando:

```bash
deactivate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

```bash
pip install pandas
```

### Baixar o checkpoint RETFound-MAE

```bash
mkdir -p ./checkpoints
```

Baixe o modelo em: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP

Coloque o arquivo `.pth` em:

`./checkpoints/` 

## Dataset

Utilizamos o dataset **RFMiD** (Retinal Fundus Multi-Disease Image Dataset)

Será necessário baixar o dataset:

`A. RFMiD_All_Classes_Dataset.zip` 

Após baixar este aquivo `.zip` , será preciso extrair e colocar a pasta na raiz do projeto. 

### Organização automática

Após adicionar a pasta extraída a raiz do projeto, no seu terminal rode o comando:

```bash
cd scripts/
```

Dentro desta pasta execute o script:

```bash
python3 ./organizar_RFMID.py
```

### Estrutura esperada

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

### Configuração

Antes de treinar, revise:

| Parâmetro | Descrição |
| --- | --- |
| `DATA_PATH` | Caminho do dataset |
| `OUTPUT_DIR` | Diretório de saída |
| `BATCH_SIZE` | Depende da GPU |
| `DEVICE` | `cuda` ou `cpu`  |

### Treinamento (fine-tune)

Certifique-se de estar na raiz do repositório

Rode o fine-tune em GPU:

```bash
sh train.sh
```

Certifique-se de que `DATA_PATH` no `train.sh` aponta para o seu dataset organizado (`../rfmid_binary` ou um symlink `.rfmid_binary` ).

### Parâmetros úteis

```yaml
 --batch_size 16
 --epochs 50
 --lr 1e-4
 --eval
```

### Logs e resultados

Os resultados são salvos em: 

- ./output_dir/retfound_mae_RFMiD_binary_finetune/
- ./output_logs/retfound_mae_RFMiD_binary_finetune/

### Executar o TensorBoard

O TensorBoard é uma ferramenta de visualização que permite acompanhar o progresso do treinamento em tempo real, incluindo métricas como loss, acurácia, gráficos e evolução dos parâmetros do modelo.

Execute: 

```bash
tensorboard --logdir ./output_logs
```

Acesse: 

`http://localhost:6006` 

### Avaliação

Após o treino, o melhor será automaticamente salvo como:

```bash
./output_dir/retfound_mae_RFMiD_binary_finetune/checkpoint-best.pth
```

Esse checkpoint será carregado para avaliação final no final do treino (se `args.eval`) estiver habilitado ou se você rodar o script novamente com `--eval` ).

Se quiser avaliar manualmente, basta adaptar `--task`  e `--resume` no main_finetune.py ou criar um pequeno script de avaliação.