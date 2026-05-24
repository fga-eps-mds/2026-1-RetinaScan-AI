<div align="center">
    <p align="center">
    <img width="450" height="220" alt="image" src="https://github.com/user-attachments/assets/43b36c5a-8fd6-48f2-9926-2ab73e38dd58" />
    </p>

# 2026-1-RetinaScan-AI

Sistema de classificação de retinografias utilizando *Deep Learning*, com suporte a múltiplos modelos para experimentação e comparação utilizando o *dataset* RFMiD.


</div>

--- 

## Equipe

<div align="center">
  <table>
    <tr>
        <td align="center">
        <a href="http://github.com/andre-maia51">
            <img src="http://github.com/andre-maia51.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>André Maia</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/artrsousa1">
            <img src="http://github.com/artrsousa1.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Arthur Ribeiro</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/cqcoding">
            <img src="http://github.com/cqcoding.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Ceci Quaresma</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/EliasOliver21">
            <img src="http://github.com/EliasOliver21.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Elias Oliveira</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/cwtshh">
            <img src="http://github.com/cwtshh.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Gustavo Costa</b></sub>
        </a>
        </td>
    </tr>
    <tr>
        <td align="center">
        <a href="https://github.com/Angelicahaas">
            <img src="https://github.com/Angelicahaas.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Harleny Angelica</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/IderlanJ">
            <img src="http://github.com/IderlanJ.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Iderlan Junio</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/Natyrodrigues">
            <img src="http://github.com/Natyrodrigues.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Natália Rodrigues</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/vnsrz">
            <img src="http://github.com/vnsrz.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Vinicius Roriz</b></sub>
        </a>
        </td>
        <td align="center">
        <a href="https://github.com/yan-luca">
            <img src="http://github.com/yan-luca.png" width="100" height="100" style="border-radius: 50%; object-fit: cover;" alt=""/>
            <br /><sub><b>Yan Luca</b></sub>
        </a>
        </td>
    </tr>
  </table>
</div> 
 
---

## Tecnologias Utilizadas

- Python 3.12
- PyTorch
- TensorBoard
- Pandas
- NumPy
- Deep Learning
- RETFound
- RFMiD Dataset

---

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
> É necessário utilizar Python 3.12.

Para sair do ambiente virtual:

```bash
deactivate
```

### 2. Instalar dependências


```bash
pip install -r requirements.txt
pip install pandas
```

### Baixar o checkpoint RETFound-MAE

```bash
mkdir -p ./checkpoints
```

Baixe o modelo em:

>https://huggingface.co/YukunZhou/RETFound_mae_natureCFP

Coloque o arquivo `.pth` em:

```txt
./checkpoints/
```


## Dataset

O projeto utiliza o dataset:

- RFMiD (Retinal Fundus Multi-Disease Image Dataset)

Baixe o arquivo:

```txt
A. RFMiD_All_Classes_Dataset.zip
```

Após o download:

1. Extraia o `.zip`
2. Coloque a pasta na raiz do projeto

---

### Organização automática

Execute:

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

Execute:

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

```txt
http://localhost:6006
```


### Avaliação

Após o treino, o melhor será automaticamente salvo como:

```bash
./output_dir/retfound_mae_RFMiD_binary_finetune/checkpoint-best.pth
```

Esse checkpoint será carregado para avaliação final no final do treino (se `args.eval`) estiver habilitado ou se você rodar o script novamente com `--eval` .

Se quiser avaliar manualmente, basta adaptar `--task`  e `--resume` no main_finetune.py ou criar um pequeno script de avaliação.

---
## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.