### First start

```bash
python -m venv venv
```

### Activate venv and install the nessesary libraries

```bash
source venv/Script/activate

pip install -r requirements.txt
```

### Workflow

0 -> 0_data_preparation -> run the pythons script to garner and prepare data
1 -> use haar detection -> evaluate the result
2 -> use yolo detection -> evaluate the result
3 -> Create report -> finish