import json, copy

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_ide = copy.deepcopy(nb)

for c in nb_ide['cells']:
    # Fix load data cell - replace Colab drive with local path
    if c.get('id') == 'code-drive':
        c['source'] = [
            'import os\n',
            'import warnings\n',
            'warnings.filterwarnings("ignore")\n',
            '\n',
            '# Chinh lai duong dan cho dung voi may cua ban\n',
            'DATA_PATH = "./data.csv"\n',
            'SAVE_PATH = "./outputs/"\n',
            '\n',
            'os.makedirs(SAVE_PATH, exist_ok=True)\n',
            '\n',
            'df_raw = pd.read_csv(DATA_PATH)\n',
            'print(f"Loaded: {df_raw.shape}")\n',
            'print(f"Target distribution (raw):")\n',
            'print(df_raw["Diabetes_binary"].value_counts())\n',
            'df_raw.head()\n'
        ]
    # Clear all outputs for clean notebook
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open('main_IDE.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_ide, f, ensure_ascii=False, indent=1)

print('main_IDE.ipynb created successfully')
