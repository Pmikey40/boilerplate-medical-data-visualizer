import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar datos
df = pd.read_csv("medical_examination.csv")

# 2. Agregar columna overweight (IMC > 25)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalizar cholesterol y gluc (0 = bueno, 1 = malo)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# -----------------------------
# 📊 GRÁFICO CATEGÓRICO
# -----------------------------
def draw_cat_plot():
    # 4. Melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 5. Agrupar
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']) \
                   .size() \
                   .reset_index(name='total')

    # 6. Catplot
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    ).fig

    # Guardar imagen
    fig.savefig('catplot.png')
    return fig


# -----------------------------
# 🔥 HEATMAP
# -----------------------------
def draw_heat_map():
    # 7. Limpiar datos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 8. Correlación
    corr = df_heat.corr()

    # 9. Máscara triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 10. Figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # 11. Heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    fig.savefig('heatmap.png')
    return fig