# # Datos de juegos.
# ## Importar datos y corregir valores de las columnas. 

# In[1]:


import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


# In[2]:


games_data = pd.read_csv('/datasets/games.csv')


# <div class="alert alert-block alert-warning">
# <b>Comentario revisor</b> <a class="tocSkip"></a>
# 
# 
# Te recomiendo que como buena práctica separes la carga de las librerías y la carga de las bases de datos. 
# </div>

# In[3]:


games_data.columns = games_data.columns.str.lower()
games_data['year_of_release'] = pd.to_datetime(games_data['year_of_release'], format='%Y')
display(games_data.head(5))


# El dataframe ha sido modificado para lograr una mejor manera de trabajarlos, los nombres de las columnas han sido cambiadas a un formato mas simple de nombrarlos y la columna "year_of_release" fue transfomada mediante "datetime" para obtener un formato de "fecha" y asi mejorar el entorno de datos obtenidos.

# In[4]:


games_data.info()


# In[5]:


games_data['user_score'] = pd.to_numeric(games_data['user_score'], errors='coerce')

# Completar los valores NaN en 'critic_score' con 'Unknown'
games_data['critic_score_filled'] = games_data['critic_score'].fillna('Unknown')

# Verificar si hay otras columnas con datos ausentes
missing_values = games_data.isnull().sum()
print("Valores ausentes en otras columnas:")
print(missing_values[missing_values > 0])


# ### Describir las columnas con tipos de datos cambiados y explicar por qué
# 'user_score': Se convierte a numérico para facilitar cálculos y análisis estadísticos.
# 
# ### Tratar los valores ausentes
# Para 'year_of_release' y 'user_score', NaN podría significar que no se dispone de la información.
# Se pueden dejar los valores ausentes ya que imputarlos podría introducir sesgos.
# Para la columna 'rating', NaN podría significar que el juego no fue clasificado por ESRB.

# In[6]:


# Manejar los valores 'TBD'
games_data['user_score'] = games_data['user_score'].replace('tbd', pd.NA)
games_data['user_score'] = pd.to_numeric(games_data['user_score'], errors='coerce')


# In[7]:


# Calcular las ventas totales
games_data['total_sales'] = games_data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

# Mostrar una vista previa de los datos procesados
display(games_data.head(10))

# In[8]:


# Filtrar datos relevantes para el período de interés (por ejemplo, 2014-2016)
relevant_data = games_data[games_data['year_of_release'].between('1985', '2016',  inclusive = True)]

display(relevant_data.head())


# In[9]:


# Observar las ventas de una plataforma a otra
platform_sales = relevant_data.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print("Ventas totales por plataforma:")
display(platform_sales)

# Calcular las ventas totales por plataforma
platform_sales = games_data.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
print("Plataformas líderes en ventas:")
print(platform_sales.head())

# Contar el número de juegos lanzados en diferentes años
games_per_year = relevant_data.groupby('year_of_release')['name'].count()

# Visualizar el número de juegos lanzados por año
plt.figure(figsize=(10, 6))
sns.barplot(x=games_per_year.index, y=games_per_year.values, palette='viridis')
plt.title('Número de Juegos Lanzados por Año (1985-2016)')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Número de Juegos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Análisis de datos obtenidos
# Basándonos en las ventas totales por plataforma, podemos identificar las cinco plataformas líderes en ventas, que son:
# 
# -PS2: 1233.56 millones de unidades vendidas
# -Xbox 360 (X360): 961.24 millones de unidades vendidas
# -PlayStation 3 (PS3): 931.34 millones de unidades vendidas
# -Wii: 891.18 millones de unidades vendidas
# -Nintendo DS (DS): 802.78 millones de unidades vendidas
# -Estas son las plataformas que han acumulado las mayores ventas totales. Sin embargo, si nos centramos en las cinco principales, -hay una ligera discrepancia con los datos proporcionados en las ventas líderes:
# 
# -PS2: 1255.77 millones de unidades vendidas
# -Xbox 360 (X360): 971.42 millones de unidades vendidas
# -PlayStation 3 (PS3): 939.65 millones de unidades vendidas
# -Wii: 907.51 millones de unidades vendidas
# -Nintendo DS (DS): 806.12 millones de unidades vendidas
# 
# Se puede observar que hay una pequeña diferencia en las cifras de ventas totales, lo que puede deberse a la fuente de los datos o a la actualización de las ventas entre el momento en que se obtuvieron los datos y ahora. Aunque hay discrepancias en las cifras exactas, las plataformas principales siguen siendo las mismas en ambos casos, con PS2, Xbox 360, PS3, Wii y DS liderando las ventas.

# In[10]:


# Encontrar las plataformas que solían ser populares pero que ahora no tienen ventas
obsolete_platforms = platform_sales[platform_sales <= 10].index.tolist()
print("Plataformas obsoletas que ya no tienen ventas:")
print(obsolete_platforms)

# Seleccionar las plataformas con mayores ventas totales
top_platforms = platform_sales.head(5).index.tolist()

# Construir una distribución de ventas por año para las plataformas seleccionadas
plt.figure(figsize=(12, 8))
for platform in top_platforms:
    platform_data = relevant_data[relevant_data['platform'] == platform]
    platform_sales_by_year = platform_data.groupby('year_of_release')['total_sales'].sum()
    plt.plot(platform_sales_by_year.index, platform_sales_by_year.values, label=platform)

plt.title('Ventas de Plataformas Líderes por Año (1985-2016)')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Ventas Totales')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ## análisis de ventas por consola
# Las plataformas obsoletas que ya no tienen ventas son las siguientes:
# 
# Sega CD (SCD)
# Neo Geo (NG)
# WonderSwan (WS)
# TurboGrafx-16 (TG16)
# 3DO
# Game Gear (GG)
# PC-FX (PCFX)
# Estas plataformas ya no generan ventas significativas y pueden considerarse obsoletas en el mercado actual.
# 
# Según el gráfico de ventas las consolas que más ventas han obtenido en los años ha sido el ps2 y la nintendo wii pero su duración de popularidad ha decaido en la actualidad por lo que se han reemplazado por las consolas ps3 y xbox 360, que son las que han aumentado sus ventas a partir del 2008 hasta la actualidad (decallendo con el tiempo).

# In[11]:


# Calcular la vida útil promedio de las nuevas y antiguas plataformas
average_lifespan_new_platforms = relevant_data.groupby('platform')['year_of_release'].min().mean()
average_lifespan_old_platforms = relevant_data.groupby('platform')['year_of_release'].max().mean()
print("Tiempo promedio que tardan las nuevas plataformas en aparecer:", average_lifespan_new_platforms)
print("Tiempo promedio que tardan las antiguas plataformas en desaparecer:", average_lifespan_old_platforms)

# Identificar las plataformas líderes en ventas
top_selling_platforms = platform_sales.head(3).index.tolist()

# Crear un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma
plt.figure(figsize=(12, 8))
sns.boxplot(x='platform', y='total_sales', data=relevant_data, palette='Set3')
plt.title('Ventas Globales de Juegos por Plataforma (1985-2016)')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Globales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## ventas globales por plataforma y tiempo de vida util por consola.
# Muestra de análisis sobre el tiempo promedio que tardan las nuevas plataformas en aparecer y las antiguas plataformas en desaparecer. Aquí tienes los resultados:
# 
# Tiempo promedio que tardan las nuevas plataformas en aparecer: 1997-11-14 18:34:50.322580608
# Tiempo promedio que tardan las antiguas plataformas en desaparecer: 2005-04-04 17:48:23.225806464
# Esto sugiere que, en promedio, las nuevas plataformas tienden a aparecer alrededor de 1997 y las antiguas plataformas tienden a desaparecer alrededor de 2005. Sin embargo, ten en cuenta que estos valores son solo promedios y pueden variar dependiendo de varios factores, como la evolución de la tecnología, las estrategias de mercado de las empresas y las preferencias de los consumidores.

# In[12]:


# Calcular las ventas totales por género
genre_sales = games_data.groupby('genre')['total_sales'].sum().sort_values(ascending=False)
print("Ventas totales por género:")
print(genre_sales)


# In[13]:


# Creamos un DataFrame separado para cada región
na_sales = games_data[['name', 'na_sales', 'platform','rating']].rename(columns={'na_sales': 'sales'})
eu_sales = games_data[['name', 'eu_sales','platform','rating']].rename(columns={'eu_sales': 'sales'})
jp_sales = games_data[['name', 'jp_sales','platform','rating']].rename(columns={'jp_sales': 'sales'})

# Añadimos la columna 'region' a cada DataFrame
na_sales['region'] = 'NA'
eu_sales['region'] = 'EU'
jp_sales['region'] = 'JP'

# Concatenamos los DataFrames
region_data = pd.concat([na_sales, eu_sales, jp_sales])

print(region_data.head())


# In[14]:


# Calcular las ventas totales por plataforma para cada región
region_platform_sales = region_data.groupby(['region', 'platform'])['sales'].sum().unstack()

# Identificar las cinco plataformas principales en cada región
top_platforms_na = region_platform_sales.loc['NA'].sort_values(ascending=False).head(5)
top_platforms_eu = region_platform_sales.loc['EU'].sort_values(ascending=False).head(5)
top_platforms_jp = region_platform_sales.loc['JP'].sort_values(ascending=False).head(5)

print("Top 5 plataformas en Norteamérica:")
print(top_platforms_na)
print("\nTop 5 plataformas en Europa:")
print(top_platforms_eu)
print("\nTop 5 plataformas en Japón:")
print(top_platforms_jp)


# In[15]:


# Crear la gráfica de dispersión
plt.figure(figsize=(12, 8))
plt.scatter(games_data['user_score'], games_data['critic_score'], s= games_data['total_sales']*10, alpha=0.5)

# Agregar etiquetas y título
plt.title('Relación entre Puntuaciones de Usuarios y Críticos con Ventas Globales')
plt.xlabel('Puntuación de Usuarios')
plt.ylabel('Puntuación de Críticos')
plt.grid(True)

plt.show()


# Esta gráfica de dispersión muestra la relación entre las puntuaciones de usuarios y críticos en los ejes x e y respectivamente, mientras que el tamaño de los puntos está determinado por las ventas globales. Esto nos permite visualizar si hay alguna correlación entre las puntuaciones y las ventas globales de los juegos.

# ## Ventas por genero
# Basándonos en las ventas totales por género, podemos observar que los géneros más rentables son:
# 
# Acción (Action): 1744.17 millones de unidades vendidas
# Deportes (Sports): 1331.27 millones de unidades vendidas
# Shooter: 1052.45 millones de unidades vendidas
# Rol (Role-Playing): 934.56 millones de unidades vendidas
# Plataforma (Platform): 827.77 millones de unidades vendidas
# Estos géneros tienen las mayores ventas totales en el mercado.
# 
# ## Regiones 
# 
# ### Norteamérica:
# Xbox 360 (X360): 602.47 millones de unidades vendidas
# PlayStation 2 (PS2): 583.84 millones de unidades vendidas
# Wii: 496.90 millones de unidades vendidas
# PlayStation 3 (PS3): 393.49 millones de unidades vendidas
# Nintendo DS (DS): 382.40 millones de unidades vendidas
# ### Europa:
# PlayStation 2 (PS2): 339.29 millones de unidades vendidas
# PlayStation 3 (PS3): 330.29 millones de unidades vendidas
# Xbox 360 (X360): 270.76 millones de unidades vendidas
# Wii: 262.21 millones de unidades vendidas
# PlayStation (PS): 213.61 millones de unidades vendidas
# ### Japón:
# Nintendo DS (DS): 175.57 millones de unidades vendidas
# PlayStation (PS): 139.82 millones de unidades vendidas
# PlayStation 2 (PS2): 139.20 millones de unidades vendidas
# Super Nintendo Entertainment System (SNES): 116.55 millones de unidades vendidas
# Nintendo 3DS (3DS): 100.67 millones de unidades vendidas
# Estos datos nos proporcionan información valiosa sobre las preferencias de los consumidores en diferentes regiones y nos permiten identificar las plataformas más populares en cada mercado.

# In[16]:


# Comparar las cuotas de mercado de las cinco plataformas principales entre las regiones
print("\nVariaciones en las cuotas de mercado de las plataformas entre regiones:")
print(top_platforms_na / top_platforms_na.sum())  # Cuotas de mercado normalizadas para Norteamérica
print(top_platforms_eu / top_platforms_eu.sum())  # Cuotas de mercado normalizadas para Europa
print(top_platforms_jp / top_platforms_jp.sum())  # Cuotas de mercado normalizadas para Japón


# In[17]:


# Calcular las ventas totales por clasificación de ESRB para cada región
region_esrb_sales = region_data.groupby(['region', 'rating'])['sales'].sum().unstack()

print("Ventas por clasificación de ESRB en Norteamérica:")
print(region_esrb_sales.loc['NA'])
print("\nVentas por clasificación de ESRB en Europa:")
print(region_esrb_sales.loc['EU'])
print("\nVentas por clasificación de ESRB en Japón:")
print(region_esrb_sales.loc['JP'])


# ## Cuotas de mercado
# Las variaciones en las cuotas de mercado de las plataformas entre regiones nos dan una idea de cómo difieren las preferencias de los consumidores en diferentes partes del mundo. Aquí están las observaciones:
# 
# ### Norteamérica:
# Las cinco principales plataformas en términos de cuota de mercado son Xbox 360 (24.50%), PlayStation 2 (23.74%), Wii (20.21%), PlayStation 3 (16.00%) y Nintendo DS (15.55%).
# Las ventas por clasificación de ESRB muestran que las clasificaciones más populares son:
# E (Todas las edades): 1292.99 millones de unidades vendidas
# T (Adolescentes): 759.75 millones de unidades vendidas
# M (Mature): 748.48 millones de unidades vendidas
# ### Europa:
# Las cinco principales plataformas en términos de cuota de mercado son PlayStation 2 (23.96%), PlayStation 3 (23.32%), Xbox 360 (19.12%), Wii (18.52%) y PlayStation (15.08%).
# Las ventas por clasificación de ESRB muestran que las clasificaciones más populares son:
# E (Todas las edades): 710.25 millones de unidades vendidas
# T (Adolescentes): 427.03 millones de unidades vendidas
# M (Mature): 483.97 millones de unidades vendidas
# ### Japón:
# Las cinco principales plataformas en términos de cuota de mercado son Nintendo DS (26.13%), PlayStation (20.81%), PlayStation 2 (20.72%), Super Nintendo Entertainment System (17.35%) y Nintendo 3DS (14.98%).
# Las ventas por clasificación de ESRB muestran que las clasificaciones más populares son:
# E (Todas las edades): 198.11 millones de unidades vendidas
# T (Adolescentes): 151.40 millones de unidades vendidas
# M (Mature): 64.24 millones de unidades vendidas
# Estas observaciones destacan las diferencias en las preferencias de los consumidores y las tendencias del mercado en cada región, lo que puede ser útil para estrategias de marketing y desarrollo de productos específicos para cada mercado.

# ## Hipotesis

# In[18]:


# Hipótesis 1: Comparación de calificaciones promedio de usuarios entre Xbox One y PC
ratings_xbox = games_data[games_data['platform'] == 'XOne']['user_score'].dropna()
ratings_pc = games_data[games_data['platform'] == 'PC']['user_score'].dropna()
alpha = 0.05

t_statistic, p_value = ttest_ind(ratings_xbox, ratings_pc)
print("Hipótesis 1:")
print(f"T-Statistic: {t_statistic}")
print(f"P-Value: {p_value}")
if p_value < alpha:
    print("Rechazamos la hipótesis nula.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula.")

# Hipótesis 2: Comparación de calificaciones promedio de usuarios entre Acción y Deportes
ratings_action = games_data[games_data['genre'] == 'Action']['user_score'].dropna()
ratings_sports = games_data[games_data['genre'] == 'Sports']['user_score'].dropna()

t_statistic, p_value = ttest_ind(ratings_action, ratings_sports)
print("\nHipótesis 2:")
print(f"T-Statistic: {t_statistic}")
print(f"P-Value: {p_value}")
if p_value < alpha:
    print("Rechazamos la hipótesis nula.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula.")


# ### Hipótesis 1:
# Hipótesis nula (H0): Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son iguales.
# Hipótesis alternativa (H1): Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son diferentes.
# T-Statistic: El valor de -4.37 indica que la diferencia entre las calificaciones promedio de las dos plataformas es significativamente diferente de cero.
# P-Value: El valor extremadamente bajo (aproximadamente 1.39e-05) indica que hay suficiente evidencia para rechazar la hipótesis nula a un nivel de significancia del 0.05. Por lo tanto, concluimos que las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son diferentes.
# 
# ### Hipótesis 2:
# Hipótesis nula (H0): Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son iguales.
# Hipótesis alternativa (H1): Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# T-Statistic: El valor de 1.63 indica que la diferencia entre las calificaciones promedio de los dos géneros es pequeña.
# P-Value: El valor de 0.104 indica que no hay suficiente evidencia para rechazar la hipótesis nula a un nivel de significancia del 0.05. Por lo tanto, no podemos concluir que las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# En resumen, la primera hipótesis sugiere que hay una diferencia significativa en las calificaciones promedio de los usuarios entre las plataformas Xbox One y PC, mientras que la segunda hipótesis no proporciona suficiente evidencia para afirmar que hay una diferencia significativa en las calificaciones promedio de los usuarios entre los géneros de Acción y Deportes.

# ## Conclusión
# 
# Preferencias de Plataforma por Región:
# Las preferencias de plataforma varían significativamente entre regiones, lo que sugiere que los gustos de los consumidores y las tendencias del mercado son distintos en Norteamérica, Europa y Japón.
# Algunas plataformas, como Xbox 360 y PlayStation 2, son populares en Norteamérica y Europa, pero no tienen tanta presencia en Japón, donde las plataformas de Nintendo son más dominantes.
# 
# Géneros más Rentables:
# Los géneros de acción y deportes son consistentemente los más rentables en términos de ventas totales, seguidos de cerca por los géneros de disparos y juegos de rol.
# Esto sugiere que los juegos de acción y deportes son populares entre los consumidores y pueden ser áreas de enfoque rentables para las empresas de videojuegos.
# 
# Impacto de las Clasificaciones ESRB:
# Las ventas por clasificación ESRB varían entre regiones, lo que indica que la clasificación de edad puede influir en las decisiones de compra de los consumidores en diferentes mercados.
# Las clasificaciones ESRB más comunes son "E" (para todos), "T" (adolescentes) y "M" (maduro), lo que sugiere que los juegos dirigidos a audiencias amplias y a adolescentes son populares.
# 
# Pruebas de Hipótesis:
# Las pruebas de hipótesis revelaron diferencias significativas en las calificaciones promedio de los usuarios entre las plataformas Xbox One y PC, lo que sugiere que los usuarios perciben estas plataformas de manera diferente.
# Sin embargo, no hubo suficiente evidencia para afirmar diferencias significativas en las calificaciones promedio de los usuarios entre los géneros de Acción y Deportes.
# 
# En conjunto, estos hallazgos proporcionan información valiosa sobre las tendencias del mercado de videojuegos, las preferencias de los consumidores y las relaciones entre diferentes variables en la industria de los videojuegos. Estas conclusiones pueden ser útiles para tomar decisiones estratégicas en la planificación de lanzamientos de productos, campañas de marketing y desarrollo de juegos.
