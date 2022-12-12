import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

def completitud(df, v_null:bool=False):
    df_ = df
    #Dataframe con descripción inicial de variables
    df_inf_null = pd.DataFrame(df_.isna().sum()).reset_index()
    df_inf_null.columns = ['variable','conteo_null']
    df_inf_null['_%_completitud'] = np.round((1-df_inf_null['conteo_null']/df_.shape[0])*100,2)
    df_inf_null['_%_incompletitud'] = np.round((df_inf_null['conteo_null']/df_.shape[0])*100,2)
    df_inf_null= df_inf_null.sort_values('conteo_null', ascending= False)
    incompleto = df_inf_null[df_inf_null['_%_incompletitud']>0]
    #Listado de variables con valores nulos
    variables_nulas = []
    for index, row in df_inf_null.iterrows():
        if row['conteo_null']>0:
            variables_nulas.append(row['variable'])
    return pd.DataFrame(df_inf_null) if v_null else variables_nulas 

# determinar numero de valores unicos por variable
def unicos(df, n_unicos, v_unicos:bool=True):
    df_=df
    lista = []
    for icol in df_.columns:
        n = len(pd.unique(df_[icol]))
        lista_final = lista.append([icol, n])
        df_lista = pd.DataFrame(lista)
        df_lista.columns = ["variable", "valores_unicos"]
        df_lista = df_lista.sort_values(by='valores_unicos', ascending=True)
    #df_lista_10 = df_lista[df_lista['valores_unicos']<=n_unicos]
    #
    variables_unicos = []
    for index, row in df_lista.iterrows():
        if row['valores_unicos']<=n_unicos:
            variables_unicos.append(row['variable'])    
    return pd.DataFrame(df_lista) if v_unicos else variables_unicos


# genera una tabla y graficas con información relevante del dataset
def inf_table(df, n_unicos):
    df_ = df
    n_ = n_unicos
    df_info_variables = pd.DataFrame(df_.dtypes).reset_index()
    df_info_variables.columns = ['variable','tipo']
    completitud_ = completitud(df_, v_null=True)
    df_completitud = completitud_
    unicos_= unicos(df_, n_unicos = n_, v_unicos=True)
    df_unicos = unicos_
    estado_variables = df_unicos.merge(df_completitud).merge(df_info_variables)
    estado_variables = estado_variables.sort_values('conteo_null', ascending= False).reset_index(drop = True)
    return estado_variables

#genera graficas de barras para las variables categoricas
def graficas_categoricas(df,cat_column):
    counts = round(df[cat_column].value_counts().sort_index()/len(df[cat_column])*100,2)
    colors = plt.cm.tab10(np.arange(len(counts)))
    plt.xlabel("Categoria")
    plt.ylabel("%")
    plt.title(cat_column)
    ax = counts.plot.bar(color=colors)
    ax.bar_label(ax.containers[0], label_type='edge')
    return ax

#genera boxplot de la variable num_column por by_column
def grafica_boxplot(df, by_column, num_column, ylab):
    ax = df.boxplot(by =by_column, column =[num_column], grid = False,figsize=(15,4))
    plt.xlabel(by_column)
    plt.ylabel(ylab)
    #plt.yscale('log')
    [ax_tmp.set_xlabel(by_column) for ax_tmp in np.asarray(ax).reshape(-1)]
    fig = np.asarray(ax).reshape(-1)[0].get_figure()
    plt.suptitle('') 
    return plt.show()

#transofrmacion numerica
def transformacion_numerica(df, metodo, col): #metodos log1p, zscore, minmax
    df_ = df
    if metodo == "log1p":
        df_[col+"_log1p"] = np.log1p(df_[col])
    elif metodo == "zscore":
        df_[col+"_zscore"] = (df_[col] - df_[col].mean()) / df_[col].std()
    elif metodo == "minmax":
        df_[col+"_minmax"] = (df_[col] - df_[col].min()) / (df_[col].max()-df_[col].min()) 
    else:
        print('elegir metodo')
          
#Transformar a int64
def transforma_int64(df, col_):
    df_ = df
    df_[col_] = df_[col_].astype("int64", errors = "ignore")

#Transformar a categoria
def transforma_cat(df, col_):
    df_ = df
    df_[col_] = df_[col_].astype("category", errors = "ignore")
        
#Transformar a date
def transforma_date(df, col_):
    df_ = df
    df_[col_] = df_[col_].astype("datetime64[ns]",errors = "ignore" )
        
#Transformar a object
def transforma_obj(df, col_):
    df_ = df
    df_[col_] = df_[col_].astype("object",errors = "ignore" )

def perfilFila(df, col, outcome):
    perfilf = pd.crosstab(index = df[outcome],
                        columns = df[col],
                        normalize = 'index', margins = True,
                        rownames = [outcome],
                        colnames = [col],
                        margins_name = 'Total').round(4)*100
    
    ax = perfilf.plot(kind='bar', stacked=True, rot=0)
    ax.legend(title=col, bbox_to_anchor=(1, 1.02), loc='upper left')
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    
    return print(perfilf,'\n*********************************'),  ax

def crossT(df, col, outcome, sig = None):
    if sig:
        sig_lev = sig
    else: 
        sig = 0.05
    ####    
    cross = pd.crosstab(df[outcome], df[col], margins = True, margins_name = "Total",
           rownames = [outcome],
           colnames = [col])
    chi2, p, dof, ex = stats.chi2_contingency(cross)
    ###
    print('*********************************\n',
        'Chi cuadrado: {}, P - Value {}, Dof: {}'.format(round(chi2,4), round(p,8), dof)) 
    if p < sig:
        print('Rechazamos Ho. Las variables NO son independientes','\n ----------------------------------- ')
    else:
        print('Aceptamos Ho. Las variables son independientes', '\n ----------------------------------- ')
    return cross 
