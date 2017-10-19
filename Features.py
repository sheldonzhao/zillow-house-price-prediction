from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np

def feature_engineering(train_df):
    print("Feature enginnering for Train dataset !!")
    print("Coordination ..")
    coords = np.vstack((train_df[['latitude', 'longitude']].values))

    # some out of range int is a good choice
    print("K-means ..")
    sample_ind = np.random.permutation(len(coords))
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
    train_df.loc[:, 'coord_cluster'] = kmeans.predict(train_df[['latitude', 'longitude']])
    print("End !!")

    print(" Start with Coord_cluster ..")

    avg = train_df.groupby('coord_cluster').mean()["bathroomcnt"]
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_bath_zone"] = avg[i]
    somme = train_df.coord_cluster.value_counts()
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "total_propriete_zone"] = somme[i]
    avg = train_df.groupby('coord_cluster').mean()['taxvaluedollarcnt']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_tax_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster').mean()['lotsizesquarefeet']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_lotsizesquarefeet_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster')['fireplaceflag'].sum()
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_fireplace_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster')["taxdelinquencyflag"].sum()
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_taxdelinquencyflag_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster').mean()['unitcnt']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_unitcnt_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster').mean()['bathroomcnt']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_bathroomcnt_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster').mean()['garagecarcnt']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_garagecarcnt_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster').mean()['calculatedfinishedsquarefeet']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_calculatedfinishedsquarefeet_zone"] = avg[i]
    avg = train_df.groupby('coord_cluster').mean()['numberofstories']
    for i in train_df.coord_cluster.unique():
        train_df.loc[train_df.coord_cluster == i, "Avg_numberofstories_zone"] = avg[i]

    print("End !!")

    print(" Start with regionidcity ..")
    # 1 -creer avg  bath on city
    avg = train_df.groupby('regionidcity').mean()["bathroomcnt"]
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_bath_city"] = avg[i]

    # creer la somme des propiete pr chq ville
    somme = train_df.regionidcity.value_counts()
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "total_propriete_city"] = somme[i]

    # 1 -creer avg  tax on city
    avg = train_df.groupby('regionidcity').mean()['taxvaluedollarcnt']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_tax_city"] = avg[i]

    # 1 -creer avg  square feet on city
    avg = train_df.groupby('regionidcity').mean()['lotsizesquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_lotsizesquarefeet_city"] = avg[i]

    # 1 -creer sum  fireplace on city
    avg = train_df.groupby('regionidcity')["fireplaceflag"].sum()
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_fireplace_city"] = avg[i]

    # 1 -creer sum  taxdelinquencyflag on city
    avg = train_df.groupby('regionidcity')["taxdelinquencyflag"].sum()

    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_taxdelinquencyflag_city"] = avg[i]

    # 1 -creer avg  unitcnt on city
    avg = train_df.groupby('regionidcity').mean()['unitcnt']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_unitcnt_city"] = avg[i]

    # 1 -creer avg  bathroomcnt on city
    avg = train_df.groupby('regionidcity').mean()['bathroomcnt']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_bathroomcnt_city"] = avg[i]

    # 1 -creer avg  garagecarcnt on city
    avg = train_df.groupby('regionidcity').mean()['garagecarcnt']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_garagecarcnt_city"] = avg[i]

    # 1 -creer avg  calculatedfinishedsquarefeet on city
    avg = train_df.groupby('regionidcity').mean()['calculatedfinishedsquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_calculatedfinishedsquarefeet_city"] = avg[i]

    # 1 -creer avg  numberofstories on city
    avg = train_df.groupby('regionidcity').mean()['numberofstories']
    # 2 -affecter avg
    for i in train_df.regionidcity.unique():
        train_df.loc[train_df.regionidcity == i, "Avg_numberofstories_city"] = avg[i]

    print("End !!")

    print(" Start with regionidcounty ..")
    # 1 -creer avg  bath on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()["bathroomcnt"]
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_bath_county"] = avg[i]

    # creer la somme des propiete pr chq regionidcounty
    somme = train_df.regionidcounty.value_counts()
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "total_propriete_region"] = somme[i]

    # 1 -creer avg  tax on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['taxvaluedollarcnt']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_tax_county"] = avg[i]

    # 1 -creer avg  square feet on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['lotsizesquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_lotsizesquarefeet_county"] = avg[i]

    # 1 -creer sum  fireplace on regionidcounty
    avg = train_df.groupby('regionidcounty')["fireplaceflag"].sum()
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_fireplace_county"] = avg[i]

    # 1 -creer sum  taxdelinquencyflag on regionidcounty
    avg = train_df.groupby('regionidcounty')["taxdelinquencyflag"].sum()
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_taxdelinquencyflag_county"] = avg[i]

    # 1 -creer avg  unitcnt on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['unitcnt']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_unitcnt_county"] = avg[i]

    # 1 -creer avg  bathroomcnt on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['bathroomcnt']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_bathroomcnt_county"] = avg[i]

    # 1 -creer avg  garagecarcnt on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['garagecarcnt']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_garagecarcnt_county"] = avg[i]

    # 1 -creer avg  calculatedfinishedsquarefeet on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['calculatedfinishedsquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_calculatedfinishedsquarefeet_county"] = avg[i]

    # 1 -creer avg  numberofstories on regionidcounty
    avg = train_df.groupby('regionidcounty').mean()['numberofstories']
    # 2 -affecter avg
    for i in train_df.regionidcounty.unique():
        train_df.loc[train_df.regionidcounty == i, "Avg_numberofstories_county"] = avg[i]

    print("End")

    print(" Start with regionidneighborhood ..")
    # 1 -creer avg  bath on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()["bathroomcnt"]
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_bath_neighborhood"] = avg[i]

    # creer la somme des propiete pr chq regionidneighborhood
    somme = train_df.regionidneighborhood.value_counts()
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "total_propriete_neighborhood"] = somme[i]

    # 1 -creer avg  tax on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['taxvaluedollarcnt']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_tax_neighborhood"] = avg[i]

    # 1 -creer avg  square feet on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['lotsizesquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_lotsizesquarefeet_neighborhood"] = avg[i]

    # 1 -creer sum  fireplace on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood')["fireplaceflag"].sum()
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_fireplace_neighborhood"] = avg[i]

    # 1 -creer sum  taxdelinquencyflag on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood')["taxdelinquencyflag"].sum()
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_taxdelinquencyflag_neighborhood"] = avg[i]

    # 1 -creer avg  unitcnt on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['unitcnt']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_unitcnt_neighborhood"] = avg[i]

    # 1 -creer avg  bathroomcnt on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['bathroomcnt']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_bathroomcnt_neighborhood"] = avg[i]

    # 1 -creer avg  garagecarcnt on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['garagecarcnt']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_garagecarcnt_neighborhood"] = avg[i]

    # 1 -creer avg  calculatedfinishedsquarefeet on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['calculatedfinishedsquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_calculatedfinishedsquarefeet_neighborhood"] = avg[i]

    # 1 -creer avg  numberofstories on regionidneighborhood
    avg = train_df.groupby('regionidneighborhood').mean()['numberofstories']
    # 2 -affecter avg
    for i in train_df.regionidneighborhood.unique():
        train_df.loc[train_df.regionidneighborhood == i, "Avg_numberofstories_neighborhood"] = avg[i]
    print("End !!")

    print(" Start with regionidzip ..")
    # 1 -creer avg  bath on regionidzip
    avg = train_df.groupby('regionidzip').mean()["bathroomcnt"]
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_bath_neighborhood"] = avg[i]

    # creer la somme des propiete pr chq regionidzip
    somme = train_df.regionidzip.value_counts()
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "total_propriete_neighborhood"] = somme[i]

    # 1 -creer avg  tax on regionidzip
    avg = train_df.groupby('regionidzip').mean()['taxvaluedollarcnt']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_tax_neighborhood"] = avg[i]

    # 1 -creer avg  square feet on regionidzip
    avg = train_df.groupby('regionidzip').mean()['lotsizesquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_lotsizesquarefeet_neighborhood"] = avg[i]

    # 1 -creer sum  fireplace on regionidzip
    avg = train_df.groupby('regionidzip')["fireplaceflag"].sum()
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_fireplace_neighborhood"] = avg[i]

    # 1 -creer sum  taxdelinquencyflag on regionidzip
    avg = train_df.groupby('regionidzip')["taxdelinquencyflag"].sum()

    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_taxdelinquencyflag_neighborhood"] = avg[i]

    # 1 -creer avg  unitcnt on regionidzip
    avg = train_df.groupby('regionidzip').mean()['unitcnt']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_unitcnt_neighborhood"] = avg[i]

    # 1 -creer avg  bathroomcnt on regionidzip
    avg = train_df.groupby('regionidzip').mean()['bathroomcnt']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_bathroomcnt_neighborhood"] = avg[i]

    # 1 -creer avg  garagecarcnt on regionidzip
    avg = train_df.groupby('regionidzip').mean()['garagecarcnt']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_garagecarcnt_neighborhood"] = avg[i]

    # 1 -creer avg  calculatedfinishedsquarefeet on regionidzip
    avg = train_df.groupby('regionidzip').mean()['calculatedfinishedsquarefeet']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_calculatedfinishedsquarefeet_neighborhood"] = avg[i]

    # 1 -creer avg  numberofstories on regionidzip
    avg = train_df.groupby('regionidzip').mean()['numberofstories']
    # 2 -affecter avg
    for i in train_df.regionidzip.unique():
        train_df.loc[train_df.regionidzip == i, "Avg_numberofstories_neighborhood"] = avg[i]
    print("End !!")

    # Total number of rooms
    train_df['N-TotalRooms'] = train_df['bathroomcnt'] + train_df['bedroomcnt']

    # Number of Extra rooms
    train_df['N-Extradf_trainRooms'] = train_df['roomcnt'] - train_df['N-TotalRooms']

    # Amout of extra space
    train_df['N-ExtraSpace'] = train_df['lotsizesquarefeet'] - train_df['calculatedfinishedsquarefeet']
    train_df['N-ExtraSpace-2'] = train_df['finishedsquarefeet15'] - train_df['finishedsquarefeet12']
    print("Done Feature enginnering !!")
    return train_df
