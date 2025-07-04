import umap
import matplotlib.pyplot as pyplot

# function for plotting embedings from different datasets in 2D space import umap
def visualize_embeddings(embeddings):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = umap_model.fit_transform(embeddings)

    print(embeddings_2d.shape)

    #plt.scatter(embeddings_2d[])