import numpy as np
import matplotlib.pyplot as plt 
import os
import random
import matplotlib as mpl


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_error_prediction(error_mean, aq_fn, coordinates, ind, expt_name = 'test_expt', iter_nb = 0, save_folder = r"data", to_save = False):
    
    n_dim = int(len(error_mean)**0.5)
    error_mean = error_mean.reshape(n_dim, n_dim)
    
    aq_fn = aq_fn.reshape(n_dim, n_dim)

    x = np.linspace(0, n_dim-1, n_dim)

    X = np.asarray(np.meshgrid(x, x)).T.reshape(-1, 2)



    fig, ax = plt.subplots(1, 2, figsize = (8, 3))
    
    a = ax[0].imshow(error_mean, origin = 'lower')
    ax[0].set_title('Error Map')
    ax[0].scatter(X[ind, 1], X[ind, 0], c = 'r', s = 15)  #sample along the index of the other axis. this is counter-intutive!!! 
    
    b = ax[1].imshow(aq_fn, vmin =0, origin = 'lower')
    ax[1].set_title('Acquisition Function')
    
    fig.colorbar(a, ax=ax[0], fraction=0.05, pad=0.04)
    fig.colorbar(b, ax=ax[1], fraction=0.05, pad=0.04)
    
    
    if to_save:
        img_name = "errormap_iter"+str(iter_nb)+'.jpg'
        
        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)
        
        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')    
    
    plt.show()  

def plot_training_loss(im2spec_train_loss, im2spec_val_loss, error_train_loss, error_val_loss):
    fig, ax = plt.subplots(1,4, figsize = (18,3))
    
    n_models = len(im2spec_train_loss)
    
    for i in range(n_models):
        ax[0].semilogy(im2spec_train_loss[i], label = 'im2spec Training loss')
        ax[1].semilogy(im2spec_val_loss[i], label = 'im2spec Validation loss')     
        
    
    
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Epoch loss")
    ax[0].set_title("im2spec training")    
    
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Epoch loss")
    ax[1].set_title("im2spec Validation")    
    
    ax[2].semilogy(error_train_loss, label = 'error Training loss')
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Epoch loss")
    ax[2].set_title("error model training")    
    
    ax[3].semilogy(error_val_loss, label = 'Error Validation loss')
    ax[3].set_xlabel("Epochs")
    ax[3].set_ylabel("Epoch loss")
    ax[3].set_title("Error model Validation")

    
    plt.show()
    
    
    

def plot_only_training_loss(im2spec_train_loss, im2spec_val_loss):
    fig, ax = plt.subplots(1,2, figsize = (10,3))
    
    n_models = len(im2spec_train_loss)
    
    for i in range(n_models):
        ax[0].semilogy(im2spec_train_loss[i], label = 'im2spec Training loss')
        ax[1].semilogy(im2spec_val_loss[i], label = 'im2spec Validation loss')

    
    
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Epoch loss")
    ax[0].set_title("im2spec training")    
    
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Epoch loss")
    ax[1].set_title("im2spec Validation")    
    
   
    plt.show()
    

def plot_embedding(spectra, spectra_train, xdata = None):       
    
    n = spectra.shape[0]
    dim = spectra.shape[1]   
    
    if xdata is not None:
        dims = xdata
    else:
        dims = np.arange(dim)
    
    mean_spectrum = spectra_train.mean(axis = 0)
    
    
    fig, ax = plt.subplots(1,2, figsize = (10,3))
    
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, n))
    for i in range(n):    
    
        ax[0].scatter(dims, spectra[i], color = colors[i])

    ax[0].set_title("Posterior predictions")
        
        
    ax[1].plot(dims, mean_spectrum)    
    ax[1].set_title("Posterior_training_mean")
    
    plt.show()
    
def plot_latent_space(embeddings, trained_embeddings = None, expt_name = 'test_expt', iter_nb = 0, 
                      save_folder = r"data", to_save = False, lat_order = [0, 1, 2]):
    
    n = embeddings.shape[0]
    dim = embeddings.shape[1]
    
    l1 = embeddings[:, lat_order[0]]
    l2 = embeddings[:, lat_order[1]]
    l3 = embeddings[:, lat_order[2]]

    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l1, l2, l3, alpha=0.2)  # edgecolors='#1f77b4
 

    ax.set_title('Latent space')
    
#     # Remove axis numbers (ticks)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])

    if trained_embeddings is not None:
        n_train = trained_embeddings.shape[0]
        scalar = np.arange(n_train)

        l1_t = trained_embeddings[:, lat_order[0]]
        l2_t = trained_embeddings[:, lat_order[1]]
        l3_t = trained_embeddings[:, lat_order[2]]

        a = ax.scatter(l1_t, l2_t, l3_t, c = scalar, s = 30, edgecolors= 'r', alpha = 1,  cmap = 'Reds')
    
        cbar = plt.colorbar(a)
        cbar.set_label(r"Exploration step", size = 12)
    
    
    if to_save:
        img_name = "latentspace_iter"+str(iter_nb)+'.jpg'
        
        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)
        
        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')        
    
    
    plt.show()
    

def plot_latent_distribution(embeddings, expt_name = 'test_expt', iter_nb = 0, save_folder = r"data", to_save = False):
    
    
    n = embeddings.shape[0]
    n_dim = int(n**0.5)
    
    l1 = embeddings[:, 0].reshape(n_dim, n_dim)
    l2 = embeddings[:, 1].reshape(n_dim, n_dim)
    l3 = embeddings[:, 2].reshape(n_dim, n_dim)
    
    

    fig, ax = plt.subplots(1, 3, figsize = (15, 4))
    
    ax[0].imshow(l1, origin = 'lower')
    ax[0].set_title('L1 distribution')
    
    ax[1].imshow(l2, origin = 'lower')
    ax[1].set_title('L2 distribution')
    
    ax[2].imshow(l3, origin = 'lower')
    ax[2].set_title('L3 distribution')
    
    
    if to_save:
        img_name = "latent_distrbn_iter"+str(iter_nb)+'.jpg'
        
        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)
        
        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')    
    
    plt.show()  
    
def cluster_latent_space(embeddings, lat_order = [0, 1, 2],  n_clusters = 2):
    
    n = embeddings.shape[0]
    dim = embeddings.shape[1]
    
    l1 = embeddings[:, lat_order[0]]
    l2 = embeddings[:, lat_order[1]]
    l3 = embeddings[:, lat_order[2]]

    
    

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)  # Cluster labels
    
    print(labels.shape)

    # Get cluster centers
    centroids = kmeans.cluster_centers_
    
    
    colors = ['red', 'blue', 'green', 'orange']
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
 
    for i in range(n_clusters):
        ax.scatter(embeddings[labels == i,lat_order[0]], embeddings[labels == i, lat_order[1]], embeddings[labels == i, lat_order[2]], 
               color=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')

    ax.set_title('Latent space')
    
#     # Remove axis numbers (ticks)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])

   
    
    plt.show()
    
    
    n = embeddings.shape[0]
    n_dim = int(n**0.5)
    
    x = np.linspace(0, n_dim-1, n_dim)

    X = np.asarray(np.meshgrid(x, x)).T.reshape(-1, 2)

    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    
    for i in range(len(X)):
        ax.scatter(X[i, 1], X[i, 0], c = colors[labels[i]], alpha=0.8)

    plt.show()
    
    
def plot_spectra(pred_spectra, orig_spectrum, error_val, expt_name = 'test_expt', iter_nb = 0, count = 0, 
                 save_folder = r"data", to_save = False, xdata = None):
    
    
    fig, ax = plt.subplots(1, figsize = (4,4))
    
    if xdata is not None:
        x = xdata
    else:
        x = np.arange(len(orig_spectrum))
    
    #ax.plot(x, orig_spectrum, color= 'black')
    ax.plot(x, orig_spectrum, 'o-', color='black', alpha = 0.6, label='original_spectrum')
    
    
    for spectrum in pred_spectra:
        ax.plot(x, spectrum, linewidth = 3, label = f'Predicted\nL1-error: {error_val:.4f}')
    ax.legend(fontsize=10)
    
    if to_save:
        img_name = "spectrum_iter"+str(iter_nb)+'_sample'+str(count)+'.jpg'
        
        save_folder = os.path.join(save_folder, expt_name)
        os.makedirs(save_folder, exist_ok = True)
        
        img_path = os.path.join(save_folder, img_name)
        plt.savefig(img_path, bbox_inches = 'tight')     
    
    plt.show()