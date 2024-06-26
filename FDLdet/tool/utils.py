import torch
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import manifold

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def feature_transform(feature, flag, dimx=2, dimy=1):
    feature_copy = []
    for i,(x_flag, y_flag) in enumerate(flag):
        f = feature[i]
        if x_flag == 1:
            f = flip(f, dimx)
        if y_flag == 1:
            f = flip(f, dimy)
        feature_copy.append(f)
    return torch.stack(feature_copy)

def show_plot(iteration,value,xlable = 'Episodes', ylabel = 'Loss',title = 'The loss change',color = 'g',xyset = (+30,+30), xy=None,xy_label=None):
    #绘制损失变化图
    plt.plot(iteration, value, color+'--', label = 'value')
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.title(title)
    if xy != None:
        plt.annotate(r"$The\ best\ accuracy: $"+str(xy_label),xy=xy,xycoords='data',xytext=xyset,
                     textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
    plt.show()
    
def OA(pre_classes, gt_classes):
    return torch.sum((pre_classes) == (gt_classes)).float()/len(pre_classes)

def get_tsne(data, n_components = 2, n_images = None):
    if n_images is not None:
        data = data[:n_images]
        
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

def get_pca(data, n_components = 19):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data

def plot_representations(data, labels, classes, n_images = None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]      
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'hsv')
    plt.show()
    #handles, _ = scatter.legend_elements(num = None)
    #legend = plt.legend(handles = handles, labels = classes)
    
def semantic_vectors_creation(Block_num,Channel_num, init=None):
    if init is None:
        semantic_vector = torch.randn(Block_num,Channel_num).cuda()
    else:
        semantic_vector = init
    semantic_vector.requires_grad = True
    return semantic_vector

def cosine_similarity(feature1:torch.tensor, feature2:torch.tensor) -> 'cosine similiarity and distance':
    '''
    feature1.shape = feature2.shape
    shape: (sample number, channel dim)
    '''
    feature1_ = feature1.repeat(feature1.shape[0],1,1).permute(1,0,2)
    feature2_ = feature2.repeat(feature2.shape[0],1,1)
    
    similarity = torch.cosine_similarity(feature1_,feature2_,dim=2)
    distance = 1 - similarity
    
    return similarity, distance

def index_means(features: torch.tensor, index: torch.tensor, block_num: int):
    '''
    features.shape: Batch_size, channel_num, W, H
    index.shape: Batch_size, W, H
    index.value: 0 ~ block num-1
    '''
    bs = features.shape[0]
    cl = features.shape[1]
    w = features.shape[2]
    h = features.shape[3]
    bnm = block_num
    
    index = index.unsqueeze(1)
    
    # get one-hot label
    index_scatter = torch.zeros(bs,bnm,w,h).cuda()
    index_scatter = index_scatter.scatter_(1, index, 1)
    
    block_value_sum = torch.sum(index_scatter,dim = (2,3))

    # computing the regional mean of features
    features_ = features.repeat(bnm,1,1,1,1).permute(1,0,2,3,4)
    index_scatter = index_scatter.unsqueeze(2)
    index_means = torch.sum(index_scatter * features_,dim = (3,4))/(block_value_sum+(block_value_sum==0).float()).unsqueeze(2) #* mask.unsqueeze(2)
    return index_means