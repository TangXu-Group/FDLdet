import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from .utils import *

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")


def semantic_vector_replace(distance, use_feature, semantic_vectors,vector_use_statistics,
                            train_batch_size, img_transform_num, Semantic_num, Channel_num,
                            opt, model, scheduler, replace_threshold=0.9):
    '''vector replace'''
    
    vector_train_attention = (1-(vector_use_statistics/torch.max(vector_use_statistics)))
    replace_postion = (vector_train_attention>replace_threshold).float().nonzero()[:,0]
    
    if len(replace_postion) != 0:
        distance = distance.reshape(train_batch_size*img_transform_num*2, 
                                               Semantic_num, distance.shape[2]*distance.shape[3])
        distance = distance.permute(1,0,2).reshape(Semantic_num, -1)

        #computing minilized distance
        distance_ = torch.min(distance, dim=1)
        v2f_index = distance_[1]

        #obtaining the feature vectors
        new_vectors = use_feature.reshape(train_batch_size*img_transform_num*2,
                                    Channel_num, use_feature.shape[2]*use_feature.shape[3]).permute(1,0,2).reshape(Channel_num, -1)
        new_vectors = new_vectors[:,v2f_index].permute(1,0)


        #updating semantic vector
        vector_use_statistics[replace_postion] = torch.max(vector_use_statistics)/2
        repalce_semantic_vector = semantic_vectors.clone().detach()
        repalce_semantic_vector[replace_postion] = new_vectors[replace_postion].detach()
        semantic_vectors = semantic_vectors_creation(Semantic_num,Channel_num,init=repalce_semantic_vector)
        semantic_vectors = semantic_vectors.cuda()
        
        #optimizing semantic vector
        opt_traget = [
        {'params': model.parameters(), 'lr': 0.001,}, 
        {'params': semantic_vectors, 'lr': 0.001,},
        ]
        opt = torch.optim.Adam(opt_traget)
        scheduler = lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)
        
        return semantic_vectors, opt, scheduler

    
def trans_consis(distance, flag, train_batch_size, img_transform_num, Semantic_num):
    batch_sum = train_batch_size*img_transform_num
    trans_dis = distance.reshape(2, batch_sum, Semantic_num, distance.shape[2], distance.shape[3]).permute(1,0,2,3,4)
    trans_dis = feature_transform(trans_dis, flag.reshape(-1,2), dimx=3, dimy=2).permute(1,0,2,3,4)
    trans_dis = trans_dis.reshape(2,train_batch_size, img_transform_num, Semantic_num, distance.shape[2], distance.shape[3])
    trans_dis = torch.std(trans_dis, dim=2)
    loss_trans = torch.mean(trans_dis)
    return loss_trans


def f2v(distance, vector_use_statistics, neg_threshold, pos_threshold):
    '''feature -> vector and vector -> feature'''
    
    #obtain the min
    distancef2v = torch.min(distance,dim=1)
    indexf2v = distancef2v[1]
    disf2v_p = distancef2v[0]

    use_vectors = list(set(np.array(indexf2v.view(-1).cpu())))
    vector_use_statistics[use_vectors] += 1

    #for selecting unmin's postion
    select_n = 1 - (distance.clone()*0).scatter_(1, indexf2v.unsqueeze(1), 1)
    #make the unmin position's distance to bigger than neg_threshold
    disf2v_n = torch.pow(torch.clamp(neg_threshold - distance, min=0.0),2)
    disf2v_n = (disf2v_n*select_n).view(-1)
    if len(disf2v_n.nonzero())<=1:
        disf2v_n = torch.mean(disf2v_n)
    else:
        disf2v_n = torch.mean(disf2v_n[disf2v_n.nonzero()])

    #make the min position's distance to smaller than pos_threshold
    disf2v_p = torch.pow(torch.clamp(disf2v_p - pos_threshold, min=0.0),2)
    disf2v_p = torch.mean(disf2v_p)

    loss_disf2v = (disf2v_p + disf2v_n)/2
    return disf2v_p, disf2v_n, loss_disf2v


def f2far(semantic_vectors, neg_threshold):
    optim_vectors = semantic_vectors
    optim_num = len(optim_vectors)

    distance = torch.sqrt(torch.sum((optim_vectors.repeat(optim_num,1,1).permute(1,0,2)
                                     - optim_vectors.unsqueeze(0)),dim=2)**2+1e-10)
    self_mask = torch.eye(optim_num).cuda()
    distance = distance * (1-self_mask) + self_mask * neg_threshold
    distance = torch.pow(torch.clamp(neg_threshold - distance, min=0.0),2)
    distance = distance.view(-1)
    loss_inter = torch.mean(distance)
#     if len(distance.nonzero())<=1:
#         loss_inter = torch.mean(distance)
#     else:
#         loss_inter = torch.mean(distance[distance.nonzero()])
    return loss_inter

# def weak_super(distance, use_f, gt, train_mask, Semantic_num, pos_threshold, neg_threshold, 
#                train_batch_size, img_transform_num, Channel_num, img_W, img_H):
#         spe_gt = gt.reshape(train_batch_size*img_transform_num,1, img_W, img_H)
#         spe_gt = nn.UpsamplingNearest2d(size = (use_f.shape[2],use_f.shape[3]))(spe_gt.float()).int()[:,0,:,:]
#         spe_train_mask = train_mask.reshape(train_batch_size*img_transform_num,1, img_W, img_H)
#         spe_train_mask = nn.UpsamplingNearest2d(size = (use_f.shape[2],use_f.shape[3]))(spe_train_mask.float()).int()[:,0,:,:]

#         use_f_dis = use_f.reshape(2, train_batch_size*img_transform_num, Channel_num, use_f.shape[2], use_f.shape[3])
#         use_f_dis = torch.sqrt(torch.sum((use_f_dis[0]-use_f_dis[1])**2,dim=1))

#         spe_distance = distance.reshape(2, train_batch_size*img_transform_num, Semantic_num, distance.shape[2], distance.shape[3])
#         index_distance = torch.min(spe_distance,dim=2)[1]

#         #unchanged region
#         unchanged_region = ((index_distance[0] != index_distance[1]).float() * (1-spe_gt) * spe_train_mask)
#         unchanged_dis = (use_f_dis*unchanged_region).reshape(-1)
#         if len(unchanged_dis.nonzero())<=1:
#             loss_unchanged = 0
#         else:
#             unchanged_dis = unchanged_dis[unchanged_dis.nonzero()]
#             unchanged_dis = torch.clamp(unchanged_dis - pos_threshold, min=0.0)
#             unchanged_dis = unchanged_dis[unchanged_dis.nonzero()]
#             loss_unchanged = torch.mean(torch.pow(unchanged_dis,2))
            
#         #changed region
#         changed_region = ((index_distance[0] == index_distance[1]).float() * spe_gt * spe_train_mask)
#         changed_dis = (use_f_dis*changed_region).reshape(-1)
#         if len(changed_dis.nonzero())<=1:
#             loss_changed = 0
#         else:
#             changed_dis = changed_dis[changed_dis.nonzero()]
#             changed_dis = torch.clamp(neg_threshold - changed_dis, min=0.0)
#             changed_dis = changed_dis[changed_dis.nonzero()]
#             loss_changed = torch.mean(torch.pow(changed_dis,2))
            
#         loss_weak = (loss_changed + loss_unchanged)/2
        
#         return loss_weak
    
def weak_super(distance, use_f, gt, train_mask, Semantic_num, pos_threshold, neg_threshold, 
               train_batch_size, img_transform_num, Channel_num, img_W, img_H):
        spe_gt = gt.reshape(train_batch_size*img_transform_num,1, img_W, img_H)
        spe_gt = nn.UpsamplingNearest2d(size = (use_f.shape[2],use_f.shape[3]))(spe_gt.float()).int()[:,0,:,:]
        spe_train_mask = train_mask.reshape(train_batch_size*img_transform_num,1, img_W, img_H)
        spe_train_mask = nn.UpsamplingNearest2d(size = (use_f.shape[2],use_f.shape[3]))(spe_train_mask.float()).int()[:,0,:,:]

        use_f_dis = use_f.reshape(2, train_batch_size*img_transform_num, Channel_num, use_f.shape[2], use_f.shape[3])
        use_f_dis = torch.sqrt(torch.sum((use_f_dis[0]-use_f_dis[1])**2,dim=1)+1e-10)

        spe_distance = distance.reshape(2, train_batch_size*img_transform_num, Semantic_num, distance.shape[2], distance.shape[3])
        index_distance = torch.min(spe_distance,dim=2)[1]

        #unchanged region
        unchanged_region = ((index_distance[0] != index_distance[1]).float() * (1-spe_gt) * spe_train_mask)
        unchanged_dis = torch.clamp(use_f_dis - pos_threshold, min=0.0)*unchanged_region

        #changed region
        changed_region = ((index_distance[0] == index_distance[1]).float() * spe_gt * spe_train_mask)
        changed_dis = torch.clamp(neg_threshold - use_f_dis, min=0.0)*changed_region

        loss_weak = torch.mean(unchanged_dis+changed_dis)
        
        return loss_weak 
    
    
    
Loss_function_classify = nn.CrossEntropyLoss()
def changeloss(train_mask, gt, prediction):
    train_mask_ = train_mask.view(-1)
    train_index = train_mask_.nonzero()[:,0]
    train_index = torch.where(train_mask_!=0)[0]

    gt_ = gt.view(-1).long()
    gt_ = gt_[train_index]
    
    prediction = prediction.permute(0,2,3,1).reshape(-1,2)
    prediction = prediction[train_index.long()]
    loss = Loss_function_classify(prediction, gt_)
#    neg = gt_.nonzero()[:,0]
#    pos = (1-gt_).nonzero()[:,0]
#    loss_pos = Loss_function_classify(prediction[pos],gt_[pos])
#    loss_neg = Loss_function_classify(prediction[neg],gt_[neg])
#    loss = loss_pos+loss_neg
    return loss

#000
# def distance_compution(use_f, semantic_vectors, Semantic_num, Channel_num, computing_type='total', cuda=device0, cop_batch_size = 10):
#     '''
#     computing_type: 'total', 'batch'
#     '''
#     if computing_type == 'total':
#         use_f_ = use_f.repeat(Semantic_num,1,1,1,1).permute(1,0,2,3,4).to(cuda)
#         semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
#         distance = torch.sqrt(torch.sum((use_f_ - semantic_vectors_)**2,dim=2)+1e-10)
#     else:
#         batch_num = Semantic_num/cop_batch_size
#         if batch_num%1 != 0.0:
#             print('can not be divided: error!')
#             return 0
#         else:
#             distances = []
#             semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
#             use_f_ = use_f.repeat(cop_batch_size,1,1,1,1).permute(1,0,2,3,4).to(cuda)
#             for i in range(int(batch_num)):
#                 distance = 1- torch.cosine_similarity(use_f_ , semantic_vectors_[:,i*cop_batch_size:(i+1)*cop_batch_size], dim=2)
#                 distances.append(distance)
#             distances = torch.cat(distances,dim=1)
#     return distances

# 111
# def distance_compution(use_f, semantic_vectors, Semantic_num, Channel_num, computing_type='total', cuda=device0, cop_batch_size = 10):
#     '''
#     computing_type: 'total', 'batch'
#     '''
#     if computing_type == 'total':
#         use_f_ = use_f.repeat(Semantic_num,1,1,1,1).permute(1,0,2,3,4).to(cuda)
#         semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
#         distance = torch.sqrt(torch.sum((use_f_ - semantic_vectors_)**2,dim=2)+1e-10)
#     else:
#         batch_num = Semantic_num/cop_batch_size
#         if batch_num%1 != 0.0:
#             print('can not be divided: error!')
#             return 0
#         else:
#             distances = []
#             semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
#             for i in range(int(batch_num)):
#                 use_f_ = use_f.repeat(cop_batch_size,1,1,1,1).permute(1,0,2,3,4).to(cuda)
#                 distance = use_f_ - semantic_vectors_[:,i*cop_batch_size:(i+1)*cop_batch_size]
#                 distance = torch.abs(distance.cpu()).to(cuda)
#                 distance = torch.sqrt(torch.sum(distance,dim=2)+1e-10)
                
#                 print(distance.shape)
#                 distances.append(distance)
#             distances = torch.cat(distances,dim=1)
#     return distances


# 222
# def distance_compution(use_f, semantic_vectors, Semantic_num, Channel_num, computing_type='total', cuda=device0, cop_batch_size = 10):
#     '''
#     computing_type: 'total', 'batch'
#     '''
#     if computing_type == 'total':
#         use_f_ = use_f.repeat(Semantic_num,1,1,1,1).permute(1,0,2,3,4).to(cuda)
#         semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
#         distance = torch.sum(torch.abs(use_f_ - semantic_vectors_),dim=2)
#     else:
#         batch_num = Semantic_num/cop_batch_size
#         if batch_num%1 != 0.0:
#             print('can not be divided: error!')
#             return 0
#         else:
#             distances = []
#             semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
#             for i in range(int(batch_num)):
#                 use_f_ = use_f.repeat(cop_batch_size,1,1,1,1).permute(1,0,2,3,4).to(cuda)
#                 distance = use_f_ - semantic_vectors_[:,i*cop_batch_size:(i+1)*cop_batch_size]
#                 distance = torch.square(distance.cpu()).cuda()
#                 distance = torch.sum(distance,dim=2)
#                 distances.append(distance)
#             distances = torch.cat(distances,dim=1)
#     return distance

def distance_compution(use_f, semantic_vectors, Semantic_num, Channel_num, computing_type='total', cuda=device0, cop_batch_size = 10):
    '''
    computing_type: 'total', 'batch'
    '''
    if computing_type == 'total':
        use_f_ = use_f.repeat(Semantic_num,1,1,1,1).permute(1,0,2,3,4).to(cuda)
        semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
        distances = torch.sum(torch.abs(use_f_ - semantic_vectors_),dim=2)
    else:
        batch_num = Semantic_num/cop_batch_size
        if batch_num%1 != 0.0:
            print('can not be divided: error!')
            return 0
        else:
            distances = []
            semantic_vectors_ = semantic_vectors.reshape(1,Semantic_num,Channel_num,1,1).to(cuda)
            V1 = use_f.repeat(cop_batch_size,1,1,1,1).permute(1,0,2,3,4).to(cuda)
            V12 = V1*V1
            for i in range(int(batch_num)):
                V2 = semantic_vectors_[:,i*cop_batch_size:(i+1)*cop_batch_size]
                distance = V12 -V1*V2 -V1*V2 +V2*V2
                distance = torch.sum(distance,dim=2)
                distances.append(distance)
            distances = torch.cat(distances,dim=1)
    return distances