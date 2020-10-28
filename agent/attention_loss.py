import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

"""
# for standard mask
# creating attention masks for 9 different actions
mask_dict = dict()
# move foward
MF_mask = torch.zeros(7,7)
MF_mask[2:5,2:5] = 1 # originally 2-5
mask_dict[0] = MF_mask
# rotate right
RR_mask = torch.zeros(7,7)
RR_mask[:,5:] = 1 # originally 5
mask_dict[1] = RR_mask
# rotate left 
RL_mask = torch.zeros(7,7)
RL_mask[:,:2] = 1 # originally 2
mask_dict[2] = RL_mask
# move back
MB_mask = torch.ones(7,7)
MB_mask[2:5,2:5] = 0 # originally 2-5
mask_dict[3] = MB_mask
# look up
LU_mask = torch.zeros(7,7)
LU_mask[:2,:] = 1 # originally 2
mask_dict[4] = LU_mask
# look down
LD_mask = torch.zeros(7,7)
LD_mask[5:,:] = 1 # originally 5
mask_dict[5] = LD_mask
# move right
MR_mask = torch.zeros(7,7)
MR_mask[:,4:] = 1 # originally 4
mask_dict[6] = MR_mask
# move left
ML_mask = torch.zeros(7,7)
ML_mask[:,:3] = 1 # originally 3
mask_dict[7] = ML_mask
# DONE
done_mask = torch.zeros(7,7)
done_mask[1:6,1:6] = 1
mask_dict[8] = done_mask
"""
"""
# for compositional mask (new)
# creating attention masks for 9 different actions
mask_dict = dict()
# move foward
MF_mask = torch.zeros(7,7)
MF_mask[2:5,2:5] = 1 # originally 2-5
mask_dict[0] = MF_mask
# rotate right
RR_mask = torch.zeros(7,7)
RR_mask[:3,4:] = 1 # originally 5
mask_dict[1] = RR_mask
# rotate left 
RL_mask = torch.zeros(7,7)
RL_mask[:3,:3] = 1 # originally 2
mask_dict[2] = RL_mask
# move back
MB_mask = torch.ones(7,7)
MB_mask[2:5,2:5] = 0 # originally 2-5
mask_dict[3] = MB_mask
# look up
LU_mask = torch.zeros(7,7)
LU_mask[:2,:] = 1 # originally 2
mask_dict[4] = LU_mask
# look down
LD_mask = torch.zeros(7,7)
LD_mask[5:,:] = 1 # originally 5
mask_dict[5] = LD_mask
# move right
MR_mask = torch.zeros(7,7)
MR_mask[1:6,5:] = 1 # originally 4
mask_dict[6] = MR_mask
# move left
ML_mask = torch.zeros(7,7)
ML_mask[1:6,:2] = 1 # originally 3
mask_dict[7] = ML_mask
# DONE
done_mask = torch.zeros(7,7)
done_mask[1:6,1:6] = 1
mask_dict[8] = done_mask
"""

"""
# mask derived based on visualization
# creating attention masks for 9 different actions
mask_dict = dict()
# move foward
MF_mask = torch.zeros(7,7)
MF_mask[2:5,2:5] = 1 # originally 2-5
mask_dict[0] = MF_mask
# rotate right
RR_mask = torch.zeros(7,7)
RR_mask[1:4,4:6] = 1 # originally 5
mask_dict[1] = RR_mask
# rotate left 
RL_mask = torch.zeros(7,7)
RL_mask[1:4,1:3] = 1 # originally 2
mask_dict[2] = RL_mask
# move back
MB_mask = torch.ones(7,7)
MB_mask[1:6,1:6] = 0 # originally 2-5
mask_dict[3] = MB_mask
# look up
LU_mask = torch.zeros(7,7)
LU_mask[:2,1:6] = 1 # originally 2
mask_dict[4] = LU_mask
# look down
LD_mask = torch.zeros(7,7)
LD_mask[5:,1:6] = 1 # originally 5
mask_dict[5] = LD_mask
# move right
MR_mask = torch.zeros(7,7)
MR_mask[2:5,4:6] = 1 # originally 4
mask_dict[6] = MR_mask
# move left
ML_mask = torch.zeros(7,7)
ML_mask[2:5,1:3] = 1 # originally 3
mask_dict[7] = ML_mask
# DONE
done_mask = torch.ones(7,7)
mask_dict[8] = done_mask

candidate_mask = torch.cat([mask_dict[idx].unsqueeze(0) for idx in mask_dict],0).view(9,49)
"""

# mask derived based on visualization (exclusive)
# creating attention masks for 9 different actions
mask_dict = dict()
# move foward
MF_mask = torch.zeros(7,7)
MF_mask[2:4,3] = 1 # originally 2-5
mask_dict[0] = MF_mask
# rotate right
RR_mask = torch.zeros(7,7)
RR_mask[1:4,5] = 1 # originally 5
mask_dict[1] = RR_mask
# rotate left 
RL_mask = torch.zeros(7,7)
RL_mask[1:4,1] = 1 # originally 2
mask_dict[2] = RL_mask
# move back
MB_mask = torch.zeros(7,7)
MB_mask[4:6,3] = 1 # originally 2-5
mask_dict[3] = MB_mask
# look up
LU_mask = torch.zeros(7,7)
LU_mask[:2,2:5] = 1 # originally 2
mask_dict[4] = LU_mask
# look down
LD_mask = torch.zeros(7,7)
LD_mask[6:,2:5] = 1 # originally 5
mask_dict[5] = LD_mask
# move right
MR_mask = torch.zeros(7,7)
MR_mask[2:5,4] = 1 # originally 4
mask_dict[6] = MR_mask
# move left
ML_mask = torch.zeros(7,7)
ML_mask[2:5,2] = 1 # originally 3
mask_dict[7] = ML_mask
# DONE
done_mask = torch.zeros(7,7)
done_mask[1:6,1:6] = 1
mask_dict[8] = done_mask

candidate_mask = torch.cat([mask_dict[idx].unsqueeze(0) for idx in mask_dict],0).view(9,49)


# # loading predefined mask
# mask_dict = torch.from_numpy(np.load('mask_thres_02.npy').astype('float32')).view(9,7,7)
# candidate_mask = mask_dict.view(9,49)


def attention_loss(attention, actions, device):
	# constructing the self-supervised ground truth based on actions
	att_gt = []

	# ###########################
	# for equal weights
	
	for i in range(len(actions)):
		att_gt.append(mask_dict[actions[i]])
	att_gt = torch.stack(att_gt,0).to(device)

	# maximizing the log likelihood
	acc_att = torch.mul(attention,att_gt).view(attention.size(0),-1).sum(-1)
	loss = -torch.log(acc_att).mean()
	# loss = torch.relu(-torch.log(acc_att)-0.35).mean() # with gradient thresholding

	# ##################################

	###########################	
	# # for balanced weights
	# att_mask = torch.zeros(len(actions),) 
	# actions = np.array(actions)
	# num_act = len(np.unique(actions))
	# act_weights = dict()
	# for cur_act in np.unique(actions):
	# 	act_weights[cur_act] = 1/np.count_nonzero(actions==cur_act)

	# for i in range(len(actions)):
	# 	att_gt.append(mask_dict[actions[i]])
	# 	att_mask[i] = act_weights[actions[i]]
	# att_gt = torch.stack(att_gt,0).to(device)
	# att_mask = att_mask.to(device)

	# # maximizing the log likelihood
	# acc_att = torch.mul(attention,att_gt).view(attention.size(0),-1).sum(-1)
	# loss = -torch.mul(torch.log(acc_att),att_mask).sum()/num_act
	# # loss = torch.mul(torch.relu(-torch.log(acc_att)-0.35),att_mask).sum()/num_act # with gradient thresholding
	# ##########################

	return loss


class Adaptive_Mask_Criterion(nn.Module):
	def __init__(self,):
		super(Adaptive_Mask_Criterion,self).__init__()
		self.mask = nn.Parameter(torch.ones(9,49),requires_grad=True) # initialize as uniform distribution

	def forward(self, attention, actions, device):
		# construct the masking based on predicted actions
		att_mask = torch.zeros(len(actions),9,49)
		for i in range(len(actions)):
			att_mask[i,actions[i],:] = 1
		att_mask = att_mask.to(device)
		att_gt = self.mask.unsqueeze(0).expand_as(att_mask)
		att_gt = torch.mul(att_gt,att_mask).sum(1)
		att_gt = F.softmax(att_gt,dim=-1)

		# minimize the cosine distance
		cos_dist = 1-F.cosine_similarity(attention.view(len(actions),-1),att_gt,dim=-1)
		loss = cos_dist.mean()
		return loss

def ada_attention_loss(attention, att_gt, actions, device):
	# construct the masking based on predicted actions
	att_mask = torch.zeros(len(actions),9,49)
	for i in range(len(actions)):
		att_mask[i,actions[i],:] = 1
	att_mask = att_mask.to(device)
	att_gt = torch.mul(att_gt,att_mask).sum(1)
	att_gt = F.relu(att_gt)
	# att_gt = torch.sigmoid(att_gt)
	# att_gt = torch.sigmoid(F.relu(att_gt))

	# maximize the aligned attention distribution
	loss = torch.mul(attention.view(len(actions),-1),att_gt).sum(-1)
	loss = -torch.log(loss).mean()

	return loss

def compositional_attention_loss(attention, policy, device, max_action=None):

	###########################	
	# for balanced weights
	att_mask = torch.zeros(len(max_action),) 
	actions = np.array(max_action)
	num_act = len(np.unique(actions))
	act_weights = dict()
	for cur_act in np.unique(actions):
		act_weights[cur_act] = 1/np.count_nonzero(actions==cur_act)
	for i in range(len(actions)):
		att_mask[i] = act_weights[actions[i]]
	att_mask = att_mask.to(device)

	# construct the masking based on the continuous action distribution
	att_gt = candidate_mask.to(device)
	att_gt = att_gt.unsqueeze(0).expand(len(policy),9,49)
	policy = policy.detach() # cut-off gradient for actions
	# attention = attention.detach() # cut-off gradient for attention
	policy = F.softmax(policy,dim=1)
	policy = policy.unsqueeze(-1).expand_as(att_gt)
	att_gt = torch.mul(att_gt,policy)
	att_gt = att_gt.mean(1)

	# maximize the attention distribution within desirable locaitions
	att_gt = att_gt/(att_gt.max(1)[0].unsqueeze(-1).expand_as(att_gt))
	loss = torch.mul(attention.view(len(policy),-1),att_gt).sum(-1)
	# loss = -torch.log(loss).mean() # equal weights
	loss = torch.mul(-torch.log(loss),att_mask).sum()/num_act # balanced weights

	# # maximize the alignment between template distribution and attention distribution via cross-entropy
	# att_gt = att_gt/(att_gt.sum(1,keepdim=True).expand_as(att_gt))
	# loss = (-att_gt*torch.log(attention.view(len(policy),-1)+1e-15)).sum(-1).mean()

	return loss

def compositional_attention_loss_att2act(attention, policy, device):
	# construct the masking based on the continuous action distribution
	att_gt = candidate_mask.to(device)
	att_gt = att_gt.unsqueeze(0).expand(len(policy),9,49)
	policy = policy.detach() # cut-off gradient for actions
	policy = F.softmax(policy,dim=1)
	att2act = torch.bmm(att_gt,attention.view(len(policy),49,1)).squeeze(-1)
	att2act = att2act/att2act.sum(-1,keepdim=True).expand_as(att2act)

	# minimize the cross-entropt between policy and actions mapped from attention
	loss = -(policy*torch.log(att2act+1e-15)).sum(-1).mean()

	return loss

def compositional_attention_loss_att2act_ada(attention, att_gt, policy):
	# construct the masking based on the continuous action distribution
	att_gt = torch.sigmoid(att_gt)
	policy = policy.detach() # cut-off gradient for actions
	policy = F.softmax(policy,dim=1)
	att2act = torch.bmm(att_gt,attention.view(len(policy),49,1)).squeeze(-1)
	att2act = att2act/att2act.sum(-1,keepdim=True).expand_as(att2act)

	# minimize the cross-entropt between policy and actions mapped from attention
	loss = -(policy*torch.log(att2act+1e-15)).sum(-1).mean()

	return loss

def compositional_attention_loss_ada(attention, att_gt, policy):
	# construct the masking based on the continuous action distribution
	att_gt = torch.sigmoid(att_gt)
	policy = policy.detach() # cut-off gradient for actions
	policy = F.softmax(policy,dim=1)
	policy = policy.unsqueeze(-1).expand_as(att_gt)
	att_mask = torch.mul(att_gt,policy)
	att_mask = att_mask.mean(1)
	att_mask = att_mask/(att_mask.max(1)[0].unsqueeze(-1).expand_as(att_mask))


	# maximize the aligned attention distribution
	loss = torch.mul(attention.view(len(policy),-1),att_mask).sum(-1)
	loss = -torch.log(loss).mean()

	# minimize density and converage
	# constraint_1 = att_gt.mean() # minimize density	
	# constraint_2 = att_gt.max(1)[0].mean()

	# penalize peak value via max
	constraint_1 = att_gt.mean(0).sum(0).max()
	constraint_2 = None

	# # minimize pixel-wise entropy and maximize the class-wise entropy
	# att_gt = att_gt.mean(0)
	# norm_pixel = att_gt/att_gt.sum(0,keepdim=True).expand_as(att_gt)
	# constraint_1 = -torch.mul(norm_pixel,torch.log(norm_pixel+1e-15)).sum(0).mean()
	# norm_class = att_gt.sum(1)/att_gt.sum()
	# constraint_2 = -torch.mul(norm_class,torch.log(norm_class+1e-15)).sum()

	return loss, constraint_1, constraint_2

def compositional_attention_loss_ada_ce(attention, att_gt, policy):
	# construct the masking based on the continuous action distribution
	policy = policy.detach() # cut-off gradient for actions
	policy = F.softmax(policy,dim=1)
	policy = policy.unsqueeze(-1).expand_as(att_gt)
	att_mask = torch.mul(att_gt,policy)
	att_mask = att_mask.mean(1)
	att_mask = F.softmax(att_mask,dim=-1)

	# maximize the alignment between template distribution and attention distribution via cross-entropy
	loss = (-att_mask*torch.log(attention.view(len(policy),-1)+1e-15)).sum(-1).mean()

	return loss















