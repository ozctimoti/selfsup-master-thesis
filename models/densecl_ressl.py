import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseReSSL(nn.Module):
    def __init__(
            self,
            backbone,
            neck,
            queue_len=65536,
            feat_dim=128,
            momentum=0.999,
            temperature=0.1,
            temperature_momentum=0.04
    ):
        super(DenseReSSL, self).__init__()

        self.K = queue_len
        self.m = momentum

        self.ts = temperature  # student
        self.tm = temperature_momentum  # teacher

        self.encoder_q = nn.Sequential(backbone(), neck())
        self.encoder_k = nn.Sequential(backbone(), neck())

        self.backbone = self.encoder_q[0]
        # encoder init OK!

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the second queue for dense output
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        # compute query features
        f_q = self.encoder_q[0](im_q)  # (N, C, S, S)
        g_q, d_q, _ = self.encoder_q[1](f_q)  # (N, C'), (N, C'', S^2), (N, C'')

        f_q = f_q.view(f_q.size(0), f_q.size(1), -1)  # (N, C, S, S) -> (N, C, S^2)

        f_q = nn.functional.normalize(f_q, dim=1)  # (N, C, S^2)
        g_q = nn.functional.normalize(g_q, dim=1)  # (N, C')
        d_q = nn.functional.normalize(d_q, dim=1)  # (N, C'', S^2)
        # q2 = nn.functional.normalize(q2, dim=1) # (N, C'')

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            f_k = self.encoder_k[0](im_k)  # keys: NxC
            g_k, d_k, k2 = self.encoder_k[1](f_k)

            f_k = f_k.view(f_k.size(0), f_k.size(1), -1)

            f_k = nn.functional.normalize(f_k, dim=1)
            g_k = nn.functional.normalize(g_k, dim=1)
            d_k = nn.functional.normalize(d_k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

            # undo shuffle
            f_k = self._batch_unshuffle_ddp(f_k, idx_unshuffle)
            g_k = self._batch_unshuffle_ddp(g_k, idx_unshuffle)
            d_k = self._batch_unshuffle_ddp(d_k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)

            # feat point set sim
            backbone_sim_matrix = torch.matmul(f_q.permute(0, 2, 1),
                                               f_k)  #  (N, S^2, C') * (N, C', S^2) -> (N, S^2, S^2)
            densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # (N, S^2)

            indexed_k_grid = torch.gather(d_k, 2,
                                          densecl_sim_ind.unsqueeze(1).expand(-1, d_k.size(1), -1))  # (N, C'', S^2)

        logitsq = torch.einsum('nc,ck->nk', [g_q, self.queue.clone().detach()])  # (N, K)
        logitsk = torch.einsum('nc,ck->nk', [g_k.detach(), self.queue.clone().detach()])  # (N, K)

        loss_global = - torch.sum(
            F.softmax(logitsk.detach() / self.tm, dim=1) * F.log_softmax(logitsq / self.ts, dim=1), dim=1
        ).mean()

        d_q = d_q.permute(0, 2, 1)  # (N, C'', S^2) -> (N, S^2, C'')
        d_q = d_q.reshape(-1, d_q.size(2))  # (NxS^2, C'')

        indexed_k_grid = indexed_k_grid.permute(0, 2, 1)  # (N, C'', S^2) -> (N, S^2, C'')
        indexed_k_grid = indexed_k_grid.reshape(-1, indexed_k_grid.size(2))  # (NxS^2, C'')

        logitsq = torch.einsum('nc,ck->nk', [d_q, self.queue2.clone().detach()])
        logitsk = torch.einsum('nc,ck->nk', [indexed_k_grid.detach(), self.queue2.clone().detach()])

        loss_dense = - torch.sum(
            F.softmax(logitsk.detach() / self.tm, dim=1) * F.log_softmax(logitsq / self.ts, dim=1), dim=1
        ).mean()

        self._dequeue_and_enqueue(g_k)
        self._dequeue_and_enqueue2(k2)

        return loss_global, loss_dense

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
