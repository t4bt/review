import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L
import numpy as np


class LayerNormalization3D(Chain):

    def __init__(self, d_model):
        super(LayerNormalization3D, self).__init__()
        with self.init_scope():
            self.ln = L.LayerNormalization(d_model)

    def __call__(self, xs):
        # in_size = (B,S,d_model)
        assert xs[0].ndim == 2
        x_section = np.cumsum([len(x) for x in xs])[:-1]
        # (B,S,d_model) ->  (B*S,d_model)
        xs_concat = F.concat(xs, axis=0)
        # LayerNormalization
        xs_concat = self.ln(xs_concat)
        # (B*S,d_model) ->  (B,S,d_model)
        xs_split = F.split_axis(xs_concat, x_section, axis=0)
        # out_size = (B,S,d_model)
        return xs_split


class FeedForward(Chain):

    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(d_model, 4*d_model)
            self.l2 = L.Linear(4*d_model, d_model)

    def __call__(self, xs):
        # in_size = (B,S,d_model)
        assert xs[0].ndim == 2
        x_section = np.cumsum([len(x) for x in xs])[:-1]
        # (B,S,d_model) ->  (B*S,d_model)
        xs_concat = F.concat(xs, axis=0)
        # FeedForward
        xs_concat = self.l2(F.relu(self.l1(xs_concat)))
        # (B*S,d_model) ->  (B,S,d_model)
        xs_split = F.split_axis(xs_concat, x_section, axis=0)
        # out_size = (B,S,d_model)
        return xs_split


class ScaledDotProductAttention(Chain):

    def __init__(self, size_per_head):
        super(ScaledDotProductAttention, self).__init__()
        self.size_per_head = size_per_head

    def __call__(self ,q, k, v):
        # q.shape = (B*num_head,St,size_per_head)
        # k.shape = (B*num_head,Ss,size_per_head)
        # v.shape = (B*num_head,Ss,size_per_head)

        # (B*num_head,St,size_per_head) x (B*num_head,Ss,size_per_head)
        #  -> (B*num_head,St,Ss)
        attn = F.batch_matmul(q, k, transb=True)
        attn *= 1 / np.sqrt(self.size_per_head)
        attn = F.softmax(attn, axis=1)
        # (B*num_head,St,Ss) x (B*num_head,Ss,size_per_head)
        #  -> (B*num_head,St,size_per_head)
        output = F.batch_matmul(attn, v)
        return output, attn


class MultiHeadAttention(Chain):

    def __init__(self, num_head=4, d_model=256, self_attention=True):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.d_model = d_model
        assert d_model % num_head == 0
        self.size_per_head = d_model // num_head
        self.split_section = [i*self.size_per_head for i in range(1,num_head)]
        self.self_attention = self_attention
        with self.init_scope():
            if self_attention:
                self.linear_QKV = L.Linear(d_model, d_model*3, nobias=True)
            else:
                self.linear_Q = L.Linear(d_model, d_model, nobias=True)
                self.linear_KV = L.Linear(d_model, d_model*2, nobias=True)
            self.attention = ScaledDotProductAttention(self.size_per_head)
            self.linear_O = L.Linear(d_model, d_model)

    def __call__(self, source, target=None):

        if self.self_attention:
            t_len = [len(x) for x in source]
            s_len = t_len
            t_section = np.cumsum(t_len)[:-1]
            s_section = np.cumsum(s_len)[:-1]
            # (B,Ss,d_model) -> (B*Ss,d_model)
            source_concat = F.concat(source, axis=0)
            Q, K, V = F.split_axis(self.linear_QKV(source_concat), 3, axis=1)
            K = F.tanh(K)
        
        else:
            t_len = [len(x) for x in target]
            s_len = [len(x) for x in source]
            t_section = np.cumsum(t_len)[:-1]
            s_section = np.cumsum(s_len)[:-1]
            # (B,St,d_model) -> (B*St,d_model)
            target_concat = F.concat(target, axis=0)
            Q = self.linear_Q(target_concat)
            # (B,Ss,d_model) -> (B*Ss,d_model)
            source_concat = F.concat(source, axis=0)
            K, V = F.split_axis(self.linear_KV(source_concat), 2, axis=1)

        # (B,S,d_model) -> (B*num_head,S,size_per_head)
        head_Q = self.to_head(Q, t_section)
        head_K = self.to_head(K, s_section)
        head_V = self.to_head(V, s_section)

        # (B*num_head,St,size_per_head), (B*num_head,St,Ss)
        M, A = self.attention(head_Q, head_K, head_V)

        # (B*num_head,St,size_per_head) -> (B,St,d_model)
        M = self.to_model(M)
        # Remove pad sequence
        M = [M[i,:t_len[i]] for i in range(len(t_len))]
        # OutputLinear
        M = self.linear_O(F.concat(M, axis=0))
        M = F.split_axis(M, t_section, axis=0)

        _, St, _ = head_Q.shape
        _, Ss, _ = head_K.shape
        A = F.split_axis(A, len(source), axis=0)
        A = F.concat([F.sum(a, axis=0).reshape(1,St,Ss) for a in A], axis=0)
        self.AttentionW = [A[i,:t_len[i],:s_len[i]] for i in range(len(t_len))]

        return M, self.AttentionW


    def to_head(self, batch, x_section):
        # insize = (B*S,d_model)
        xs = F.split_axis(batch, x_section, axis=0)
        pad = F.pad_sequence(xs, padding=0.)
        b, l, _ = pad.shape
        head = pad.reshape(b, l, self.num_head, self.size_per_head)
        # (B,S,num_head,size_per_head) -> (B,num_head,S,size_per_head)
        head = F.transpose(head, (0,2,1,3))
        # (B,num_head,S,size_per_head) -> (B*num_head,S,size_per_head)
        return F.concat(head, axis=0)

    def to_model(self, batch):
        b_h, s, d = batch.shape
        b = int(b_h / self.num_head)
        # (B*num_head,S,size_per_head) -> B*(num_head,S,size_per_head)
        head = F.split_axis(batch, b, axis=0)
        # B*(num_head,S,size_per_head) -> (B*S,num_head,size_per_head)
        pad = F.concat([F.transpose(h, (1,0,2)) for h in head], axis=0)
        # (B*S,num_head,size_per_head) -> (B,S,d_model)
        return pad.reshape(b, s, self.d_model)


class TransformerBlock(Chain):

    def __init__(self, num_head, d_model, self_attention):
        super(TransformerBlock, self).__init__()
        self.self_attention = self_attention
        self.A = None
        with self.init_scope():
            self.multi_head_self_attention_1 = MultiHeadAttention(
                                                        num_head=num_head,
                                                        d_model=d_model,
                                                        self_attention=True)
            self.layernorm_1 = LayerNormalization3D(d_model)
            self.feedforward = FeedForward(d_model)
            self.layernorm_f = LayerNormalization3D(d_model)
            if not self_attention:
                self.multi_head_self_attention_2 = MultiHeadAttention(
                                                        num_head=num_head,
                                                        d_model=d_model,
                                                        self_attention=False)
                self.layernorm_2 = LayerNormalization3D(d_model)

    def __call__(self, source, target=None):
        # source = (B,Ss,d_model)
        # target = (B,St,d_model)

        if self.self_attention:
            assert target is None
            # Self-Attention
            M, self.A = self.multi_head_self_attention_1(source)
            # LayerNormalization
            M_ = self.layernorm_1(self.add(source, M))
            # FeedForward
            FF = self.feedforward(M_)
            # LayerNormalization
            F_ = self.layernorm_f(self.add(M_, FF))

        else:
            assert target is not None
            # Self-Attention
            M1, self.A_self = self.multi_head_self_attention_1(target)
            # LayerNormalization
            M_1 = self.layernorm_1(self.add(target, M1))
            # Source-Target Attention
            M2, self.A = self.multi_head_self_attention_2(source, M_1)
            # LayerNormalization
            M_2 = self.layernorm_2(self.add(M_1, M2))
            # FeedForward
            FF = self.feedforward(M_2)
            # LayerNormalization
            F_ = self.layernorm_f(self.add(M_2, FF))

        return F_

    def add(self, a, b):
        # a.shape = b.shape = (B,S,d_model)
        x_section = np.cumsum([len(x) for x in a])[:-1]
        # (B,S,d_model) ->  (B*S,d_model)
        a_concat = F.concat(a, axis=0)
        b_concat = F.concat(b, axis=0)
        # Add
        xs_concat = a_concat + b_concat
        # (B*S,d_model) ->  (B,S,d_model)
        xs_split = F.split_axis(xs_concat, x_section, axis=0)
        return xs_split
