import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from types import SimpleNamespace
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.parallel.scatter_gather import gather


# Number of Linear input connections depends on output of conv2d layers
# and therefore the input image size, so compute it.
def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


# https://github.com/pytorch/pytorch/issues/16885
class CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    # def gather(self, outputs, output_device):
    #     keys = outputs[0].__dict__.keys()
    #
    #     output = SimpleNamespace()
    #     return gather(outputs, output_device, dim=self.dim)


class BoltzmannPolicy(nn.Module):

    def __init__(self, temp_start, temp_end, temp_decay):
        super(BoltzmannPolicy, self).__init__()
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay = temp_decay
        self.temp = self.current_temperature(step=0)

    def current_temperature(self, step):
        return self.temp_end + (self.temp_start - self.temp_end) * math.exp(-1. * step/self.temp_decay)

    def forward(self, x, step):
        self.temp = self.current_temperature(step)
        return F.softmax(x/self.temp, dim=-1)

    def sample(self, x, step):
        pi = Categorical(self.forward(x, step))
        return pi.sample()


class EpsilonGreedyPolicy(nn.Module):

    def __init__(self, eps_start, eps_end, eps_decay):
        super(EpsilonGreedyPolicy, self).__init__()
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = self.current_epsilon(step=0)

    def current_epsilon(self, step):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step/self.eps_decay)

    def forward(self, x, step):
        self.eps = self.current_epsilon(step)
        pi = torch.zeros(x.shape) + self.eps/x.shape[-1]
        pi[range(x.shape[0]), x.argmax(-1)] += (1 - self.eps)
        return pi

    def sample(self, x, step):
        pi = Categorical(self.forward(x, step))
        return pi.sample().reshape((x.shape[0], -1))


class DQN(nn.Module):

    def __init__(self, h, w, c, outputs, conv_layers=((16, 5, 2), (32, 5, 2), (32, 5, 2))):
        super(DQN, self).__init__()

        prev_layer_n_channels = c
        self.convs_and_bns = nn.ModuleList()
        convw, convh = w, h
        for filters, kernel_size, stride in conv_layers:
            self.convs_and_bns.append(nn.Conv2d(prev_layer_n_channels, filters, kernel_size=kernel_size, stride=stride))
            # self.convs_and_bns.append(nn.BatchNorm2d(filters))
            self.convs_and_bns.append(nn.ReLU())
            prev_layer_n_channels = filters

            convw, convh = conv2d_size_out(convw, kernel_size, stride), conv2d_size_out(convh, kernel_size, stride)

        linear_input_size = convw * convh * prev_layer_n_channels
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for module in self.convs_and_bns:
            x = module(x)
        return self.head(x.view(x.size(0), -1))


class AttentionPolicyDistillation(nn.Module):

    def __init__(self, device, k_dim, c_dim, preds, actions, mechanism, softmax, affine, fit_residuals, standardize=False):
        # k_dim: key dimension
        # c_dim: context dimension
        # heads: number of output heads
        # preds: number of predicates
        # actions: number of actions
        # mechanism: how to do the distillation
        # softmax: whether to use probabilistic weights

        super(AttentionPolicyDistillation, self).__init__()

        self.device = device

        if mechanism == 'lin_action_independent':
            self.key_layer = nn.Linear(actions, k_dim)
            self.query_layer = nn.Linear(actions, k_dim)
            v_dim = actions

        elif mechanism == 'lin_action_independent_with_state_concat':
            self.key_layer = nn.Linear(c_dim + actions, k_dim)
            self.query_layer = nn.Linear(c_dim + actions, k_dim)
            v_dim = actions

        elif mechanism == 'lin_state_action_concat':
            self.key_layer = nn.Linear(c_dim + 2 * actions, k_dim)
            self.query_layer = nn.Linear(c_dim + 2 * actions, k_dim)
            v_dim = 1

        elif mechanism == '2layer_state_action_concat':
            self.state_embedding = nn.Linear(c_dim, 16)
            self.action_embedding = nn.Linear(actions, 8)
            self.key_layer = nn.Linear(16 + 8 + actions, k_dim)
            self.query_layer = nn.Linear(16 + 8 + actions, k_dim)
            v_dim = 1

        elif mechanism == '2layer_state_action_concat_no_q':
            self.state_embedding = nn.Linear(c_dim, 16)
            self.action_embedding = nn.Linear(actions, 8)
            self.key_layer = nn.Linear(16 + 8 + actions, k_dim)
            self.query_layer = nn.Linear(16 + 8, k_dim)
            v_dim = 1

        elif mechanism == 'global_regression':
            self.regression_layer = nn.Linear(preds, 1, bias=False)
            v_dim = 1
            if affine:
                raise RuntimeError('Affine is incompatible with global regression.')

        else:
            raise NotImplementedError

        self.scaling = k_dim ** -0.5
        self.softmax = softmax
        self.actions = actions
        self.mechanism = mechanism
        self.affine = affine
        self.v_dim = v_dim
        self.fit_residuals = fit_residuals
        self.standardize = standardize

        if affine:
            self.scale = nn.Parameter(torch.ones(preds, v_dim))
            self.offset = nn.Parameter(torch.zeros(preds, v_dim))

        if fit_residuals:
            # self.residual_layer = nn.Sequential(nn.Linear(c_dim + (v_dim == 1) * actions, k_dim),
            #                                     nn.Linear(k_dim, v_dim))
            self.residual_layer = nn.Linear(c_dim, actions)

        self.demon_scale_mean = nn.Parameter(torch.zeros(preds, actions), requires_grad=False)
        self.demon_scale_std = nn.Parameter(torch.ones(preds, actions), requires_grad=False)
        self.target_scale_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.target_scale_std = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward_residuals(self, context):
        if not self.fit_residuals:
            return None
        # (32, 128)
        # if self.v_dim == 1:
        #     context = context.repeat_interleave(self.actions, dim=0)
        #     action_embeddings = torch.eye(self.actions).repeat((context.shape[0], 1))
        #     context = torch.cat([context, action_embeddings], dim=-1)
        return SimpleNamespace(residuals=self.residual_layer(context))

    def standardize_predq(self, pred_q):
        if self.standardize:
            return (pred_q - self.demon_scale_mean)/self.demon_scale_std
        else:
            return pred_q

    def forward(self, context, q, pred_q):
        pred_q = self.standardize_predq(pred_q)
        # (32, 128), (32, 7), (32, 10, 7)
        batch_size = context.shape[0]
        n_actions = q.shape[-1]
        n_preds = pred_q.shape[-2]

        if self.mechanism == 'lin_action_independent':
            values = pred_q # (32, 10, 7)

            queries = self.query_layer(q).unsqueeze(1) # (32, 1, k_dim)

            keys = self.key_layer(pred_q) # (32, 10, k_dim)

            return self.attend(keys, queries, values)

        elif self.mechanism == 'lin_action_independent_with_state_concat':
            values = pred_q # (32, 10, 7)

            queries = self.query_layer(torch.cat([context, q], dim=1)).unsqueeze(1)  # (32, 1, k_dim)

            context = context.unsqueeze(1).expand(-1, n_preds, -1) # (32, 10, 128)
            keys = self.key_layer(torch.cat([context, pred_q], dim=2)) # (32, 10, k_dim)

            return self.attend(keys, queries, values)

        elif self.mechanism == 'lin_state_action_concat':
            values = pred_q.transpose(1, 2).reshape(-1, n_preds, 1) # (32 * 7, 10, 1)

            context = context.repeat_interleave(n_actions, dim=0) # (32 * 7, 128)
            action_embeddings = torch.eye(n_actions).repeat((batch_size, 1)).to(self.device) # (32 * 7, 7)
            q = q.repeat_interleave(n_actions, dim=0) # (32 * 7, 7)
            queries = self.query_layer(torch.cat([context, action_embeddings, q], dim=1)).unsqueeze(1) # (32 * 7, 1, k_dim)

            context = context.unsqueeze(1).expand(-1, n_preds, -1)
            pred_q = pred_q.repeat_interleave(n_actions, dim=0)
            action_embeddings = action_embeddings.unsqueeze(1).expand(-1, n_preds, -1)
            keys = self.key_layer(torch.cat([context, action_embeddings, pred_q], dim=2)) #(32 * 7, 10, k_dim)

            return self.attend(keys, queries, values)

        elif self.mechanism == '2layer_state_action_concat':
            values = pred_q.transpose(1, 2).reshape(-1, n_preds, 1) # (32 * 7, 10, 1)

            context_emb = self.state_embedding(context) # (32, 16)
            action_emb = self.action_embedding(torch.eye(n_actions, device=self.device)) # (7, 8)

            context = context_emb.repeat_interleave(n_actions, dim=0)  # (32 * 7, 16)
            action_embeddings = action_emb.repeat((batch_size, 1))  # (32 * 7, 8)
            q = q.repeat_interleave(n_actions, dim=0)  # (32 * 7, 7)
            queries = self.query_layer(torch.cat([context, action_embeddings, q], dim=1)).unsqueeze(1)  # (32 * 7, 1, k_dim)

            context = context.unsqueeze(1).expand(-1, n_preds, -1)
            pred_q = pred_q.repeat_interleave(n_actions, dim=0)
            action_embeddings = action_embeddings.unsqueeze(1).expand(-1, n_preds, -1)
            keys = self.key_layer(torch.cat([context, action_embeddings, pred_q], dim=2))  # (32 * 7, 10, k_dim)

            return self.attend(keys, queries, values)

        elif self.mechanism == '2layer_state_action_concat_no_q':
            values = pred_q.transpose(1, 2).reshape(-1, n_preds, 1)  # (32 * 7, 10, 1)

            context_emb = self.state_embedding(context)  # (32, 16)
            action_emb = self.action_embedding(torch.eye(n_actions, device=self.device))  # (7, 8)

            context = context_emb.repeat_interleave(n_actions, dim=0)  # (32 * 7, 16)
            action_embeddings = action_emb.repeat((batch_size, 1))  # (32 * 7, 8)
            queries = self.query_layer(torch.cat([context, action_embeddings], dim=1)).unsqueeze(1)  # (32 * 7, 1, k_dim)

            context = context.unsqueeze(1).expand(-1, n_preds, -1)
            pred_q = pred_q.repeat_interleave(n_actions, dim=0)
            action_embeddings = action_embeddings.unsqueeze(1).expand(-1, n_preds, -1)
            keys = self.key_layer(torch.cat([context, action_embeddings, pred_q], dim=2))  # (32 * 7, 10, k_dim)

            return self.attend(keys, queries, values)

        elif self.mechanism == 'global_regression':
            values = pred_q.transpose(1, 2).reshape(-1, n_preds)  # (32 * 7, 10)
            output = self.regression_layer(values).unsqueeze(1) # (32 * 7, 1)
            if self.softmax:
                output = F.softmax(output, dim=-1)

            return SimpleNamespace(output=output,
                                   weights=self.regression_layer.weight)

        else:
            raise NotImplementedError

    def attend(self, k, q, v):
        q *= self.scaling
        if self.affine:
            v = v * self.scale + self.offset
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if self.softmax:
            attn_output_weights = F.softmax(attn_output_weights.float(), dim=-1,
                                            dtype=torch.float32 if attn_output_weights.dtype == torch.float16
                                            else attn_output_weights.dtype)

        attn_output = torch.bmm(attn_output_weights, v)
        if self.standardize:
            attn_output = attn_output * self.target_scale_std + self.target_scale_mean

        return SimpleNamespace(output=attn_output,
                               weights=attn_output_weights)


class Horde(nn.Module):

    @staticmethod
    def build_convnet(h, w, c, conv_layers, batch_norm):
        prev_layer_n_channels = c
        convs_and_bns = nn.ModuleList()

        convw, convh = w, h
        for filters, kernel_size, stride in conv_layers:
            convs_and_bns.append(nn.Conv2d(prev_layer_n_channels, filters, kernel_size=kernel_size, stride=stride))
            if batch_norm:
                convs_and_bns.append(nn.BatchNorm2d(filters))
            convs_and_bns.append(nn.ReLU())
            prev_layer_n_channels = filters

            convw, convh = conv2d_size_out(convw, kernel_size, stride), conv2d_size_out(convh, kernel_size, stride)

        # Return network, and size of embedding that it generates
        return nn.Sequential(*convs_and_bns), convw * convh * prev_layer_n_channels

    def activate_phase(self):
        if self.phase == 2:
            for e in self.convs_and_bns:
                e.requires_grad = False
            self.main_demon.requires_grad = False
        elif self.phase == 3:
            for e in self.convs_and_bns:
                e.requires_grad = False
            self.main_demon.requires_grad = False
            self.prediction_demons.requires_grad = False
            self.control_demons.requires_grad = False

    def __init__(self, device, h, w, c, heads, outputs,
                 conv_layers=((16, 5, 2), (32, 5, 2), (32, 5, 2)),
                 batch_norm=False,
                 detach_aux_demons=True,
                 two_streams=False,
                 phase=1):

        super(Horde, self).__init__()
        self.device = device
        self.heads = heads
        self.n_outputs = outputs
        self.detach = detach_aux_demons
        self.two_streams = two_streams
        self.phase = phase

        # Can't detach if we're learning a separate representation for
        assert (self.two_streams and not self.detach or not self.two_streams), "Can't detach with two streams."

        self.convs_and_bns, self.linear_input_size = self.build_convnet(h, w, c, conv_layers, batch_norm)
        if self.two_streams:
            self.demon_convs_and_bns, _ = self.build_convnet(h, w, c, conv_layers, batch_norm)

        self.main_demon = nn.Linear(self.linear_input_size, outputs)
        if self.heads > 0:
            self.prediction_demons = nn.Linear(self.linear_input_size, outputs * heads)
            self.control_demons = nn.Linear(self.linear_input_size, outputs * heads)

    def forward(self, x):
        # Main representation
        z = self.convs_and_bns(x)
        z = z.view(z.size(0), -1)

        if self.two_streams:
            # If separate streams, then use the other conv net to get the demon representation
            demon_z = self.demon_convs_and_bns(x)
            demon_z = demon_z.view(demon_z.size(0), -1)
        else:
            # If same stream, make sure to correctly detach
            demon_z = z.detach() if self.detach else z

        return SimpleNamespace(embedding=z,
                               demon_embedding=demon_z,
                               main_demon=self.main_demon(z),
                               control_demons=self.control_demons(demon_z).
                               reshape(demon_z.size(0), self.heads, self.n_outputs) if self.heads > 0 else None,
                               prediction_demons=self.prediction_demons(demon_z).
                               reshape(demon_z.size(0), self.heads, self.n_outputs) if self.heads > 0 else None)

    def forward_q(self, x):
        # Common representation
        x = self.convs_and_bns(x)
        x = x.view(x.size(0), -1)
        return self.main_demon(x)

    def copy_convnet_to_demon_convnet(self):
        self.demon_convs_and_bns.load_state_dict(self.convs_and_bns.state_dict())


class MultiHeadDQN(nn.Module):

    def __init__(self, device, h, w, c, heads, outputs,
                 attn_heads=1,
                 conv_layers=((16, 5, 2), (32, 5, 2), (32, 5, 2)),
                 output_wts=True,
                 detach_attn_context=True):

        super(MultiHeadDQN, self).__init__()

        self.device = device
        self.detach_attn_context = detach_attn_context

        prev_layer_n_channels = c
        self.convs_and_bns = nn.ModuleList()
        convw, convh = w, h
        for filters, kernel_size, stride in conv_layers:
            self.convs_and_bns.append(nn.Conv2d(prev_layer_n_channels, filters, kernel_size=kernel_size, stride=stride))
            # self.convs_and_bns.append(nn.BatchNorm2d(filters))
            self.convs_and_bns.append(nn.ReLU())
            prev_layer_n_channels = filters

            convw, convh = conv2d_size_out(convw, kernel_size, stride), conv2d_size_out(convh, kernel_size, stride)

        linear_input_size = convw * convh * prev_layer_n_channels

        self.main_head = nn.Linear(linear_input_size, outputs)
        self.auxiliary_heads = nn.ModuleList([nn.Linear(linear_input_size, outputs) for _ in range(heads)])

        if attn_heads > 0 and self.auxiliary_heads:
            self.attention = MultiheadAttention(embed_dim=outputs, num_heads=attn_heads,
                                                output_wts=output_wts, context_dim=linear_input_size)

    def forward(self, x):
        # Common representation
        for module in self.convs_and_bns:
            x = module(x)
        x = x.view(x.size(0), -1)

        # Main Q function
        main_output = self.main_head(x)

        # Auxiliary Q functions
        auxiliary_outputs = []
        for head in self.auxiliary_heads:
            auxiliary_outputs.append(head(x.view(x.size(0), 1, -1)))

        if auxiliary_outputs:
            auxiliary_outputs = torch.cat(auxiliary_outputs, dim=1) # (batch, heads, outputs)
            # Query vector is the main Q function
            # Each key corresponds to a single predicate Q^p_i function
            # Each value also corresponds to a single predicate Q^p_i function
            # Attention weight tells us some measure of how much information is shared by Q with Q^p_i
            # Detach all the Q functions before throwing them into the attention mechanism to decouple
            # Q-learning from the learning of the attention mechanism
            # Are there other ways to find the attention weights that leverage more side information?
            # detached_auxiliary_outputs = torch.tensor(auxiliary_outputs.transpose(0, 1).detach(),
            #                                           device=self.device, requires_grad=True)
            context = x.detach() if self.detach_attn_context else x
            attention_output, attention_weights = self.attention.forward(query=main_output.unsqueeze(0).detach(),
                                                                         key=auxiliary_outputs.transpose(0, 1).detach(),
                                                                         value=auxiliary_outputs.transpose(0, 1).detach(),
                                                                         context=context)
            tilde_q = attention_output.transpose(0, 1)

        else:
            auxiliary_outputs = tilde_q = attention_weights = None

        return SimpleNamespace(embedding=x,
                               q=main_output,
                               aux_qs=auxiliary_outputs,
                               tilde_q=tilde_q,
                               attention=attention_weights)

    def forward_q(self, x):
        # Common representation
        for module in self.convs_and_bns:
            x = module(x)

        # Main Q function
        return self.main_head(x.view(x.size(0), -1))


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, context_dim, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 output_wts=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # Concatenate the key/query to the context and process
        self.context_layer = nn.Linear(context_dim + embed_dim, embed_dim)

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if output_wts:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.out_proj = None

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight[:self.embed_dim, :])
        xavier_uniform_(self.in_proj_weight[self.embed_dim:(self.embed_dim * 2), :])
        xavier_uniform_(self.in_proj_weight[(self.embed_dim * 2):, :])

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
        if self.out_proj is not None:
            xavier_uniform_(self.out_proj.weight)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, context=None, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)

        # use the context to improve the query and keys
        q = self.context_layer(torch.cat([context.unsqueeze(0), q], dim=-1))
        k = self.context_layer(torch.cat([context.unsqueeze(0).expand(k.shape[0], -1, -1), k], dim=-1))

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights.float(), dim=-1,
            dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        if self.out_proj is not None:
            attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights

    def _in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def _in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def _in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def _in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def _in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


if __name__ == '__main__':
    horde_1 = Horde('cpu', 4, 4, 3, 0, 6)
    horde_2 = Horde('cpu', 4, 4, 3, 12, 6)

    for e in horde_2.named_parameters():
        print(e[0], e[1].mean())
    horde_2.load_state_dict(horde_1.state_dict(), strict=False)
    for e in horde_2.named_parameters():
        print(e[0], e[1].mean())