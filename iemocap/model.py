"""TriMEF model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, RobertaModel
from collections import OrderedDict
import math
from reliability_logit_fusion import (
    ReliabilityAwareLogitResidual,
    reliability_auxiliary_losses,
)
from decision_moe import DecisionLevelEmotionMoE
from interaction import JointTrimodalAttention, TextAnchorResidual


from contrastive import SimpleContrastiveLearning, VideoTextContrastiveLearning
from avel import AVEL
from utec import UnifiedTrimodalEmotionContrastive


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertCoAttention(nn.Module):

    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + s2_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(
            s1_hidden_states, s2_hidden_states, s2_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertSelfEncoder(nn.Module):
    def __init__(self, config):
        super(BertSelfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([layer])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([layer for _ in range(layer_num)])

    def forward(
        self,
        s1_hidden_states,
        s2_hidden_states,
        s2_attention_mask,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(
                s1_hidden_states, s2_hidden_states, s2_attention_mask
            )
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers


class ActivateFun(nn.Module):
    def __init__(self, activate_fun):
        super(ActivateFun, self).__init__()
        self.activate_fun = activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == "relu":
            return torch.relu(x)
        elif self.activate_fun == "gelu":
            return self._gelu(x)


class ASRHead(nn.Module):

    def __init__(self, config, vocab_size):
        super(ASRHead, self).__init__()

        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.hidden_dropout_prob,
        )

        self.layer_norm = nn.LayerNorm(256 * 2)

        self.output_projection = nn.Linear(256 * 2, vocab_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):

        x = self.dropout(x)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)

        x = self.layer_norm(lstm_out)
        x = self.activation(x)

        logits = self.output_projection(x)
        return logits


class AttentionPooling(nn.Module):

    def __init__(self, hidden_size, attention_hidden_size=256, dropout=0.1):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden_size, 1),
        )

    def forward(self, hidden_states, attention_mask=None):

        attention_scores = self.attention(hidden_states)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )

        attention_weights = F.softmax(attention_scores, dim=1)

        pooled = torch.sum(hidden_states * attention_weights, dim=1)

        return pooled, attention_weights


class MMERConfig:

    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.intermediate_size = 3072


class MMERModel(nn.Module):

    def __init__(
        self,
        wav2vec2_path,
        roberta_path,
        num_emotions=4,
        ctc_vocab_size=33,
        freeze_audio_extractor=True,
        use_contrastive=True,
        contrastive_weight=0.1,
        video_feature_dim=512,
        use_video=False,
        use_vt_contrastive=True,
        video_fusion_variant="scalar",
        video_residual_scale=1.0,
        video_gate_bias=-2.0,
        video_gate_groups=12,
        video_dropout=0.35,
        use_avel=False,
        avel_temperature=0.07,
        use_utec=False,
        utec_temperature=0.07,
        logit_fusion_source="learned",
        video_evidence_temperature=2.0,
        video_evidence_target_scale=0.25,
        video_evidence_warmup_epochs=5,
        video_evidence_ramp_epochs=5,
        use_decision_moe=False,
        interaction_variant="pairwise_residual",
        joint_num_layers=6,
        joint_feedforward_size=3584,
    ):
        super(MMERModel, self).__init__()

        self.config = MMERConfig()
        self.num_emotions = num_emotions
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.use_video = use_video
        self.use_vt_contrastive = use_video and use_vt_contrastive
        self.use_avel = use_video and use_avel
        self.avel_temperature = float(avel_temperature)
        self.use_utec = bool(use_video and use_contrastive and use_utec)
        self.utec_temperature = float(utec_temperature)
        self.video_fusion_variant = video_fusion_variant
        self.video_residual_scale = float(video_residual_scale)
        self.video_gate_groups = int(video_gate_groups)
        self.video_dropout = float(video_dropout)
        self.logit_fusion_source = logit_fusion_source
        self.use_decision_moe = bool(use_decision_moe)
        self.interaction_variant = str(interaction_variant)
        self.joint_num_layers = int(joint_num_layers)
        self.joint_feedforward_size = int(joint_feedforward_size)
        if self.interaction_variant not in {
            "pairwise_residual",
            "joint_attention",
            "text_anchor",
        }:
            raise ValueError(
                f"Unsupported interaction_variant: {self.interaction_variant}"
            )
        if self.interaction_variant != "pairwise_residual" and not self.use_video:
            raise ValueError("trimodal interaction variants require video input")
        if self.use_decision_moe and not self.use_video:
            raise ValueError("decision-level trimodal MoE requires video input")
        if self.video_fusion_variant not in {
            "scalar",
            "group",
            "vector",
            "joint",
            "film",
            "symconcat",
            "logit_residual",
        }:
            raise ValueError(
                f"Unsupported video fusion variant: {self.video_fusion_variant}"
            )
        if (
            self.logit_fusion_source != "learned"
            and self.video_fusion_variant != "logit_residual"
        ):
            raise ValueError(
                "expert logit_fusion_source requires video_fusion_variant="
                "'logit_residual'"
            )
        if (
            not math.isfinite(self.video_residual_scale)
            or self.video_residual_scale < 0
        ):
            raise ValueError("video_residual_scale must be finite and non-negative")
        if not math.isfinite(float(video_gate_bias)):
            raise ValueError("video_gate_bias must be finite")
        if self.video_gate_groups < 1 or 1536 % self.video_gate_groups != 0:
            raise ValueError("video_gate_groups must divide 1536")
        if not 0.0 <= self.video_dropout < 1.0:
            raise ValueError("video_dropout must be in [0, 1)")
        if not math.isfinite(self.avel_temperature) or self.avel_temperature <= 0.0:
            raise ValueError("avel_temperature must be finite and positive")
        if not math.isfinite(self.utec_temperature) or self.utec_temperature <= 0.0:
            raise ValueError("utec_temperature must be finite and positive")
        if use_utec and not self.use_utec:
            raise ValueError("UTEC requires both video input and contrastive learning")
        if self.use_utec and (self.use_vt_contrastive or self.use_avel):
            raise ValueError("UTEC replaces the legacy VT and AVEL objectives")

        print(f"Loading Wav2Vec2 from: {wav2vec2_path}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            wav2vec2_path,
            output_hidden_states=True,
            return_dict=True,
            apply_spec_augment=False,
        )

        if freeze_audio_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            print("Wav2Vec2 feature extractor frozen")

        print(f"Loading RoBERTa from: {roberta_path}")
        self.roberta = RobertaModel.from_pretrained(roberta_path)

        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        freeze_layers = 6
        for i in range(freeze_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

        print(f"--> RoBERTa: Embeddings & First {freeze_layers} layers frozen.")

        self.self_attention_text = BertSelfEncoder(self.config)

        self.audio2text = nn.Linear(768, 768)
        if self.interaction_variant == "pairwise_residual":
            self.audio2text_v2 = nn.Linear(768, 768)
            self.text2audio_attention = BertCrossEncoder(self.config, layer_num=1)
            self.audio2text_attention = BertCrossEncoder(self.config, layer_num=1)
            self.text2text_attention = BertCrossEncoder(self.config, layer_num=1)
            self.gate = nn.Linear(768 * 2, 768)
        else:
            _unused_audio2text_v2 = nn.Linear(768, 768)
            if self.interaction_variant == "text_anchor":
                self.text2audio_attention = BertCrossEncoder(self.config, layer_num=1)
            else:
                _unused_text2audio_attention = BertCrossEncoder(
                    self.config, layer_num=1
                )
            _unused_audio2text_attention = BertCrossEncoder(self.config, layer_num=1)
            _unused_text2text_attention = BertCrossEncoder(self.config, layer_num=1)
            _unused_gate = nn.Linear(768 * 2, 768)
            _reference_text2audio = (
                self.text2audio_attention
                if self.interaction_variant == "text_anchor"
                else _unused_text2audio_attention
            )
            self.pairwise_reference_audio_interaction_parameter_count = sum(
                parameter.numel()
                for module in (
                    self.self_attention_text,
                    self.audio2text,
                    _unused_audio2text_v2,
                    _reference_text2audio,
                    _unused_audio2text_attention,
                    _unused_text2text_attention,
                    _unused_gate,
                )
                for parameter in module.parameters()
            )
            del _unused_audio2text_v2
            if self.interaction_variant == "joint_attention":
                del _unused_text2audio_attention
            del _unused_audio2text_attention
            del _unused_text2text_attention
            del _unused_gate

        self.dropout = nn.Dropout(0.1)
        self.dropout_audio = nn.Dropout(0.1)

        if self.interaction_variant == "pairwise_residual":
            self.ser_attention_pooling = AttentionPooling(
                hidden_size=768 * 2, attention_hidden_size=256, dropout=0.1
            )
            self.classifier = nn.Sequential(
                OrderedDict(
                    [
                        ("dropout", nn.Dropout(0.2)),
                        ("linear", nn.Linear(768 * 2, num_emotions)),
                    ]
                )
            )
        else:

            _unused_ser_attention_pooling = AttentionPooling(
                hidden_size=768 * 2,
                attention_hidden_size=256,
                dropout=0.1,
            )
            _unused_classifier = nn.Sequential(
                OrderedDict(
                    [
                        ("dropout", nn.Dropout(0.2)),
                        ("linear", nn.Linear(768 * 2, num_emotions)),
                    ]
                )
            )
            del _unused_ser_attention_pooling
            del _unused_classifier
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + 27101)
                self.ser_attention_pooling = AttentionPooling(
                    hidden_size=768,
                    attention_hidden_size=256,
                    dropout=0.1,
                )
                self.classifier = nn.Sequential(
                    OrderedDict(
                        [
                            ("dropout", nn.Dropout(0.2)),
                            ("linear", nn.Linear(768, num_emotions)),
                        ]
                    )
                )

        print("SER Head: Attention Pooling + Dropout(0.2) + Linear")

        print("Initializing Enhanced ASR Head (BiLSTM)...")
        self.ctc_head = ASRHead(self.config, ctc_vocab_size)

        self.projection_original = nn.Sequential(
            nn.Linear(768 * 2, 768), ActivateFun("gelu"), nn.Linear(768, 768)
        )

        self.projection_augmented = nn.Sequential(
            nn.Linear(768 * 2, 768), ActivateFun("gelu"), nn.Linear(768, 768)
        )

        self.temperature = 0.07

        if self.use_contrastive:
            if self.use_utec:

                _unused_at_rng_alignment = SimpleContrastiveLearning(
                    hidden_size=768,
                    projection_dim=256,
                    temperature=0.07,
                )
                del _unused_at_rng_alignment

                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(
                        torch.initial_seed() + 27017
                    )
                    self.unified_emotion_contrastive = (
                        UnifiedTrimodalEmotionContrastive(
                            audio_dim=self.config.hidden_size,
                            text_dim=self.config.hidden_size,
                            video_dim=self.config.hidden_size,
                            projection_dim=256,
                            temperature=self.utec_temperature,
                        )
                    )
                print("Unified trimodal emotion contrastive learning enabled")
            else:
                self.contrastive_module = SimpleContrastiveLearning(
                    hidden_size=768, projection_dim=256, temperature=0.07
                )
                print("Contrastive learning module enabled")

        if self.use_video:
            with torch.random.fork_rng(devices=[]):
                self.video2text = nn.Linear(video_feature_dim, self.config.hidden_size)
                self.video2text_v2 = nn.Linear(
                    video_feature_dim, self.config.hidden_size
                )
                self.text2video_attention = BertCrossEncoder(self.config, layer_num=1)
                self.video2text_attention = BertCrossEncoder(self.config, layer_num=1)
                self.video_text2text_attention = BertCrossEncoder(
                    self.config, layer_num=1
                )

                if (
                    self.use_vt_contrastive
                    and self.video_fusion_variant != "logit_residual"
                ):
                    self.video_text_contrastive = VideoTextContrastiveLearning(
                        hidden_size=768,
                        projection_dim=256,
                        temperature=0.07,
                    )
                elif (
                    not self.use_vt_contrastive
                    and self.video_fusion_variant != "logit_residual"
                ):

                    _unused_vt_rng_alignment = VideoTextContrastiveLearning(
                        hidden_size=768,
                        projection_dim=256,
                        temperature=0.07,
                    )
                    del _unused_vt_rng_alignment

                fusion_size = self.config.hidden_size * 2
                if self.video_fusion_variant == "logit_residual":
                    self.video_text_attention_pooling = AttentionPooling(
                        hidden_size=fusion_size,
                        attention_hidden_size=256,
                        dropout=0.1,
                    )
                    self.reliability_logit_fusion = ReliabilityAwareLogitResidual(
                        num_classes=num_emotions,
                        at_dim=fusion_size,
                        vt_dim=fusion_size,
                        video_dim=self.config.hidden_size,
                        video_dropout=self.video_dropout,
                        residual_scale=self.video_residual_scale,
                        gate_bias=video_gate_bias,
                        fusion_source=logit_fusion_source,
                        evidence_temperature=video_evidence_temperature,
                        evidence_target_scale=video_evidence_target_scale,
                        evidence_warmup_epochs=video_evidence_warmup_epochs,
                        evidence_ramp_epochs=video_evidence_ramp_epochs,
                    )
                elif self.video_fusion_variant == "symconcat":

                    self.video_symmetric_gate = nn.Linear(
                        self.config.hidden_size * 2, self.config.hidden_size
                    )
                    self.symconcat_attention_pooling = AttentionPooling(
                        hidden_size=fusion_size * 2,
                        attention_hidden_size=256,
                        dropout=0.1,
                    )
                    self.symconcat_classifier = nn.Sequential(
                        OrderedDict(
                            [
                                ("dropout", nn.Dropout(0.2)),
                                ("linear", nn.Linear(fusion_size * 2, num_emotions)),
                            ]
                        )
                    )

                    for parameter in self.ser_attention_pooling.parameters():
                        parameter.requires_grad = False
                    for parameter in self.classifier.parameters():
                        parameter.requires_grad = False
                else:
                    if self.video_fusion_variant == "joint":
                        residual_input_size = fusion_size * 2
                        residual_output_size = fusion_size
                    elif self.video_fusion_variant == "film":
                        residual_input_size = fusion_size
                        residual_output_size = fusion_size * 2
                    else:
                        residual_input_size = fusion_size
                        residual_output_size = fusion_size
                    if self.video_fusion_variant == "vector":
                        gate_output_size = fusion_size
                    elif self.video_fusion_variant == "group":
                        gate_output_size = self.video_gate_groups
                    else:
                        gate_output_size = 1
                    self.video_residual_norm = nn.LayerNorm(residual_input_size)
                    self.video_residual_projection = nn.Linear(
                        residual_input_size, residual_output_size
                    )
                    self.video_residual_gate = nn.Linear(
                        fusion_size * 2, gate_output_size
                    )
                    nn.init.zeros_(self.video_residual_projection.weight)
                    nn.init.zeros_(self.video_residual_projection.bias)
                    nn.init.zeros_(self.video_residual_gate.weight)
                    nn.init.constant_(self.video_residual_gate.bias, video_gate_bias)

                if (
                    self.use_vt_contrastive
                    and self.video_fusion_variant == "logit_residual"
                ):
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(
                            torch.initial_seed() + 27011
                        )
                        self.video_text_contrastive = VideoTextContrastiveLearning(
                            hidden_size=768,
                            projection_dim=256,
                            temperature=0.07,
                        )
                if self.use_avel:
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(
                            torch.initial_seed() + 27013
                        )
                        self.audio_video_emotion_alignment = AVEL(
                            audio_dim=self.config.hidden_size,
                            video_dim=self.config.hidden_size,
                            projection_dim=256,
                            temperature=self.avel_temperature,
                        )

        if self.use_video and self.interaction_variant != "pairwise_residual":
            reference_video_modules = (
                self.video2text,
                self.video2text_v2,
                self.text2video_attention,
                self.video2text_attention,
                self.video_text2text_attention,
                self.video_residual_norm,
                self.video_residual_projection,
                self.video_residual_gate,
            )
            self.pairwise_reference_video_interaction_parameter_count = sum(
                parameter.numel()
                for module in reference_video_modules
                for parameter in module.parameters()
            )
            removable = [
                "video2text_v2",
                "video2text_attention",
                "video_text2text_attention",
                "video_residual_norm",
                "video_residual_projection",
                "video_residual_gate",
            ]
            if self.interaction_variant == "joint_attention":
                removable.append("text2video_attention")
            for name in removable:
                delattr(self, name)

            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + 27103)
                if self.interaction_variant == "joint_attention":
                    self.joint_trimodal_attention = JointTrimodalAttention(
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_attention_heads,
                        num_layers=self.joint_num_layers,
                        feedforward_size=self.joint_feedforward_size,
                        dropout=self.config.hidden_dropout_prob,
                        gradient_checkpointing=True,
                    )
                else:
                    self.text_anchor_residual = TextAnchorResidual(
                        hidden_size=self.config.hidden_size
                    )

            self.pairwise_reference_interaction_parameter_count = (
                self.pairwise_reference_audio_interaction_parameter_count
                + self.pairwise_reference_video_interaction_parameter_count
            )
            interaction_modules = [
                self.self_attention_text,
                self.audio2text,
                self.video2text,
            ]
            if self.interaction_variant == "joint_attention":
                interaction_modules.append(self.joint_trimodal_attention)
            else:
                interaction_modules.extend(
                    (
                        self.text2audio_attention,
                        self.text2video_attention,
                        self.text_anchor_residual,
                    )
                )
            self.interaction_parameter_count = sum(
                parameter.numel()
                for module in interaction_modules
                for parameter in module.parameters()
            )
            self.interaction_parameter_ratio_vs_reference = (
                self.interaction_parameter_count
                / self.pairwise_reference_interaction_parameter_count
            )

        self.decision_moe = None
        if self.use_decision_moe:
            self.decision_moe = DecisionLevelEmotionMoE(
                hidden_size=self.config.hidden_size,
                num_classes=num_emotions,
                dropout=0.2,
            )

        self._video_parameter_prefixes = (
            "video2text.",
            "video2text_v2.",
            "text2video_attention.",
            "video2text_attention.",
            "video_text2text_attention.",
            "video_text_contrastive.",
            "audio_video_emotion_alignment.",
            "video_text_attention_pooling.",
            "video_symmetric_gate.",
            "symconcat_attention_pooling.",
            "symconcat_classifier.",
            "video_residual_norm.",
            "video_residual_projection.",
            "video_residual_gate.",
            "reliability_logit_fusion.",
            "text_anchor_residual.video_projection.",
        )

        self._base_requires_grad = {
            name: parameter.requires_grad for name, parameter in self.named_parameters()
        }
        self.audio_text_frozen = False

        print(
            f"MMER Model initialized: {num_emotions} emotions, {ctc_vocab_size} CTC tokens"
        )

    def set_audio_text_trainable(self, trainable):

        trainable = bool(trainable)
        affected = 0
        trainable_count = 0
        for name, parameter in self.named_parameters():
            if name.startswith(self._video_parameter_prefixes):
                continue
            affected += parameter.numel()
            base_value = self._base_requires_grad.get(name, parameter.requires_grad)
            parameter.requires_grad = bool(trainable and base_value)
            if parameter.requires_grad:
                trainable_count += parameter.numel()
        self.audio_text_frozen = not trainable
        return {
            "audio_text_parameter_count": int(affected),
            "audio_text_trainable_count": int(trainable_count),
            "audio_text_frozen": bool(self.audio_text_frozen),
        }

    @property
    def video_evidence_selection_active_epoch(self):
        if (
            not self.use_video
            or self.video_fusion_variant != "logit_residual"
            or self.logit_fusion_source == "learned"
        ):
            return 1
        return self.reliability_logit_fusion.evidence_selection_active_epoch

    def set_video_evidence_epoch(self, epoch):

        if (
            not self.use_video
            or self.video_fusion_variant != "logit_residual"
            or self.logit_fusion_source == "learned"
        ):
            return 0.0
        return self.reliability_logit_fusion.set_evidence_epoch(epoch)

    def enforce_frozen_audio_text_eval(self):

        if not self.audio_text_frozen:
            return
        video_roots = {
            prefix.split(".", 1)[0] for prefix in self._video_parameter_prefixes
        }
        for name, module in self.named_children():
            if name not in video_roots:
                module.eval()

    def _forward_joint_interaction(
        self,
        audio_hidden,
        text_hidden,
        text_enhanced,
        audio_projected,
        audio_sequence_mask,
        extended_audio_mask,
        text_attention_mask,
        text_target_mask,
        ctc_logits,
        mode,
        video_features,
        video_attention_mask,
        video_quality,
        emotion_labels,
    ):

        if video_features is None or video_attention_mask is None:
            raise ValueError("video_features and video_attention_mask are required")
        if (
            video_features.dim() != 3
            or video_features.size(-1) != self.video2text.in_features
        ):
            raise ValueError(
                "video_features must have shape (batch, frames, video_feature_dim)"
            )
        if video_attention_mask.shape != video_features.shape[:2]:
            raise ValueError("video_attention_mask shape mismatch")

        video_features = video_features.to(dtype=text_enhanced.dtype)
        video_attention_mask = video_attention_mask.to(device=video_features.device)
        video_valid = video_attention_mask.bool().any(dim=1)
        if video_quality is None:
            video_quality = text_enhanced.new_ones(text_enhanced.size(0))
        else:
            video_quality = (
                video_quality.to(
                    device=text_enhanced.device,
                    dtype=text_enhanced.dtype,
                )
                .view(-1)
                .clamp(0.0, 1.0)
            )
        video_projected = self.video2text(video_features)
        logit_fusion_outputs = None
        logit_auxiliary_outputs = None

        if self.interaction_variant == "joint_attention":
            joint = self.joint_trimodal_attention(
                text_tokens=text_enhanced,
                audio_tokens=audio_projected,
                video_tokens=video_projected,
                text_mask=text_attention_mask,
                text_target_mask=text_target_mask,
                audio_mask=audio_sequence_mask,
                video_mask=video_attention_mask,
                video_quality=video_quality,
            )
            fused_token_features = joint["text_tokens"]
            pooled_features = joint["pooled"]
            attention_weights = None
            emotion_logits = self.classifier(pooled_features)
            video_gate_values = joint["video_reliability"]
        elif self.interaction_variant == "text_anchor":
            safe_video_mask = video_attention_mask.long().clone()
            safe_video_mask[~video_valid, 0] = 1
            extended_video_mask = safe_video_mask.unsqueeze(1).unsqueeze(2)
            extended_video_mask = extended_video_mask.to(dtype=text_enhanced.dtype)
            extended_video_mask = (1.0 - extended_video_mask) * -10000.0
            audio_context = self.text2audio_attention(
                text_enhanced, audio_projected, extended_audio_mask
            )[-1]
            video_context = self.text2video_attention(
                text_enhanced, video_projected, extended_video_mask
            )[-1]
            anchor = self.text_anchor_residual(
                text_tokens=text_enhanced,
                audio_context=audio_context,
                video_context=video_context,
                text_target_mask=text_target_mask,
                video_quality=video_quality,
                video_valid=video_valid,
            )
            fused_token_features = anchor["tokens"]
            pooled_features, attention_weights = self.ser_attention_pooling(
                fused_token_features,
                attention_mask=text_target_mask,
            )
            emotion_logits = self.classifier(pooled_features)
            video_gate_values = anchor["video_reliability"]
        else:
            raise AssertionError(
                "joint interaction helper called for inherited interaction"
            )

        decision_moe_outputs = None
        if self.use_decision_moe:
            decision_moe_outputs = self.decision_moe(
                audio_tokens=audio_hidden,
                audio_mask=audio_sequence_mask,
                text_tokens=text_enhanced,
                text_mask=text_target_mask,
                video_tokens=video_projected,
                video_mask=video_attention_mask,
                fused_logits=emotion_logits,
            )
            emotion_logits = decision_moe_outputs["logits"]

        if emotion_labels is not None and (
            emotion_labels.dim() != 1
            or emotion_labels.size(0) != emotion_logits.size(0)
        ):
            raise ValueError("emotion_labels must have shape (batch,)")

        contrastive_outputs = None
        if self.use_contrastive and mode == "train":
            if not self.use_utec:
                raise AssertionError("joint interaction requires UTEC")
            if emotion_labels is None:
                raise ValueError("emotion_labels are required when UTEC is enabled")
            utec_outputs = self.unified_emotion_contrastive(
                audio_hidden=audio_hidden,
                text_hidden=text_hidden,
                video_hidden=video_projected,
                audio_attention_mask=audio_sequence_mask,
                text_attention_mask=text_target_mask,
                video_attention_mask=video_attention_mask,
                emotion_labels=emotion_labels,
                video_quality=video_quality,
            )
            zero = utec_outputs["loss"].new_zeros(())
            contrastive_outputs = {
                "total_contrastive_loss": utec_outputs["loss"],
                "loss_utec": utec_outputs["loss"],
                "loss_utec_audio_text": utec_outputs["loss_audio_text"],
                "loss_utec_audio_video": utec_outputs["loss_audio_video"],
                "loss_utec_text_video": utec_outputs["loss_text_video"],
                "loss_audio_text": zero,
                "loss_video_text": zero,
                "loss_fused_audio": zero,
                "loss_fused_text": zero,
            }

        return {
            "emotion_logits": emotion_logits,
            "ctc_logits": ctc_logits,
            "fused_features": pooled_features,
            "fused_token_features": fused_token_features,
            "audio_hidden": audio_hidden,
            "text_hidden": text_hidden,
            "text_target_mask": text_target_mask,
            "contrastive_outputs": contrastive_outputs,
            "loss_audio_video_emotion": emotion_logits.new_zeros(()),
            "logit_fusion_outputs": logit_fusion_outputs,
            "logit_auxiliary_losses": logit_auxiliary_outputs,
            "decision_moe_outputs": decision_moe_outputs,
            "video_projected": video_projected,
            "video_attention_mask": video_attention_mask,
            "video_gate": video_gate_values,
            "attention_weights": attention_weights,
        }

    def forward(
        self,
        audio_input,
        text_input_ids,
        text_attention_mask,
        text_target_mask=None,
        audio_attention_mask=None,
        mode="train",
        video_features=None,
        video_attention_mask=None,
        video_quality=None,
        emotion_labels=None,
        rank_margin=0.0,
    ):

        if text_target_mask is None:
            text_target_mask = text_attention_mask
        if text_target_mask.shape != text_attention_mask.shape:
            raise ValueError("text_target_mask must match text_attention_mask shape")
        text_target_mask = text_target_mask.to(
            device=text_attention_mask.device,
            dtype=text_attention_mask.dtype,
        )
        if bool((text_target_mask > text_attention_mask).any()):
            raise ValueError("text_target_mask must be a subset of text_attention_mask")
        if bool((text_target_mask.sum(dim=1) == 0).any()):
            raise ValueError("every sample must contain target text tokens")

        audio_outputs = self.wav2vec2(audio_input, attention_mask=audio_attention_mask)
        audio_hidden = audio_outputs.last_hidden_state

        ctc_logits = self.ctc_head(audio_hidden)

        text_outputs = self.roberta(text_input_ids, attention_mask=text_attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_hidden = self.dropout(text_hidden)

        extended_text_mask = text_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_text_mask = extended_text_mask.to(dtype=next(self.parameters()).dtype)
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0

        text_self_att = self.self_attention_text(text_hidden, extended_text_mask)
        text_enhanced = text_self_att[-1]

        audio_projected = self.audio2text(audio_hidden)
        audio_projected_v2 = (
            self.audio2text_v2(audio_hidden)
            if self.interaction_variant == "pairwise_residual"
            else None
        )

        if audio_attention_mask is not None:

            audio_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                audio_attention_mask.sum(-1)
            ).long()
            max_audio_len = audio_hidden.size(1)
            audio_sequence_mask = torch.zeros(
                audio_hidden.size(0),
                max_audio_len,
                dtype=torch.long,
                device=audio_hidden.device,
            )
            for i, length in enumerate(audio_lengths):
                audio_sequence_mask[i, :length] = 1
        else:
            audio_sequence_mask = torch.ones(
                audio_hidden.size(0),
                audio_hidden.size(1),
                dtype=torch.long,
                device=audio_hidden.device,
            )

        extended_audio_mask = audio_sequence_mask.unsqueeze(1).unsqueeze(2)
        extended_audio_mask = extended_audio_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0

        if self.interaction_variant != "pairwise_residual":
            return self._forward_joint_interaction(
                audio_hidden=audio_hidden,
                text_hidden=text_hidden,
                text_enhanced=text_enhanced,
                audio_projected=audio_projected,
                audio_sequence_mask=audio_sequence_mask,
                extended_audio_mask=extended_audio_mask,
                text_attention_mask=text_attention_mask,
                text_target_mask=text_target_mask,
                ctc_logits=ctc_logits,
                mode=mode,
                video_features=video_features,
                video_attention_mask=video_attention_mask,
                video_quality=video_quality,
                emotion_labels=emotion_labels,
            )

        text2audio_output = self.text2audio_attention(
            text_enhanced, audio_projected, extended_audio_mask
        )
        text2audio_features = text2audio_output[-1]

        audio2text_output = self.audio2text_attention(
            audio_projected_v2, text_enhanced, extended_text_mask
        )
        audio2text_features = audio2text_output[-1]

        text2text_output = self.text2text_attention(
            text_enhanced, audio2text_features, extended_audio_mask
        )
        text_final = text2text_output[-1]

        merge_representation = torch.cat((text_final, text2audio_features), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))
        gated_audio_features = torch.mul(gate_value, text2audio_features)

        fused_features = torch.cat((text_final, gated_audio_features), dim=-1)

        video_projected = None
        video_text_features = None
        logit_fusion_outputs = None
        video_gate_values = fused_features.new_zeros(
            fused_features.size(0), fused_features.size(1), 1
        )
        if self.use_video:
            if video_features is None or video_attention_mask is None:
                raise ValueError(
                    "video_features and video_attention_mask are required when use_video=True"
                )
            if (
                video_features.dim() != 3
                or video_features.size(-1) != self.video2text.in_features
            ):
                raise ValueError(
                    "video_features must have shape (batch, frames, video_feature_dim)"
                )
            if video_attention_mask.shape != video_features.shape[:2]:
                raise ValueError(
                    "video_attention_mask must match the first two video feature dimensions"
                )

            video_features = video_features.to(dtype=text_enhanced.dtype)
            video_attention_mask = video_attention_mask.to(device=video_features.device)
            video_valid = video_attention_mask.bool().any(dim=1)

            safe_video_mask = video_attention_mask.long().clone()
            safe_video_mask[~video_valid, 0] = 1
            extended_video_mask = safe_video_mask.unsqueeze(1).unsqueeze(2)
            extended_video_mask = extended_video_mask.to(dtype=text_enhanced.dtype)
            extended_video_mask = (1.0 - extended_video_mask) * -10000.0

            video_projected = self.video2text(video_features)
            video_projected_v2 = self.video2text_v2(video_features)
            video_text_anchor = text_enhanced
            if self.video_fusion_variant == "logit_residual":

                video_text_anchor = text_enhanced.detach()

            text2video_output = self.text2video_attention(
                video_text_anchor, video_projected, extended_video_mask
            )
            text2video_features = text2video_output[-1]

            video2text_output = self.video2text_attention(
                video_projected_v2, video_text_anchor, extended_text_mask
            )
            video2text_features = video2text_output[-1]

            video_text2text_output = self.video_text2text_attention(
                video_text_anchor, video2text_features, extended_video_mask
            )
            video_text_final = video_text2text_output[-1]
            video_text_features = torch.cat(
                (video_text_final, text2video_features), dim=-1
            )

            if self.video_fusion_variant == "logit_residual":

                pass
            elif self.video_fusion_variant == "symconcat":

                internal_video_gate = torch.sigmoid(
                    self.video_symmetric_gate(video_text_features)
                )
                gated_video_features = internal_video_gate * text2video_features
                symmetric_video_features = torch.cat(
                    (video_text_final, gated_video_features), dim=-1
                )
                fused_features = torch.cat(
                    (fused_features, symmetric_video_features), dim=-1
                )
                video_gate_values = internal_video_gate.mean(dim=-1, keepdim=True)
            else:
                fusion_context = torch.cat(
                    (fused_features, video_text_features), dim=-1
                )
                if self.video_fusion_variant == "joint":
                    residual_input = fusion_context
                else:
                    residual_input = video_text_features
                projected_video = self.video_residual_projection(
                    self.video_residual_norm(residual_input)
                )
                if self.video_fusion_variant == "film":
                    video_gamma, video_beta = projected_video.chunk(2, dim=-1)
                    video_delta = torch.tanh(video_gamma) * fused_features + video_beta
                else:
                    video_delta = projected_video

                internal_video_gate = torch.sigmoid(
                    self.video_residual_gate(fusion_context)
                )
                if self.video_fusion_variant == "group":
                    internal_video_gate = internal_video_gate.repeat_interleave(
                        video_delta.size(-1) // self.video_gate_groups, dim=-1
                    )
                if video_quality is None:
                    quality = internal_video_gate.new_ones(
                        internal_video_gate.size(0), 1, 1
                    )
                else:
                    quality = (
                        video_quality.to(
                            device=internal_video_gate.device,
                            dtype=internal_video_gate.dtype,
                        )
                        .view(-1, 1, 1)
                        .clamp(0.0, 1.0)
                    )
                valid_scale = video_valid.to(internal_video_gate.dtype).view(-1, 1, 1)
                internal_video_gate = internal_video_gate * quality * valid_scale
                fused_features = (
                    fused_features
                    + self.video_residual_scale * internal_video_gate * video_delta
                )
                video_gate_values = internal_video_gate.mean(dim=-1, keepdim=True)

        if self.use_video and self.video_fusion_variant == "symconcat":
            ser_pooling = self.symconcat_attention_pooling
            ser_classifier = self.symconcat_classifier
        else:
            ser_pooling = self.ser_attention_pooling
            ser_classifier = self.classifier
        pooled_features, attention_weights = ser_pooling(
            fused_features, attention_mask=text_target_mask
        )

        emotion_logits = ser_classifier(pooled_features)

        if self.use_video and self.video_fusion_variant == "logit_residual":
            video_text_pooled, _ = self.video_text_attention_pooling(
                video_text_features,
                attention_mask=text_target_mask,
            )
            logit_fusion_outputs = self.reliability_logit_fusion(
                at_pooled=pooled_features,
                at_logits=emotion_logits,
                vt_pooled=video_text_pooled,
                video_tokens=video_projected,
                video_mask=video_attention_mask,
            )
            emotion_logits = logit_fusion_outputs["fused_logits"]
            video_gate_values = logit_fusion_outputs["video_gate"]

        decision_moe_outputs = None
        if self.use_decision_moe:
            decision_moe_outputs = self.decision_moe(
                audio_tokens=audio_hidden,
                audio_mask=audio_sequence_mask,
                text_tokens=text_enhanced,
                text_mask=text_target_mask,
                video_tokens=video_projected,
                video_mask=video_attention_mask,
                fused_logits=emotion_logits,
            )
            emotion_logits = decision_moe_outputs["logits"]

        logit_auxiliary_outputs = None
        if emotion_labels is not None:
            if emotion_labels.dim() != 1 or emotion_labels.size(
                0
            ) != emotion_logits.size(0):
                raise ValueError("emotion_labels must have shape (batch,)")
            if logit_fusion_outputs is not None:
                logit_auxiliary_outputs = reliability_auxiliary_losses(
                    logit_fusion_outputs,
                    emotion_labels,
                    rank_margin=rank_margin,
                )

        contrastive_outputs = None
        if self.use_contrastive and mode == "train":
            if self.use_utec:
                if emotion_labels is None:
                    raise ValueError("emotion_labels are required when UTEC is enabled")
                if video_projected is None or video_attention_mask is None:
                    raise ValueError("video input is required when UTEC is enabled")
                utec_outputs = self.unified_emotion_contrastive(
                    audio_hidden=audio_hidden,
                    text_hidden=text_hidden,
                    video_hidden=video_projected,
                    audio_attention_mask=audio_sequence_mask,
                    text_attention_mask=text_target_mask,
                    video_attention_mask=video_attention_mask,
                    emotion_labels=emotion_labels,
                    video_quality=video_quality,
                )
                zero = utec_outputs["loss"].new_zeros(())
                contrastive_outputs = {
                    "total_contrastive_loss": utec_outputs["loss"],
                    "loss_utec": utec_outputs["loss"],
                    "loss_utec_audio_text": utec_outputs["loss_audio_text"],
                    "loss_utec_audio_video": utec_outputs["loss_audio_video"],
                    "loss_utec_text_video": utec_outputs["loss_text_video"],
                    "loss_audio_text": zero,
                    "loss_video_text": zero,
                    "loss_fused_audio": zero,
                    "loss_fused_text": zero,
                }

            elif isinstance(self.contrastive_module, SimpleContrastiveLearning):

                contrastive_loss = self.contrastive_module(
                    audio_hidden=audio_hidden,
                    text_hidden=text_hidden,
                    audio_attention_mask=extended_audio_mask.squeeze(1)
                    .squeeze(1)
                    .ne(-10000.0)
                    .long(),
                    text_attention_mask=text_target_mask,
                )
                video_text_loss = contrastive_loss.new_zeros(())
                if self.use_vt_contrastive:
                    video_text_loss = self.video_text_contrastive(
                        video_hidden=video_projected,
                        text_hidden=(
                            text_hidden.detach()
                            if self.video_fusion_variant == "logit_residual"
                            else text_hidden
                        ),
                        video_attention_mask=video_attention_mask,
                        text_attention_mask=text_target_mask,
                        video_quality=video_quality,
                    )
                contrastive_outputs = {
                    "total_contrastive_loss": contrastive_loss + video_text_loss,
                    "loss_audio_text": contrastive_loss,
                    "loss_video_text": video_text_loss,
                    "loss_fused_audio": torch.tensor(
                        0.0, device=contrastive_loss.device
                    ),
                    "loss_fused_text": torch.tensor(
                        0.0, device=contrastive_loss.device
                    ),
                }
            else:

                contrastive_outputs = self.contrastive_module.forward_with_fused(
                    audio_hidden=audio_hidden,
                    text_hidden=text_hidden,
                    fused_features=fused_features,
                    audio_attention_mask=extended_audio_mask.squeeze(1)
                    .squeeze(1)
                    .ne(-10000.0)
                    .long(),
                    text_attention_mask=text_target_mask,
                )

        loss_audio_video_emotion = emotion_logits.new_zeros(())
        if self.use_avel and mode == "train":
            if emotion_labels is None:
                raise ValueError("emotion_labels are required when AVEL is enabled")
            loss_audio_video_emotion = self.audio_video_emotion_alignment(
                audio_hidden=audio_hidden.detach(),
                video_hidden=video_projected,
                audio_attention_mask=audio_sequence_mask,
                video_attention_mask=video_attention_mask,
                emotion_labels=emotion_labels,
            )

        return {
            "emotion_logits": emotion_logits,
            "ctc_logits": ctc_logits,
            "fused_features": pooled_features,
            "audio_hidden": audio_hidden,
            "text_hidden": text_hidden,
            "text_target_mask": text_target_mask,
            "contrastive_outputs": contrastive_outputs,
            "loss_audio_video_emotion": loss_audio_video_emotion,
            "logit_fusion_outputs": logit_fusion_outputs,
            "logit_auxiliary_losses": logit_auxiliary_outputs,
            "decision_moe_outputs": decision_moe_outputs,
            "video_projected": video_projected,
            "video_attention_mask": video_attention_mask,
            "video_gate": video_gate_values,
            "attention_weights": attention_weights,
        }
