import torch
import torch.nn as nn
from transformers import BertForPreTraining, BertConfig, BertForSequenceClassification
from typing import Optional, Tuple
from torchinfo import summary


class EmotionBert(BertForPreTraining):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, 28)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            next_sentence_label: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        outputs = self.bert.forward(input_ids, attention_mask=attention_mask)
        classifier_output = self.classifier(outputs.pooler_output)

        return classifier_output


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model_alt = EmotionBert.from_pretrained('bert-base-uncased')

summary(
    model=model,
    input_data=torch.ones(1, 512).long(),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

summary(
    model=model_alt,
    input_data=torch.ones(1, 512).long(),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)