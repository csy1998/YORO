# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/21 16:17
@Description: Bert for Non-Autoregressive Grammar error correction
"""
import torch
from torch import nn
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForNonAutoregressiveGec(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForNonAutoregressiveGec, self).__init__(config)

        #self.vocab_size = config.vocab_size
        self.vocab_size = 16250
        
        self.alpha_add = config.alpha_add
        self.alpha_del = config.alpha_del
        self.alpha_vocab = config.alpha_vocab
        self.alpha_position = config.alpha_position

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.del_classifier = nn.Linear(config.hidden_size, 1)
        self.add_classifier = nn.Linear(config.hidden_size, 1)
        self.vocab_classifier = nn.Linear(config.hidden_size, self.vocab_size)
        self.position_classifier = nn.Linear(config.hidden_size, self.vocab_size)

        self.apply(self.init_weights)

    # TODO 按batch输入的对齐问题
    # add_vocab_index: [batch_size, max_add_num]
    # if add_vocab_index[sentence_id][add_id] > -1 then active
    def forward(self, input_ids, attention_mask, del_label=None, add_label=None, add_vocab_index=None,
                add_vocab_position=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        del_logit = self.del_classifier(sequence_output)
        add_logit = self.add_classifier(sequence_output)
        # vocab_logits = self.vocab_classifier(sequence_output)
        # position_logits = self.position_classifier(sequence_output)

        del_logit = torch.sigmoid(del_logit)
        add_logit = torch.sigmoid(add_logit)
        # vocab_logits = torch.sigmoid(vocab_logits)

        del_logit = del_logit.view([input_ids.size()[0], input_ids.size()[1]])
        add_logit = add_logit.view([input_ids.size()[0], input_ids.size()[1]])

        #outputs = (del_logit, add_logit, vocab_logits, position_logits)
        outputs = (del_logit, add_logit)

        if del_label is not None:
            binary_clf_loss = BCELoss(reduction="none")
            mse_loss = MSELoss(reduction="none")

            # del label loss
            del_mask = attention_mask == 1  # [batch_size * max_length]
            del_loss = binary_clf_loss(del_logit, del_label) # [batch_size, max_length]
            del_loss.masked_fill_(~del_mask, 0.)
            del_loss = torch.mean(del_loss)
            
            # add label loss
            add_mask = attention_mask == 1  # [batch_size * max_length]
            add_loss = binary_clf_loss(add_logit, add_label) # [batch_size, max_length]
            add_loss.masked_fill_(~add_mask, 0.)
            add_loss = torch.mean(add_loss)
            
            # add vocab loss
            # vocab_logits = vocab_logits.view([input_ids.size()[0], -1])               # [batch_size, max_length * vocab_size]
            # vocab_labels = torch.zeros_like(vocab_logits)                             # [batch_size, max_length * vocab_size]
            # vocab_logit_index = add_vocab_index + (add_vocab_index == -1).long()      # [batch_size, max_add_num]
            # vocab_labels.scatter_(1, vocab_logit_index, 1)                            # [batch_size, max_length * vocab_size]
            # vocab_labels.index_fill_(1, torch.tensor([0]).to("cuda"), 0)              # [batch_size, max_length * vocab_size]
            # vocab_loss = binary_clf_loss(vocab_logits, vocab_labels)                  # [batch_size, max_length * vocab_size]
            # vocab_loss = torch.mean(vocab_loss)


            # # add vocab loss (mask no add place)
            # vocab_mask = add_label.view(-1) == 1   # [batch_size * max_length]
            # vocab_labels = torch.zeros_like(vocab_logits).view([input_ids.size()[0], -1])  
            # vocab_logits = vocab_logits.view(-1, self.vocab_size)[vocab_mask]                    
            
            # # vocab_index_mask = add_vocab_index != -1
            # # vocab_logit_index = add_vocab_index.masked_fill_(~vocab_index_mask, 0)
            # vocab_logit_index = add_vocab_index + (add_vocab_index == -1).long()    # [batch_size, max_add_num]
            # vocab_labels.scatter_(1, vocab_logit_index, 1)                          # [batch_size, max_length * vocab_size]
            # vocab_labels[:,0] = 0                                                   # [batch_size, max_length * vocab_size]
            # vocab_labels = vocab_labels.view(-1, self.vocab_size)
            # vocab_labels = vocab_labels[vocab_mask]
            # vocab_loss = binary_clf_loss(vocab_logits, vocab_labels)                # [-1, vocab_size]
            # vocab_loss = torch.mean(vocab_loss)                    
        

            # # add position loss
            # position_logits = position_logits.view([input_ids.size()[0], -1])           # [batch_size, max_length * vocab_size]
            # position_loss_mask = add_vocab_index != -1                                  # [batch_size, max_add_num]
            # position_logit_index = add_vocab_index + (add_vocab_index == -1).long()     # [batch_size, max_add_num]
            # active_position_logits = position_logits.gather(1, position_logit_index)    # [batch_size, max_add_num]
            # position_loss_all = mse_loss(active_position_logits, add_vocab_position)    # [batch_size, max_add_num]
            # position_loss_all.masked_fill_(~position_loss_mask, 0.)
            # position_loss = torch.mean(position_loss_all)
            

            # weighted total loss
            loss = self.alpha_del * del_loss + \
                self.alpha_add * add_loss 
                #self.alpha_vocab * vocab_loss + \
                #self.alpha_position * position_loss
    
            #loss_all = (del_loss, add_loss, vocab_loss, position_loss)
            loss_all = (del_loss, add_loss)
            outputs = (loss, loss_all) + outputs

        return outputs  # (loss), (logits), (hidden_states), (attentions)
