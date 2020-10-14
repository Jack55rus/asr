import torch
import editdistance as ed

from data.text import CharacterTextEncoder 

class Evaluater(object):
    def __init__(self, vocab_list):
        self.tokinazer = CharacterTextEncoder.load_from_file(vocab_list)

    def evaluate(self, pred, target, visual=False, visual_batch=False):
        bs = pred.size(0)
        total_err = 0
        if visual_batch:
            pred_list = []
            true_list = []
        for b in range(bs):
            # print('b', b)
            pred_sen = pred[b]
            true_sen = target[b]
            pred_sen = self.tokinazer.decode(pred_sen.cpu().detach().numpy(), target=False)
            true_sen = self.tokinazer.decode(true_sen.cpu().detach().numpy(), target=True)
            if visual and b==0:
                print('Pred:',pred_sen)
                print('True:',true_sen)
                print('===============')
            if visual_batch:
                pred_list.append(pred_sen)
                true_list.append(true_sen)
            dist = ed.eval(pred_sen,true_sen)/len(true_sen)
            total_err += dist
        total_err /= bs
        if visual_batch:
            return total_err, pred_list, true_list
        else:
            return total_err