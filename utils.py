# Aux functions

def compute_accuracy(preds, labels):
    pred_labels = preds.argmax(dim=1, keepdim=True)
    acc = pred_labels.eq(labels.view_as(pred_labels)).sum().item() / len(
        pred_labels)
    return acc
