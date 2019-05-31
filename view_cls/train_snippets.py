import time
import copy
import torch
import logging

from save_checkpoint import save_checkpoint
def train_model(model, criterion, optimizer, scheduler,dataloaders, num_epochs,device):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    
                    outputs = torch.sigmoid(outputs)
                    preds = (outputs >= 0.5).float()
                   

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects.append(torch.sum(preds == labels.data, dim=0))

            epoch_loss = running_loss / dataset_sizes[phase]
            running_corrects = torch.stack(running_corrects,dim=0)
            epoch_acc = torch.mean(torch.sum(running_corrects,dim=0).float() / float(dataset_sizes[phase])) #mean acc
            
            if phase == 'train':
                scheduler.step(epoch_loss)

            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print('saving checkpoint as best model')
                logging.info('saving checkpoint as best model')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'best_loss': epoch_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=True)

        print()
        logging.info('')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model