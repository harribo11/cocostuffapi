import torch
import sys
import wandb
from scipy.io import savemat

#model should already be on GPU
def train_epoch(model, optimizer, dataLoader, lossFunction, showLoss=False, showMemory=False, batchSize=32, miniBatchSize=1):
    examplesThisBatch = 0
    accumulatedLoss = 0
    for (idx, batch) in enumerate(dataLoader):
        if showMemory:
          checkMemory()
        batch_x = batch[0].cuda()
        

        outputs = model.forward(batch_x)
        del batch_x

        batch_y = batch[1].to(torch.int64).cuda()

        loss = lossFunction(outputs,batch_y)
        del batch_y
        del outputs

        if showLoss:
            print(loss.item())

        if showMemory:
          checkMemory()

        loss = loss*miniBatchSize/batchSize
        accumulatedLoss += loss.item()
        loss.backward()
        del loss

        examplesThisBatch += miniBatchSize
        
        if showMemory:
          checkMemory()

        if examplesThisBatch >= batchSize:
          wandb.log({'Training Loss': accumulatedLoss})
          optimizer.step()
          optimizer.zero_grad()
          examplesThisBatch = 0
          accumulatedLoss = 0

def evaluate(model, optimizer, dataLoader, lossFunction, folder=None, showLoss=False, showMemory=False, miniBatchSize=1):
    examplesThisBatch = 0
    accumulatedLoss = 0
    with torch.no_grad():
      for (idx, batch) in enumerate(dataLoader):
          batch_x = batch[0]
          batch_y = batch[1].to(torch.int64)
          padding = batch[2]
          imgIds = batch[3]

          outputs = model.forward(batch_x.cuda())
          del batch_x

          loss = lossFunction(outputs,batch_y.cuda())
          del batch_y

          outputs = torch.argmax(outputs,1)
          
          if folder is not None:
            for i in range(outputs.size()[0]):
              padLeft, padRight, padTop, padBottom = padding[i]
              h = outputs.size()[1]
              w = outputs.size()[2]
              mdict = {'S':outputs[i,padTop:h-padBottom,padLeft:w-padRight].cpu().numpy()}
              path = folder + '/' + str(imgIds[i]).zfill(12) + '.mat'
              savemat(path, mdict)
            
          loss = loss*miniBatchSize

          if showLoss:
              print(loss.item())

          accumulatedLoss += loss.item() * miniBatchSize
          examplesThisBatch += miniBatchSize
    wandb.log({'Test Loss': loss/examplesThisBatch})
  



def checkMemory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)

    if r == 0:
      print("No GPU Memory Allocated")
      return

    f0 = a/r
    f1 = a/t
    print("Reserved GPU Memory Full: "+"{:.2%}".format(f0), " Total GPU Memory Full"+"{:.2%}".format(f1))
