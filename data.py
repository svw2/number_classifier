import imageio
import torch
import numpy as np
class Image_Dataset:
    def __init__(self, path, dstype="train"):
        self.path = path
        self.name_to_number = {}
        self.filenames = []
        self.unumbers = []
        with open(path+"/numbers.csv","r") as f:
            for linenum,line in enumerate(f):
                if(linenum == 0):
                    continue
                line = line.strip("\n")
                index = 0
                if(dstype not in line):
                    continue
                for i in line:
                    if(i == ","):
                        break
                    index+=1
                fn = path + "/" + line[0:index]
                number = line[index+1:]
                number = int(number)
                self.name_to_number[fn] = number
                if(number not in self.unumbers):
                    self.unumbers.append(number)
                self.filenames.append(fn)


    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        image = imageio.v2.imread(self.filenames[index])
        image = torch.tensor(np.array(image/255.0,dtype = np.float64)).float()[None]
        number = self.name_to_number[self.filenames[index]]   
        n = self.unumbers.index(number)
        return image,torch.tensor(n).long()
    

                
                        




                

            